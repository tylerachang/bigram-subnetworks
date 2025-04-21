"""
Utils for circuit training and analysis.
"""

import codecs
import json
import numpy as np
import os
import pickle
import random
import torch
from tqdm import tqdm

import NeuroSurgeon
from NeuroSurgeon.Models import model_configs, circuit_model
from NeuroSurgeon.Masking import mask_layer

from utils.model_utils import load_model, read_examples


def get_mask_dict(circuit_model1, orig_model_name=None):
    mask_dict = dict()
    param_names = sorted(list(circuit_model1.state_dict().keys()))
    n_kept = 0
    n_total = 0
    for p in param_names:
        if 'weight_mask_params' in p:
            param_mask = np.array(circuit_model1.state_dict()[p].cpu())
            param_mask = param_mask > 0.0
            updated_name = p.replace('.weight_mask_params', '.weight').replace('wrapped_model.', '')
            mask_dict[updated_name] = param_mask
            n_kept += np.sum(param_mask)
            n_total += len(param_mask.flatten())
    proportion_kept = n_kept / n_total
    print(f'Computed mask with proportion kept: {proportion_kept}')
    if orig_model_name is not None:
        mask_dict['orig_model_name'] = orig_model_name
    return mask_dict, proportion_kept


def select_layers(layer_names, model_type, print_out=False):
    target_layers = None  # Placeholder.
    assert model_type in ['gpt2', 'pythia', 'olmo']
    if model_type == 'gpt2':
        # We want to target all attention and mlp layers in the GPT2 architecture,
        # as well as the LM Head.
        target_layers = [
            '.'.join(target_layer.split('.')[:-1])
            for target_layer in layer_names
            if ('.h.' in target_layer
            and 'weight' in target_layer
            and 'ln' not in target_layer)
            # or ('lm_head' in target_layer)  # Comment in to include unembedding layer.
        ]
    elif model_type == 'pythia':
        target_layers = [
            '.'.join(target_layer.split('.')[:-1])
            for target_layer in layer_names
            if ('embed' not in target_layer
            and 'weight' in target_layer
            and 'layernorm' not in target_layer
            and 'final_layer_norm' not in target_layer)
        ]
    elif model_type == 'olmo':
        # No layer norm params.
        target_layers = [
            '.'.join(target_layer.split('.')[:-1])
            for target_layer in layer_names
            if ('.layers.' in target_layer
            and 'weight' in target_layer)
        ]
    if print_out:
        print('Target layers:')
        print(target_layers)
        print('Excluded layers:')
        print([l for l in layer_names
               if '.'.join(l.split('.')[:-1]) not in target_layers])
    return target_layers


"""
Checks whether the xent loss curve was stable.

Checks whether:
The loss at every step is not add_threshold greater or mult_threshold times
greater than the mean of the previous n_prev steps. If this occurs, that
mean must be re-attained at least reattain_threshold steps before the end of
training.
"""
def check_loss_stability(losses):
    n_prev = 10  # Consider the past n steps.
    add_threshold = 0.25  # Warn if loss increase greater than this (absolute).
    mult_threshold = 1.25  # Warn if loss increase greater than this (multiplicative).
    reattain_threshold = 100
    # Check stability.
    is_stable = True
    prev = [losses[0]]
    for i, loss in enumerate(losses):
        if i == 0: continue  # First value already included.
        prev_mean = np.mean(prev)
        spike = False
        if loss > prev_mean + add_threshold: spike = True
        if loss > prev_mean * mult_threshold: spike = True
        if spike:
            if i >= (len(losses) - reattain_threshold):
                is_stable = False
                break
            if np.min(losses[i:len(losses)-reattain_threshold]) > prev_mean:
                # Losses do not re-attain the previous mean at least
                # reattain_threshold steps before the end of training.
                is_stable = False
                break
        # Update previous losses.
        prev.append(loss)
        if len(prev) > n_prev: prev.pop(0)
    return is_stable
"""
For a looser criterion (unused), checks whether:
If the loss at some step is large_threshold greater than the mean of the
previous n_prev steps, then that mean must be re-attained at least
reattain_steps before the end of training (note this can be difficult because
xent sometimes increases slowly to accomodate the L1 loss penalty and
temperature increase).
If the loss at some step is small_threshold greater than the mean of the
previous n_prev steps, then this just must be further than end_step_threshold
from the end of training (or mean re-attained as before).
"""
def check_loss_stability_loose(losses):
    n_prev = 10  # Consider the past n steps.
    large_threshold = 1.0  # Warn if loss increase greater than this (absolute).
    reattain_steps = 100
    small_threshold = 0.25
    end_step_threshold = 500
    # Check stability.
    is_stable = True
    prev = [losses[0]]
    for i, loss in enumerate(losses):
        if i == 0: continue  # First value already included.
        prev_mean = np.mean(prev)
        if loss > prev_mean + large_threshold:
            if i >= (len(losses) - reattain_steps):
                is_stable = False
                break
            if np.min(losses[i:len(losses)-reattain_steps]) > prev_mean:
                # Losses do not re-attain the previous mean at least
                # reattain_steps steps before the end of training.
                is_stable = False
                break
        if loss > prev_mean + small_threshold:
            if i >= (len(losses) - reattain_steps):
                is_stable = False
                break
            if np.min(losses[i:len(losses)-reattain_steps]) <= prev_mean:
                # OK: loss is re-attained.
                pass
            elif i >= (len(losses) - end_step_threshold):
                # Not re-attained, and not within end_step_threshold.
                is_stable = False
                break
        # Update previous losses.
        prev.append(loss)
        if len(prev) > n_prev: prev.pop(0)
    return is_stable


# Note: l0_lambda is also normalized by number of trainable mask parameters.
# temp_increase is the multiplicative increase in temperature per optimizer step.
# Mask parameter is multiplied by temperature before the sigmoid, instead
# of divided, so it really is more of a temperature *decrease*.
# max_steps counts every batch, not accounting for gradient accumulation;
# similarly for save_every (i.e. if you increase grad_acc_steps, you should also
# increase max_steps proportionally).
def train_circuit(model_dir, target_fn, inputs_path, outpath,
                  model_type='gpt2', checkpoint=None,
                  batch_size=32, temp_increase=1.001, max_steps=10000,
                  l0_lambda=1e-2, learning_rate=0.0001, save_every=None,
                  grad_acc_steps=1, shuffle_train=False):
    assert outpath.endswith('.pickle')
    if os.path.isfile(outpath):
        print(f'Circuit outpath already exists: {outpath}')
        return False
    _, _, model1 = load_model(
            model_dir, model_type, checkpoint=checkpoint,
            tokenizer_path_override=None, config_path_override=None,
            cache_dir='hf_cache', override_for_hidden_states=False)
    layer_names = list(model1.state_dict().keys())
    target_layers = select_layers(layer_names, model_type, print_out=False)
    # Circuit config.
    circuit_config = model_configs.CircuitConfig(
        mask_method='continuous_sparsification', # Binary Mask Optimization method
        mask_hparams = {
            'ablation': 'none', # Don't invert the learned mask
            'mask_unit': 'weight', # Mask at the weight-level
            'mask_bias': False, # Don't mask biases
            'mask_init_value': 0.0 # Initialize the mask parameters at 0 (0.50 after sigmoid). To start closer to original model, use 5.0.
        },
        target_layers=target_layers, # Replace the layers specified above
        freeze_base=True, # Don't train the model weights when training the mask
        add_l0=True, # Use L0 Regularization. Unused param, manually added instead.
        l0_lambda=l0_lambda, # Multiplier on L0 norm for balancing the loss function
    )
    circuit_model1 = circuit_model.CircuitModel(circuit_config, model1).to('cuda')
    optimizer = torch.optim.AdamW(circuit_model1.parameters(), lr=learning_rate)
    # Count trainable mask parameters.
    n_trainable = 0
    state_dict = circuit_model1.state_dict()
    for param_name in list(state_dict.keys()):
        if 'weight_mask_params' in param_name:
            n_params = len(circuit_model1.state_dict()[param_name].flatten())
            n_trainable += n_params
            # If some model parameters are zero already, set that mask to zero
            # (negative large number before sigmoid).
            orig_name = param_name.replace('.weight_mask_params', '.weight')
            orig_weights = state_dict[orig_name]
            state_dict[param_name][orig_weights == 0.0] = -1000000.0
    circuit_model1.load_state_dict(state_dict)  # Update model with state dict.
    # If l0_lambda is inf, just an empty or full circuit.
    if np.isinf(l0_lambda):
        mask_dict, proportion_kept = get_mask_dict(circuit_model1, orig_model_name=model_dir)
        if l0_lambda < 0.0:  # Negative infinity: keep all parameters.
            print('Negative infinity l0_lambda: keeping all parameters.')
            for k in list(mask_dict.keys()):
                mask_dict[k] = np.ones_like(mask_dict[k])
            proportion_kept = 1.0
        elif l0_lambda > 0.0:  # Positive infinity: drop all parameters.
            print('Positive infinity l0_lambda: dropping all parameters.')
            for k in list(mask_dict.keys()):
                mask_dict[k] = np.zeros_like(mask_dict[k])
            proportion_kept = 0.0
        with open(outpath, 'wb') as handle:
            pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # For logging:
        json_outpath = outpath.replace('.pickle', '.json')
        out_dict = dict()
        out_dict['outpath'] = outpath
        out_dict['model_dir'] = model_dir
        out_dict['checkpoint'] = checkpoint
        out_dict['train_inputs'] = None
        out_dict['train_steps'] = None
        out_dict['max_steps'] = None
        out_dict['grad_acc_steps'] = None
        out_dict['learning_rate'] = None
        out_dict['batch_size'] = None
        out_dict['temp_increase'] = None
        out_dict['l0_lambda'] = l0_lambda
        out_dict['mask_size'] = n_trainable
        out_dict['target_layers'] = target_layers
        out_dict['xent_loss_per_step'] = None
        out_dict['proportion_kept'] = proportion_kept
        with codecs.open(json_outpath, 'wb', encoding='utf-8') as f:
            f.write(json.dumps(out_dict) + '\n')
        print('Saved circuit.')
        return None
    # Now, actually train a circuit.
    # For continuous sparsification.
    class TemperatureCallback:
        # A simple callback that updates the probes temperature parameter,
        # which transforms a soft mask into a hard mask
        def __init__(self, temp_increase):
            # To use final temp setting:
            # self.temp_increase = final_temp ** (1.0 / total_steps)
            self.temp_increase = temp_increase
        def update(self, model):
            temp = model.temperature
            model.temperature = temp * self.temp_increase
    # Define loss and train dataset.
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    train_dataset = read_examples(inputs_path)
    assert len(train_dataset) >= (max_steps * batch_size)
    if shuffle_train:
        random.shuffle(train_dataset)
    # Run continuous sparsification.
    xent_losses = []
    batch_xent = 0.0
    temp_callback = TemperatureCallback(temp_increase)
    for step in tqdm(range(max_steps)):
        # Get inputs. Assume sequences of same length in input examples.
        # Dummy inputs:
        # input_ids = torch.Tensor(np.zeros((32, 128))).int().cuda()
        input_ids = train_dataset[batch_size*step:batch_size*(step+1)]
        input_ids = torch.Tensor(input_ids).int().cuda()
        # Compute (masked) model logits and target probs.
        logits = circuit_model1(input_ids=input_ids).logits
        target_probs = torch.Tensor(target_fn(input_ids))
        # May need to remove dims from logits if tokenizer has extra tokens
        # added (e.g. padding vocab to multiple of power of two). This makes
        # logits dims match target_probs.
        vocab_size = target_probs.size()[-1]
        if logits.size()[-1] != vocab_size:
            logits = logits[:, :, :vocab_size]
        # Note: transpose because the class labels/probabilities must be index 1.
        xent_loss = loss_fn(logits.transpose(-1, 1), target_probs.transpose(-1, 1).cuda())
        loss = xent_loss + \
                ((1.0 / n_trainable) * circuit_config.l0_lambda * circuit_model1._compute_l0_loss()) # Manually adding L0 Loss
        loss = loss / grad_acc_steps
        loss.backward()
        batch_xent += xent_loss.detach().cpu().item() / grad_acc_steps
        if (step+1) % grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # Check proportion of parameters with mask above 0.9 or below 0.1.
            # Stop only if the number of remaining parameters is less than 1% of
            # the learned circuit. Rather than setting a constant threshold,
            # this ensures that if a smaller circuit is learned, fewer
            # "undecided" parameters are allowed. Still, at most 1% of
            # parameters can be "undecided" (a circuit of size 100%).
            n_below = 0
            n_above = 0
            for _, layer in circuit_model1.wrapped_model.named_modules():
                if issubclass(type(layer), mask_layer.MaskLayer):
                    layer_mask = layer._compute_mask('weight_mask_params')
                    n_below += torch.sum(layer_mask < 0.1).item()
                    n_above += torch.sum(layer_mask > 0.9).item()
            n_remaining = n_trainable - (n_below + n_above)
            if n_remaining < 0.01 * n_above: break
            # Increase temperature each optimizer step.
            temp_callback.update(circuit_model1)
            xent_losses.append(batch_xent)
            batch_xent = 0.0
        # Printing of most recent values.
        if (step+1) % 100 == 0:
            print(f'step: {step+1}, proportion_undecided: {n_remaining/n_trainable}, '
                  + f'xent_loss: {xent_loss.cpu().item()}')
        # Save.
        if (save_every is not None) and ((step+1) % save_every == 0):
            mask_dict, proportion_kept = get_mask_dict(circuit_model1, orig_model_name=model_dir)
            with open(outpath + f'_step{step+1}', 'wb') as handle:
                pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del mask_dict
            print('Saved mask.')
    mask_dict, proportion_kept = get_mask_dict(circuit_model1, orig_model_name=model_dir)
    # Save circuit.
    with open(outpath, 'wb') as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # For logging:
    json_outpath = outpath.replace('.pickle', '.json')
    out_dict = dict()
    out_dict['outpath'] = outpath
    out_dict['model_dir'] = model_dir
    out_dict['checkpoint'] = checkpoint
    out_dict['train_inputs'] = inputs_path
    out_dict['train_steps'] = step+1
    out_dict['max_steps'] = max_steps
    out_dict['grad_acc_steps'] = grad_acc_steps
    out_dict['learning_rate'] = learning_rate
    out_dict['batch_size'] = batch_size
    out_dict['temp_increase'] = temp_increase
    out_dict['l0_lambda'] = l0_lambda
    out_dict['mask_size'] = n_trainable
    out_dict['target_layers'] = target_layers
    out_dict['xent_loss_per_step'] = xent_losses
    out_dict['proportion_kept'] = proportion_kept
    with codecs.open(json_outpath, 'wb', encoding='utf-8') as f:
        f.write(json.dumps(out_dict) + '\n')
    print('Saved circuit.')
    return circuit_model1


def load_circuit_model(model_dir, circuit_path, model_type, checkpoint=None,
                       use_complement=False, randomize=False):
    # Load original model.
    config1, tokenizer1, model1 = load_model(
            model_dir, model_type, checkpoint=checkpoint,
            tokenizer_path_override=None, config_path_override=None,
            cache_dir='hf_cache')
    if not circuit_path: return config1, tokenizer1, model1
    layer_names = list(model1.state_dict().keys())
    target_layers = select_layers(layer_names, model_type)
    # Load mask.
    with open(circuit_path, 'rb') as f:
        circuit_dict = pickle.load(f)
    # Get an update state dict.
    state_dict = model1.state_dict()
    for l in target_layers:
        # Format: wrapped_model.x.y.z.weight_mask_params
        # circuit_key = 'wrapped_model.' + l + '.weight_mask_params'  # FROM OLD CODE.
        circuit_key = l+'.weight'
        # Optionally use complement of the circuit.
        if use_complement:  # Use all parameters except the circuit.
            mask = ~circuit_dict[circuit_key]
        else:
            mask = circuit_dict[circuit_key]
        # Optionally randomize to use a random sample of the same size within
        # each parameter block.
        if randomize:
            mask_shape = mask.shape
            mask = mask.flatten()
            np.random.shuffle(mask)
            mask = mask.reshape(*mask_shape)
        # Format: x.y.z.weight
        state_dict[l+'.weight'] = (state_dict[l+'.weight'].cpu() * mask).cuda()
    model1.load_state_dict(state_dict)  # Update model with state dict.
    # Config and tokenizer unchanged.
    return config1, tokenizer1, model1


# Converts the parameter mask keys to [att_[qkvo]/mlp_[0/1]]_[layer].
# Each weight matrix is shape (out_dims, in_dims).
def convert_mask_names(circuit_dict, model_type):
    new_dict = dict()
    layer = 0
    while True:
        if model_type == 'gpt2':
            # Check if finished all layers.
            if f'transformer.h.{layer}.attn.c_attn.weight' not in circuit_dict:
                break
            # Attention.
            c_attn = circuit_dict[f'transformer.h.{layer}.attn.c_attn.weight']
            assert c_attn.shape[1] % 3 == 0  # This dimension should be divisible by 3.
            n_dims = c_attn.shape[1] // 3
            att_q = c_attn[:, :n_dims].T  # Transpose because HF GPT-2 implementation uses (in,out) shape.
            att_k = c_attn[:, n_dims:2*n_dims].T
            att_v = c_attn[:, 2*n_dims:3*n_dims].T
            # In this case (GPT-2), this should be square.
            assert att_q.shape[0] == att_q.shape[1]
            att_o = circuit_dict[f'transformer.h.{layer}.attn.c_proj.weight'].T
            # MLP.
            mlp_0 = circuit_dict[f'transformer.h.{layer}.mlp.c_fc.weight'].T
            mlp_1 = circuit_dict[f'transformer.h.{layer}.mlp.c_proj.weight'].T
        elif model_type == 'pythia':
            # Check if finished all layers.
            if f'gpt_neox.layers.{layer}.attention.query_key_value.weight' not in circuit_dict:
                break
            # Attention.
            att = circuit_dict[f'gpt_neox.layers.{layer}.attention.query_key_value.weight']
            assert att.shape[0] % 3 == 0  # This dimension should be divisible by 3.
            n_dims = att.shape[0] // 3
            att_q = att[:n_dims, :]  # Do not transpose because the implementation uses (out,in) shape.
            att_k = att[n_dims:2*n_dims, :]
            att_v = att[2*n_dims:3*n_dims, :]
            # In this case (Pythia), this should be square.
            assert att_q.shape[0] == att_q.shape[1]
            att_o = circuit_dict[f'gpt_neox.layers.{layer}.attention.dense.weight']
            # MLP.
            mlp_0 = circuit_dict[f'gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight']
            mlp_1 = circuit_dict[f'gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight']
        else:
            # Not implemented for other model types.
            # This will cause errors!
            break
        # Update dict.
        new_dict[f'att_q_{layer}'] = att_q
        new_dict[f'att_k_{layer}'] = att_k
        new_dict[f'att_v_{layer}'] = att_v
        new_dict[f'att_o_{layer}'] = att_o
        new_dict[f'mlp_0_{layer}'] = mlp_0
        new_dict[f'mlp_1_{layer}'] = mlp_1
        layer += 1  # Increment.
    return new_dict


def get_circuit_overlap_summary(circuit_path1, circuit_path2, model_type):
    with open(circuit_path1, 'rb') as f:
        circuit_dict1 = pickle.load(f)
    with open(circuit_path2, 'rb') as f:
        circuit_dict2 = pickle.load(f)
    summary = ''
    circuit_dict1 = convert_mask_names(circuit_dict1, model_type)
    circuit_dict2 = convert_mask_names(circuit_dict2, model_type)
    n_layers = len(circuit_dict1) // 6
    all_params1 = []
    all_params2 = []
    for layer in range(n_layers):
        attention1 = np.concatenate([
                circuit_dict1[f'att_q_{layer}'].flatten(), circuit_dict1[f'att_k_{layer}'].flatten(),
                circuit_dict1[f'att_v_{layer}'].flatten(), circuit_dict1[f'att_o_{layer}'].flatten(),
                ])
        mlp1 = np.concatenate([
                circuit_dict1[f'mlp_0_{layer}'].flatten(), circuit_dict1[f'mlp_1_{layer}'].flatten(),
                ])
        attention2 = np.concatenate([
                circuit_dict2[f'att_q_{layer}'].flatten(), circuit_dict2[f'att_k_{layer}'].flatten(),
                circuit_dict2[f'att_v_{layer}'].flatten(), circuit_dict2[f'att_o_{layer}'].flatten(),
                ])
        mlp2 = np.concatenate([
                circuit_dict2[f'mlp_0_{layer}'].flatten(), circuit_dict2[f'mlp_1_{layer}'].flatten(),
                ])
        shared_attention = round(np.mean(attention1 & attention2), 3)
        only1_attention = round(np.mean(attention1 & ~attention2), 3)
        only2_attention = round(np.mean(~attention1 & attention2), 3)
        summary += f'Layer {layer}, attention: {shared_attention} shared, {only1_attention} only1, {only2_attention} only2\n'
        shared_mlp = round(np.mean(mlp1 & mlp2), 3)
        only1_mlp = round(np.mean(mlp1 & ~mlp2), 3)
        only2_mlp = round(np.mean(~mlp1 & mlp2), 3)
        summary += f'Layer {layer}, mlp: {shared_mlp} shared, {only1_mlp} only1, {only2_mlp} only2\n'
        all_params1.extend([attention1, mlp1])
        all_params2.extend([attention2, mlp2])
    all_params1 = np.concatenate(all_params1)
    all_params2 = np.concatenate(all_params2)
    shared = round(np.mean(all_params1 & all_params2), 3)
    only1 = round(np.mean(all_params1 & ~all_params2), 3)
    only2 = round(np.mean(~all_params1 & all_params2), 3)
    summary += f'Overall: {shared} shared, {only1} only1, {only2} only2\n'
    return summary


# Optionally split by the number of attention heads (attention_heads).
def get_circuit_summary(circuit_path, model_type, attention_heads=None):
    with open(circuit_path, 'rb') as f:
        circuit_dict = pickle.load(f)
    summary = 'Parameters kept:\n'
    circuit_dict = convert_mask_names(circuit_dict, model_type)
    n_layers = len(circuit_dict) // 6
    all_params = []
    for layer in range(n_layers):
        # Attention.
        attention = np.concatenate([
                circuit_dict[f'att_q_{layer}'].flatten(), circuit_dict[f'att_k_{layer}'].flatten(),
                circuit_dict[f'att_v_{layer}'].flatten(), circuit_dict[f'att_o_{layer}'].flatten(),
                ])
        summary += f'Layer {layer}, attention: {round(np.mean(attention), 3)}\n'
        # Optionally split by attention heads.
        if attention_heads is not None:
            n_dims = circuit_dict[f'att_q_{layer}'].shape[-1]
            assert n_dims % attention_heads == 0
            head_dims = n_dims // attention_heads
            for head_i in range(attention_heads):
                dims = np.arange(head_i*head_dims, (head_i+1)*head_dims)
                head_params = np.concatenate([
                        circuit_dict[f'att_q_{layer}'][dims, :].flatten(),
                        circuit_dict[f'att_k_{layer}'][dims, :].flatten(),
                        circuit_dict[f'att_v_{layer}'][dims, :].flatten(),
                        circuit_dict[f'att_o_{layer}'][:, dims].flatten(),
                        ])
                summary += f'  Attention head {head_i}: {round(np.mean(head_params), 4)}\n'
        # MLP.
        mlp = np.concatenate([
                circuit_dict[f'mlp_0_{layer}'].flatten(), circuit_dict[f'mlp_1_{layer}'].flatten(),
                ])
        summary += f'Layer {layer}, mlp: {round(np.mean(mlp), 3)}\n'
        all_params.extend([attention, mlp])
    all_params = np.concatenate(all_params)
    summary += f'Overall: {round(np.mean(all_params), 3)}\n'
    return summary
