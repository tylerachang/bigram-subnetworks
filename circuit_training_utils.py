"""
Utils for circuit training. Note that the full experiment code for the paper
is in the paper_code directory, but it is much less clean. Circuit training
code builds on the continuous_sparsification code from:
https://github.com/mlepori1/NeuroSurgeon

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


def get_mask_dict(circuit_model1, orig_model_name=None):
    mask_dict = dict()
    param_names = sorted(list(circuit_model1.state_dict().keys()))
    n_kept = 0
    n_total = 0
    for p in param_names:
        if 'weight_mask_params' in p:
            param_mask = np.array(circuit_model1.state_dict()[p].cpu())
            param_mask = param_mask > 0.0
            orig_name = p.replace('.weight_mask_params', '.weight').replace('wrapped_model.', '')
            mask_dict[orig_name] = param_mask
            n_kept += np.sum(param_mask)
            n_total += len(param_mask.flatten())
    proportion_kept = n_kept / n_total
    print(f'Computed mask with proportion kept: {proportion_kept}')
    if orig_model_name is not None:
        mask_dict['orig_model_name'] = orig_model_name
    return mask_dict, proportion_kept


def select_layers(layer_names, model_type, print_out=False):
    target_layers = None  # Placeholder.
    assert model_type in ['gpt2', 'gptneox']
    # We want to target all attention and mlp layers.
    if model_type == 'gpt2':
        target_layers = [
            '.'.join(target_layer.split('.')[:-1])
            for target_layer in layer_names
            if ('.h.' in target_layer
            and 'weight' in target_layer
            and 'ln' not in target_layer)
        ]
    elif model_type == 'gptneox':
        target_layers = [
            '.'.join(target_layer.split('.')[:-1])
            for target_layer in layer_names
            if ('embed' not in target_layer
            and 'weight' in target_layer
            and 'layernorm' not in target_layer
            and 'final_layer_norm' not in target_layer)
        ]
    if print_out:
        print('Target layers for subnetwork training:')
        print(target_layers)
        print('Not considering for subnetwork training (will always be kept):')
        print([l for l in layer_names
               if '.'.join(l.split('.')[:-1]) not in target_layers])
    return target_layers


"""
Train a subnetwork (circuit). Assumes there is given some target_fn that outputs
next token probabilities that the subnetwork is trying to recreate (minimizes
cross-entropy loss). The target_fn should have inputs and outputs with the same
format as the original model, but output probabilities instead of logits.

Outputs the trained subnetwork dictionary to a .pickle file. The inputs_path
should contain tokenized input examples in a txt file. Each line should be a
space-separated list of int token ids.

This function can be modified to use a custom loss function, but currently
it defaults to cross-entropy between the subnetwork and target_fn (with a
sparsity term lambda as described in the paper).

Note: max_steps counts every batch, not accounting for gradient accumulation.
I.e. if you increase grad_acc_steps, you should also increase max_steps
proportionally.
"""
def train_circuit(model_dir, target_fn, inputs_path, outpath,
                  batch_size=32, temperature_change=1.001, max_steps=15000,
                  l0_lambda=1e-1, learning_rate=0.00005,
                  grad_acc_steps=1, shuffle_train=True, hf_cache='hf_cache'):
    assert outpath.endswith('.pickle')
    if os.path.isfile(outpath):
        print(f'Circuit outpath already exists: {outpath}')
        return False
    # Load model.
    model1 = AutoModelForCausalLM.from_pretrained(model_dir, cache_dir=hf_cache)
    layer_names = list(model1.state_dict().keys())
    model_type = 'gptneox' if isinstance(model1, transformers.GPTNeoXForCausalLM)
    model_type = 'gpt2' if isinstance(model1, transformers.GPT2LMHeadModel)
    target_layers = select_layers(layer_names, model_type, print_out=True)
    # Create circuit model.
    circuit_config = model_configs.CircuitConfig(
        mask_method='continuous_sparsification',
        mask_hparams = {
            'ablation': 'none', # Don't invert the learned mask.
            'mask_unit': 'weight', # Mask at the weight-level.
            'mask_bias': False, # Don't mask biases.
            'mask_init_value': 0.0 # Initialize the mask parameters at 0 (0.50 after sigmoid).
        },
        target_layers=target_layers,
        freeze_base=True, # Don't train the model weights when training the mask.
        add_l0=False, # Use L0 Regularization. Unused param, manually added instead.
        l0_lambda=l0_lambda, # Multiplier on L0 norm.
    )
    circuit_model1 = circuit_model.CircuitModel(circuit_config, model1)
    if torch.cuda.is_available():
        circuit_model1 = circuit_model1.cuda()
    optimizer = torch.optim.AdamW(circuit_model1.parameters(), lr=learning_rate)
    # Count trainable mask parameters.
    n_trainable = 0
    state_dict = circuit_model1.state_dict()
    for param_name in list(state_dict.keys()):
        if 'weight_mask_params' in param_name:
            n_params = len(circuit_model1.state_dict()[param_name].flatten())
            n_trainable += n_params
    # Temperature decrease per step.
    class TemperatureCallback:
        # Mask parameter is multiplied by temperature before the sigmoid,
        # instead of divided, so this increase is really more of a *decrease*.
        def __init__(self, temperature_change):
            self.temperature_change = temperature_change
        def update(self, model):
            temp = model.temperature
            model.temperature = temp * self.temperature_change
    temp_callback = TemperatureCallback(temperature_change)
    # Read input examples for training.
    def read_examples(inpath):
        ds = []
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        for l in infile:
            example = [int(token_id) for token_id in l.strip().split()]
            ds.append(example)
        infile.close()
        return ds
    train_dataset = read_examples(inputs_path)
    assert len(train_dataset) >= (max_steps * batch_size)
    if shuffle_train:
        random.shuffle(train_dataset)
    # Define loss.
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # Run continuous sparsification.
    xent_losses = []
    batch_xent = 0.0
    for step in tqdm(range(max_steps)):
        # Get inputs. Assume sequences of same length in input examples.
        # Dummy inputs:
        # input_ids = torch.Tensor(np.zeros((32, 128))).int().cuda()
        input_ids = train_dataset[batch_size*step:batch_size*(step+1)]
        input_ids = torch.Tensor(input_ids).int()
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
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
            # parameters can be "undecided".
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
    # Save circuit.
    mask_dict, proportion_kept = get_mask_dict(circuit_model1, orig_model_name=model_dir)
    with open(outpath, 'wb') as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # For logging:
    json_outpath = outpath.replace('.pickle', '.json')
    out_dict = dict()
    out_dict['outpath'] = outpath
    out_dict['model_dir'] = model_dir
    out_dict['train_inputs'] = inputs_path
    out_dict['train_steps'] = step+1
    out_dict['max_steps'] = max_steps
    out_dict['grad_acc_steps'] = grad_acc_steps
    out_dict['learning_rate'] = learning_rate
    out_dict['batch_size'] = batch_size
    out_dict['temperature_change'] = temperature_change
    out_dict['l0_lambda'] = l0_lambda
    out_dict['n_trainable'] = n_trainable
    out_dict['target_layers'] = target_layers
    out_dict['xent_loss_per_step'] = xent_losses
    out_dict['proportion_kept'] = proportion_kept
    with codecs.open(json_outpath, 'wb', encoding='utf-8') as f:
        f.write(json.dumps(out_dict) + '\n')
    print('Saved circuit.')
    return circuit_model1
