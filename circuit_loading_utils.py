"""
Utils for circuit loading and modifying.
"""

import huggingface_hub
import numpy as np
import pickle
import transformers


# Default bigram subnetwork listed first.
RELEASED_BIGRAM_SUBNETWORKS = {
        'EleutherAI/pythia-70m': ['lambda1e1', 'lambda1'],
        'EleutherAI/pythia-160m': ['lambda1e1', 'lambda1'],
        'EleutherAI/pythia-410m': ['lambda1e1', 'lambda1'],
        'EleutherAI/pythia-1b': ['lambda1e2', 'lambda1e1', 'lambda1'],
        'openai-community/gpt2': ['lambda1e1', 'lambda1'],
        'openai-community/gpt2-medium': ['lambda1e1', 'lambda1'],
        'openai-community/gpt2-large': ['lambda1e2', 'lambda1e1', 'lambda1'],
}


# Load a bigram subnetwork dictionary, mapping original model parameter names
# to numpy binary masks with the same shapes as the original parameters.
def load_bigram_subnetwork_dict(model_name, lambda_str=None):
    # Check that we have the model supported.
    assert model_name in RELEASED_BIGRAM_SUBNETWORKS, 'Model not supported.'
    # Use default lambda unless otherwise specified.
    if lambda_str is None: lambda_str = RELEASED_BIGRAM_SUBNETWORKS[model_name][0]
    if not lambda_str.startswith('lambda'): lambda_str = 'lambda' + lambda_str
    assert lambda_str in RELEASED_BIGRAM_SUBNETWORKS[model_name], 'Model not supported for this lambda value.'
    # Download subnetwork dictionary pickle file from Hugging Face.
    model_lastname = model_name.split('/')[-1]
    if model_lastname == 'gpt2': model_lastname = 'gpt2-small'  # On our repository, we include the -small suffix for clarity.
    repo_id = f'tylerachang/bigram-subnetworks-{model_lastname}'
    downloaded_path = huggingface_hub.hf_hub_download(repo_id=repo_id,
            filename=f'subnetwork_dict_{lambda_str}.pickle')
    # Load downloaded subnetwork dictionary.
    with open(downloaded_path, 'rb') as f:
        mask_dict = pickle.load(f)
    return mask_dict


# Load an empty subnetwork dictionary.
# Note that embedding and layernorm parameters will still be included.
def load_empty_subnetwork_dict(model_name):
    # Need to load the original bigram subnetwork to get keys and parameter shapes.
    mask_dict = load_bigram_subnetwork_dict(model_name)
    for k in list(mask_dict.keys()):
        if k == 'orig_model_name': continue  # Not an actual layer.
        mask_dict[k][:, :] = False
    return mask_dict


# Load a full subnetwork dictionary.
def load_full_subnetwork_dict(model_name):
    # Need to load the original bigram subnetwork to get keys and parameter shapes.
    mask_dict = load_bigram_subnetwork_dict(model_name)
    for k in list(mask_dict.keys()):
        if k == 'orig_model_name': continue  # Not an actual layer.
        mask_dict[k][:, :] = True
    return mask_dict


# In a subnetwork dictionary, set an attention head to be kept or dropped.
def set_attention_head(mask_dict, layer, head_index, to_keep=True, cache_dir='hf_cache'):
    orig_model_name = mask_dict['orig_model_name']
    config = transformers.AutoConfig.from_pretrained(orig_model_name, cache_dir=cache_dir)
    if 'GPTNeoXForCausalLM' in config.architectures:
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        assert head_index < num_heads, 'Attention head index out of range.'
        head_size = hidden_size // num_heads
        # Shape: out_dims, in_dims.
        mask_dict[f'gpt_neox.layers.{layer}.attention.query_key_value.weight'][head_size*head_index:head_size*(head_index+1), :] = to_keep
        mask_dict[f'gpt_neox.layers.{layer}.attention.query_key_value.weight'][hidden_size+head_size*head_index:hidden_size+head_size*(head_index+1), :] = to_keep
        mask_dict[f'gpt_neox.layers.{layer}.attention.query_key_value.weight'][2*hidden_size+head_size*head_index:2*hidden_size+head_size*(head_index+1), :] = to_keep
        # Also shape: out_dims, in_dims.
        mask_dict[f'gpt_neox.layers.{layer}.attention.dense.weight'][:, head_size*head_index:head_size*(head_index+1)] = to_keep
    elif 'GPT2LMHeadModel' in config.architectures:
        hidden_size = config.n_embd
        num_heads = config.n_head
        assert head_index < num_heads, 'Attention head index out of range.'
        head_size = hidden_size // num_heads
        # Shape: in_dims, out_dims.
        mask_dict[f'transformer.h.{layer}.attn.c_attn.weight'][:, head_size*head_index:head_size*(head_index+1)] = to_keep
        mask_dict[f'transformer.h.{layer}.attn.c_attn.weight'][:, hidden_size+head_size*head_index:hidden_size+head_size*(head_index+1)] = to_keep
        mask_dict[f'transformer.h.{layer}.attn.c_attn.weight'][:, 2*hidden_size+head_size*head_index:2*hidden_size+head_size*(head_index+1)] = to_keep
        # Also shape: in_dims, out_dims.
        mask_dict[f'transformer.h.{layer}.attn.c_proj.weight'][head_size*head_index:head_size*(head_index+1), :] = to_keep
    else:
        print('ERROR: model type not supported.')
    return mask_dict  # Also modifies the original dictionary directly.


# In a subnetwork dictionary, set an MLP dimension to be kept or dropped.
# This treats MLP layers as key-value memories as in Geva et al. (2021).
def set_mlp_dimensions(mask_dict, layer, mlp_dimensions, to_keep=True, cache_dir='hf_cache'):
    assert hasattr(mlp_dimensions, '__iter__'), 'Please provide mlp_dimensions as a list.'
    orig_model_name = mask_dict['orig_model_name']
    config = transformers.AutoConfig.from_pretrained(orig_model_name, cache_dir=cache_dir)
    if 'GPTNeoXForCausalLM' in config.architectures:
        # Shape: out_dims, in_dims.
        mask_dict[f'gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight'][mlp_dimensions, :] = to_keep
        mask_dict[f'gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight'][:, mlp_dimensions] = to_keep
    elif 'GPT2LMHeadModel' in config.architectures:
        # Shape: in_dims, out_dims.
        mask_dict[f'transformer.h.{layer}.mlp.c_fc.weight'][:, mlp_dimensions] = to_keep
        mask_dict[f'transformer.h.{layer}.mlp.c_proj.weight'][mlp_dimensions, :] = to_keep
    else:
        print('ERROR: model type not supported.')
    return mask_dict  # Also modifies the original dictionary directly.


# Load a model given a subnetwork dictionary.
# The model parameters are kept according to the subnetwork mask.
def load_subnetwork_model(model_name, mask_dict, cache_dir='hf_cache'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    config = transformers.AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    # Update model state dict.
    state_dict = model.state_dict()
    for layer_name, mask in mask_dict.items():
        if layer_name == 'orig_model_name': continue  # Not an actual layer.
        state_dict[layer_name] = state_dict[layer_name] * mask
    model.load_state_dict(state_dict)  # Update model with state dict.
    # Config and tokenizer unchanged.
    return model, tokenizer, config


# Return the number of active and total parameters in the mask.
def get_parameter_count(mask_dict):
    n_active = 0
    n_total = 0
    for k, v in mask_dict.items():
        if k == 'orig_model_name': continue  # Not an actual layer.
        n_active += np.sum(v)
        n_total += len(v.flatten())
    return n_active, n_total
