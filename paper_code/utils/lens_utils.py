"""
Utils for tuned lens analyses.
"""

import codecs
import os
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression, Ridge
import torch

from utils.model_utils import get_hidden_states, read_examples, load_model, get_token_embeddings
from utils.model_overrides import GPT2LensWrapper, GPTNeoXLensWrapper, OlmoLensWrapper
from utils.circuit_utils import load_circuit_model


"""
Save a linear transformation that maximizes similarity between each layer and
the initial/final token representations in a model. Also computes the
SVD of representations at each layer.
"""
def compute_transformations(model, tokenizer, inputs_path, outpath_prefix, model_type,
        n_layers=12, n_tokens_to_fit=128000, batch_size=8, include_second_to_last=True):
    examples = read_examples(inputs_path)
    # Assume these are hidden states before layernorm.
    hidden_states = get_hidden_states(model, examples, batch_size, tokenizer,
                                      list(range(n_layers+1)))
    n_states = hidden_states[0].shape[0]
    mask = None
    if n_states > n_tokens_to_fit:
        # The same mask must be used for all layers.
        mask = np.random.choice(n_states, n_tokens_to_fit)
    for layer in range(n_layers+1):
        # Shape: (n_tokens, n_dims).
        layer_states = hidden_states[layer]
        if mask is not None:
            layer_states = layer_states[mask, :]
        # Save.
        np.save(f'{outpath_prefix}_reps_layer{layer}.npy',
                layer_states, allow_pickle=False)
    final_reps = hidden_states[-1]
    if mask is not None:
        final_reps = final_reps[mask, :]
    # Save second-to-last layer if desired.
    if include_second_to_last:
        secondlast_reps_path = f'{outpath_prefix}_reps_secondlast.npy'
        secondlast_reps = hidden_states[-2]
        if mask is not None:
            secondlast_reps = secondlast_reps[mask, :]
        np.save(secondlast_reps_path, secondlast_reps, allow_pickle=False)
        del secondlast_reps
    del hidden_states, layer_states
    # Also save initial token representations (no position embs etc) and
    # final representations (after layernorm).
    in_reps_path = f'{outpath_prefix}_reps_input.npy'
    out_reps_path = f'{outpath_prefix}_reps_output.npy'
    token_reps = get_token_embeddings(model, examples, model_type)
    if mask is not None:
        token_reps = token_reps[mask, :]
    np.save(in_reps_path, token_reps, allow_pickle=False)
    # Don't apply final layernorm here.
    # if model_type == 'gpt2':
    #     ln_f = model.transformer.ln_f.cpu()
    # final_reps = ln_f(torch.tensor(final_reps)).detach().numpy()
    np.save(out_reps_path, final_reps, allow_pickle=False)
    del token_reps, final_reps, examples

    # Especially when ablating models, there can be nan values in the
    # activations do to exploding values. These will propagate to the final
    # layer.
    out_reps = np.load(out_reps_path, allow_pickle=False)
    is_nan = np.any(np.isnan(out_reps), axis=-1)
    n_nan = np.sum(is_nan)
    if n_nan != 0: print(f'Warning: {n_nan} nan representations.')
    assert n_nan < n_tokens_to_fit // 100  # Do not allow more than 1% nan.
    del out_reps

    # Run linear regressions.
    for layer in range(n_layers+1):
        print(f'Running computations for layer {layer}.')
        layer_reps_path = f'{outpath_prefix}_reps_layer{layer}.npy'
        layer_reps = np.load(layer_reps_path, allow_pickle=False)[~is_nan, :]

        # Run SVD on mean-centered representations.
        print('Running SVD.')
        layer_mean = np.mean(layer_reps, axis=0)
        mean_centered = layer_reps - layer_mean.reshape(1, -1)
        _, s, vh = scipy.linalg.svd(mean_centered, full_matrices=False, compute_uv=True, overwrite_a=False)
        mean_outpath = f'{outpath_prefix}_mean_layer{layer}.npy'
        s_outpath = f'{outpath_prefix}_s_layer{layer}.npy'
        vh_outpath = f'{outpath_prefix}_vh_layer{layer}.npy'
        np.save(mean_outpath, layer_mean, allow_pickle=False)
        np.save(s_outpath, s, allow_pickle=False)
        np.save(vh_outpath, vh, allow_pickle=False)
        del s, vh, mean_centered

        # Estimate linear transformation from layer to final.
        print('Estimating linear transformation to final.')
        out_reps = np.load(out_reps_path, allow_pickle=False)[~is_nan, :]
        # Note: may overwrite layer_reps if copy_X is False.
        # reg = LinearRegression(fit_intercept=True, copy_X=True)
        reg = Ridge(alpha=1.0, fit_intercept=True, copy_X=True)
        reg = reg.fit(layer_reps, out_reps)
        linear_map = reg.coef_ # Shape: (n_targets, n_features).
        intercept = reg.intercept_  # Shape: (n_targets).
        np.save(f'{outpath_prefix}_linear_layer{layer}_out.npy',
                linear_map, allow_pickle=False)
        np.save(f'{outpath_prefix}_intercept_layer{layer}_out.npy',
                intercept, allow_pickle=False)
        del reg, linear_map, intercept, out_reps

        # Estimate linear transformation from layer to second-to-last.
        if include_second_to_last:
            print('Estimating linear transformation to second-to-last.')
            out_reps = np.load(secondlast_reps_path, allow_pickle=False)[~is_nan, :]
            # Note: may overwrite layer_reps if copy_X is False.
            # reg = LinearRegression(fit_intercept=True, copy_X=True)
            reg = Ridge(alpha=1.0, fit_intercept=True, copy_X=True)
            reg = reg.fit(layer_reps, out_reps)
            linear_map = reg.coef_ # Shape: (n_targets, n_features).
            intercept = reg.intercept_  # Shape: (n_targets).
            np.save(f'{outpath_prefix}_linear_layer{layer}_secondlast.npy',
                    linear_map, allow_pickle=False)
            np.save(f'{outpath_prefix}_intercept_layer{layer}_secondlast.npy',
                    intercept, allow_pickle=False)
            del reg, linear_map, intercept, out_reps

        # Estimate linear transformation from layer to initial.
        print('Estimating linear transformation to initial.')
        in_reps = np.load(in_reps_path, allow_pickle=False)[~is_nan, :]
        # Note: may overwrite layer_reps because copy_X is False.
        # reg = LinearRegression(fit_intercept=True, copy_X=False)
        reg = Ridge(alpha=1.0, fit_intercept=True, copy_X=True)
        reg = reg.fit(layer_reps, in_reps)
        linear_map = reg.coef_ # Shape: (n_targets, n_features).
        intercept = reg.intercept_  # Shape: (n_targets).
        np.save(f'{outpath_prefix}_linear_layer{layer}_in.npy',
                linear_map, allow_pickle=False)
        np.save(f'{outpath_prefix}_intercept_layer{layer}_in.npy',
                intercept, allow_pickle=False)
        del reg, linear_map, intercept, in_reps, layer_reps
        os.remove(layer_reps_path)
    os.remove(in_reps_path)
    os.remove(out_reps_path)
    if include_second_to_last:
        os.remove(secondlast_reps_path)
    return True


def load_lens_model(model_dir, model_type, n_layers_to_keep,
        circuit_path=None, checkpoint=None,
        linear_map=None, linear_bias=None, apply_lnf=True, use_complement=False):
    # Load original model.
    config, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type, checkpoint=checkpoint, use_complement=use_complement)
    if model_type == 'gpt2':
        model.transformer = GPT2LensWrapper(model.transformer, n_layers_to_keep,
                linear_map=linear_map, linear_bias=linear_bias, apply_lnf=apply_lnf)
    elif model_type == 'pythia':
        model.gpt_neox = GPTNeoXLensWrapper(model.gpt_neox, n_layers_to_keep,
                linear_map=linear_map, linear_bias=linear_bias, apply_lnf=apply_lnf)
    elif model_type == 'olmo':
        model.model = OlmoLensWrapper(model.model, n_layers_to_keep,
                linear_map=linear_map, linear_bias=linear_bias, apply_lnf=apply_lnf)
    # Config and tokenizer unchanged.
    return config, tokenizer, model
