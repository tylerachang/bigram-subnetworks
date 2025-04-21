"""
Analyze activations.
"""
import codecs
import json
import numpy as np
import os
import scipy
import torch
from utils.ngram_utils import NGramModel
from utils.circuit_utils import load_circuit_model
from utils.lens_utils import compute_transformations, load_lens_model
from utils.model_utils import compute_correlation_metrics, compute_input_token_probs


# For each model: run full (neginf), circuit, and circuit complement.

# GPT2 ours.
# 21 checkpoints, approximately evenly spaced log10.
CKPTS = [None, 0, 101, 205, 421, 647, 1009, 1535, 2592, 4049, 6329, 9931, 15749,
         25422, 39831, 62938, 100800, 160072, 252862, 398125, 625532]
MODEL_DIR = '../../monolingual-pretraining/models/gpt2_0'
MODEL_TYPE = 'gpt2'
INPUTS_PATH = '../tokenized_inputs/orig_train_subset_1k_reshuffled.txt'
OUTDIR = '../bigram_lens_results_ridge/gpt2_ours_neginf'
LAMBDA_STR = 'neginf'
USE_CIRCUIT_COMPLEMENT = False
EVAL_INPUTS_PATH = '../tokenized_inputs/orig_eval_subset_1k.txt'
NGRAM_N = 2
CIRCUITS_DIR = '../trained_circuits_gpt2_ours'
N_LAYERS = 12

# Pythia 160m.
# Checkpoints evenly spaced log10.
CKPTS = [None, 0, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 25000, 40000, 63000, 100000, 143000]
MODEL_DIR = 'EleutherAI/pythia-160m'  # Change for different sizes.
MODEL_TYPE = 'pythia'
INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_fortrain_tail1k.txt'
OUTDIR = '../bigram_lens_results_ridge/pythia_160m_1e1'
LAMBDA_STR = '1e1'
USE_CIRCUIT_COMPLEMENT = False
EVAL_INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_foreval_1k.txt'
NGRAM_N = 2
CIRCUITS_DIR = '../trained_circuits_pythia_160m'
N_LAYERS = 12

# Pythia 1b.
# Checkpoints evenly spaced log10.
CKPTS = [None, 0, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 25000, 40000, 63000, 100000, 143000]
MODEL_DIR = 'EleutherAI/pythia-1b'  # Change for different sizes.
MODEL_TYPE = 'pythia'
INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_fortrain_tail1k.txt'
OUTDIR = '../bigram_lens_results_ridge/pythia_1b_1e2'
LAMBDA_STR = '1e2'
USE_CIRCUIT_COMPLEMENT = False
EVAL_INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_foreval_1k.txt'
NGRAM_N = 2
CIRCUITS_DIR = '../trained_circuits_pythia_1b'
N_LAYERS = 16

# GPT2-small.
CKPTS = [None]  # Only have final checkpoint.
MODEL_DIR = 'openai-community/gpt2'  # Change for different sizes.
MODEL_TYPE = 'gpt2'
INPUTS_PATH = '../tokenized_inputs/gpt2_oscar_en_fortrain_tail1k.txt'
OUTDIR = '../bigram_lens_results_ridge/gpt2_small_1e1'
LAMBDA_STR = '1e1'
USE_CIRCUIT_COMPLEMENT = False
EVAL_INPUTS_PATH = '../tokenized_inputs/gpt2_oscar_en_foreval_1k.txt'
NGRAM_N = 2
CIRCUITS_DIR = '../trained_circuits_gpt2_small'
N_LAYERS = 12


# Fit tuned lens for each layer in each checkpoint.
BATCH_SIZE = 8
ngram_name = {1: 'unigram', 2: 'bigram', 3: 'trigram'}[NGRAM_N]
for ckpt in CKPTS:
    ckpt_outdir = os.path.join(OUTDIR, f'ckpt{ckpt}')
    os.makedirs(ckpt_outdir, exist_ok=True)
    outpath_prefix = os.path.join(ckpt_outdir, 'activations')
    check_path = f'{outpath_prefix}_linear_layer0_out.npy'  # Check if this file already exists first.
    circuit_path = f'{CIRCUITS_DIR}/{ngram_name}_ckpt{ckpt}_t1e-3_lambda{LAMBDA_STR}_lr5e-5.pickle'
    _, tokenizer, model = load_circuit_model(MODEL_DIR, circuit_path, model_type=MODEL_TYPE,
            checkpoint=ckpt, use_complement=USE_CIRCUIT_COMPLEMENT, randomize=False)
    if not os.path.isfile(check_path):
        compute_transformations(model, tokenizer, INPUTS_PATH, outpath_prefix, MODEL_TYPE,
                n_layers=N_LAYERS, n_tokens_to_fit=128000, batch_size=BATCH_SIZE,
                include_second_to_last=True)
    json_outpath = os.path.join(ckpt_outdir, 'scores.json')
    if not os.path.isfile(json_outpath):
        outdict = dict()
        # Evaluate correlation with final output, and recall for predicting input. I.e.:
        # (1) how close to the output using the tuned lens at each layer, i.e. how well
        # a linear transformation approximates the next token prediction.
        # (2) how accurately predicts the input token using the input lens at each layer,
        # i.e. how well a linear transformation approximates the input token.
        for layer in range(N_LAYERS+1):

            # Correlation with final output.
            # Define target function.
            softmax = torch.nn.Softmax(dim=-1)
            def target_fn(input_ids):
                logits = model(input_ids=input_ids).logits.detach()
                probs = softmax(logits)
                return probs
            # Load and run lens model.
            linear_map = np.load(f'{outpath_prefix}_linear_layer{layer}_out.npy', allow_pickle=False)
            linear_bias = np.load(f'{outpath_prefix}_intercept_layer{layer}_out.npy', allow_pickle=False)
            config, tokenizer, lens_model = load_lens_model(MODEL_DIR, MODEL_TYPE, layer,
                    circuit_path=circuit_path, checkpoint=ckpt,
                    linear_map=linear_map, linear_bias=linear_bias, apply_lnf=True,
                    use_complement=USE_CIRCUIT_COMPLEMENT)
            model_surprisals, target_surprisals, xent = compute_correlation_metrics(lens_model, target_fn, EVAL_INPUTS_PATH, batch_size=BATCH_SIZE)
            target_surprisals[np.isinf(target_surprisals)] = np.max(target_surprisals[~np.isinf(target_surprisals)])
            model_surprisals[np.isinf(model_surprisals)] = np.max(model_surprisals[~np.isinf(model_surprisals)])
            r, p = scipy.stats.pearsonr(model_surprisals, target_surprisals)
            outdict[f'layer{layer}_out_xent'] = float(xent)
            outdict[f'layer{layer}_out_corr'] = r
            del lens_model, model_surprisals, target_surprisals

            # Predicting input token.
            linear_map = np.load(f'{outpath_prefix}_linear_layer{layer}_in.npy', allow_pickle=False)
            linear_bias = np.load(f'{outpath_prefix}_intercept_layer{layer}_in.npy', allow_pickle=False)
            config, tokenizer, lens_model = load_lens_model(MODEL_DIR, MODEL_TYPE, layer,
                    circuit_path=circuit_path, checkpoint=ckpt,
                    linear_map=linear_map, linear_bias=linear_bias, apply_lnf=False,
                    use_complement=USE_CIRCUIT_COMPLEMENT)
            # Models with untied input/output embeddings need to revert to
            # input token embedding here to predict input tokens.
            if MODEL_TYPE == 'pythia':
                lens_model.embed_out.weight = lens_model.gpt_neox.orig_model.embed_in.weight
            all_surprisals, all_ranks = compute_input_token_probs(lens_model, EVAL_INPUTS_PATH, batch_size=BATCH_SIZE)
            outdict[f'layer{layer}_in_recall10'] = float(np.mean(all_ranks < 10))
            outdict[f'layer{layer}_in_recall5'] = float(np.mean(all_ranks < 5))
            outdict[f'layer{layer}_in_recall1'] = float(np.mean(all_ranks < 1))
            outdict[f'layer{layer}_in_surprisal'] = float(np.mean(all_surprisals))
            del lens_model, all_surprisals, all_ranks

            # Compute the median rotation for the output and input transformations.
            def get_scalings_and_rotations(l):
                # https://textbooks.math.gatech.edu/ila/complex-eigenvalues.html
                # https://haoye.us/post/2019-12-05-interpreting-complex-eigenvalues/
                eig_vals, eig_vecs = np.linalg.eig(l)
                # Complex eigenvalues come in conjugate pairs because the linear map is
                # real-valued. Scaling is the modulus of each complex eigenvalue.
                scalings = np.absolute(eig_vals)
                # Angle is the argument of each complex eigenvalue.
                # Counterclockwise angle in (-180, 180].
                rotations = np.angle(eig_vals, deg=True)
                # Absolute angle.
                rotations = np.absolute(rotations)
                return scalings, rotations
            linear_map = np.load(f'{outpath_prefix}_linear_layer{layer}_in.npy')
            scalings, rotations = get_scalings_and_rotations(linear_map)
            outdict[f'layer{layer}_in_rotation'] = float(np.median(rotations))
            linear_map = np.load(f'{outpath_prefix}_linear_layer{layer}_out.npy')
            scalings, rotations = get_scalings_and_rotations(linear_map)
            outdict[f'layer{layer}_out_rotation'] = float(np.median(rotations))
            linear_map = np.load(f'{outpath_prefix}_linear_layer{layer}_secondlast.npy')
            scalings, rotations = get_scalings_and_rotations(linear_map)
            outdict[f'layer{layer}_secondlast_rotation'] = float(np.median(rotations))

        # Write output.
        with codecs.open(json_outpath, 'wb', encoding='utf-8') as f:
            f.write(json.dumps(outdict) + '\n')
    del model
