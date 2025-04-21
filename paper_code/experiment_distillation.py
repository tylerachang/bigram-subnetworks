"""
Experiment if our target function is the original model, i.e. trying to distill
to a subset of parameters.
"""

import codecs
import json
import numpy as np
import os
import scipy
import torch

from utils.ngram_utils import NGramModel
from utils.model_utils import load_model
from utils.circuit_utils import train_circuit, load_circuit_model, check_loss_stability
from utils.model_utils import compute_correlation_metrics, get_model_surprisals, get_target_surprisals


# Pythia-1b
CKPTS = [None]
NGRAM_CACHE_DIR = '../ngrams_cache'
NGRAM_PREFIX = 'pythia_train'
MODEL_DIR = 'EleutherAI/pythia-1b'
MODEL_TYPE = 'pythia'
INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_fortrain_500k.txt'
OUTDIR = '../experimental/distillation/pythia_1b'
NGRAM_N = 2
LAMBDAS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]  # Exclude 0.0 because no sparsity can perfectly recreate model.
EVAL_INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_foreval_10k.txt'
NUM_ATTEMPTS = 3
GRAD_ACC_STEPS = 2  # Should divide 32.

# Pythia-160m
CKPTS = [None]
NGRAM_CACHE_DIR = '../ngrams_cache'
NGRAM_PREFIX = 'pythia_train'
MODEL_DIR = 'EleutherAI/pythia-160m'
MODEL_TYPE = 'pythia'
INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_fortrain_500k.txt'
OUTDIR = '../experimental/distillation/pythia_160m'
NGRAM_N = 2
LAMBDAS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]  # Exclude 0.0 because no sparsity can perfectly recreate model.
EVAL_INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_foreval_10k.txt'
NUM_ATTEMPTS = 3
GRAD_ACC_STEPS = 1  # Should divide 32.

# Our GPT-2 model.
CKPTS = [None]
NGRAM_CACHE_DIR = '../ngrams_cache'
NGRAM_PREFIX = 'orig_train'
MODEL_DIR = '../../monolingual-pretraining/models/gpt2_0'
MODEL_TYPE = 'gpt2'
INPUTS_PATH = '../tokenized_inputs/orig_train_subset_500k.txt'
OUTDIR = '../experimental/distillation/gpt2_ours'
NGRAM_N = 2
LAMBDAS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]  # Exclude 0.0 because no sparsity can perfectly recreate model.
EVAL_INPUTS_PATH = '../tokenized_inputs/orig_eval_subset_10k.txt'
NUM_ATTEMPTS = 3
GRAD_ACC_STEPS = 1  # Should divide 32.

# Official GPT-2 model.
CKPTS = [None]
NGRAM_CACHE_DIR = '../ngrams_cache'
NGRAM_PREFIX = 'gpt2_train'
MODEL_DIR = 'openai-community/gpt2'  # Change for different sizes.
MODEL_TYPE = 'gpt2'
INPUTS_PATH = '../tokenized_inputs/gpt2_oscar_en_fortrain_500k.txt'
OUTDIR = '../experimental/distillation/gpt2_small'  # Change for different sizes.
NGRAM_N = 2
LAMBDAS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]  # Exclude 0.0 because no sparsity can perfectly recreate model.
EVAL_INPUTS_PATH = '../tokenized_inputs/gpt2_oscar_en_foreval_10k.txt'
NUM_ATTEMPTS = 3
GRAD_ACC_STEPS = 4  # Should divide 32.


# Define target function (i.e. the full model).
os.makedirs(OUTDIR, exist_ok=True)
_, _, target_model = load_model(
        MODEL_DIR, MODEL_TYPE, checkpoint=None,
        cache_dir='hf_cache', override_for_hidden_states=False)
softmax = torch.nn.Softmax(dim=-1)
def target_fn(input_ids):
    logits = target_model(input_ids=input_ids).logits.detach()
    probs = softmax(logits)
    return probs


"""
Below is identical to train_circuits.py! Except when # MODIFIED included.
"""
for ckpt in CKPTS:
    for lambda_val in LAMBDAS:
        lambda_name = {0.0: '0', 0.1: '1e-1', 0.5: '5e-1',
                       1.0: '1', 5.0: '5', 10.0: '1e1',
                       50.0: '5e1', 100.0: '1e2',
                       500.0: '5e2', 1000.0: '1e3',
                       -np.inf: 'neginf', np.inf: 'inf'}[lambda_val]
        # MODIFIED:
        outpath = os.path.join(OUTDIR, f'distill_ckpt{ckpt}_t1e-3_lambda{lambda_name}_lr5e-5.pickle')
        finished_outpaths = []
        # Training circuits for smaller models can be less stable. E.g. for
        # Pythia-14m, training does not always converge even for lr 1e-5 and
        # temp_increase 5e-4 (with max steps increased to 30000 due to slower
        # convergence). However, these defaults work well for other sizes.
        learning_rate = 0.00005
        temp_increase = 1.001
        max_steps = 15000
        json_path = outpath.replace('.pickle', '.json')
        if os.path.isfile(outpath):
            print(f'Already found outpath: {outpath}')
            finished_outpaths.append(outpath)
        else:
            # Train circuit.
            for attempt_i in range(NUM_ATTEMPTS):  # Attempt with different shuffles of train.
                print(f'Running attempt {attempt_i}.')
                attempt_outpath = outpath.replace('.pickle', f'_attempt{attempt_i}.pickle')
                circuit = train_circuit(MODEL_DIR, target_fn, INPUTS_PATH, attempt_outpath,
                        model_type=MODEL_TYPE, checkpoint=ckpt,
                        grad_acc_steps=GRAD_ACC_STEPS,
                        max_steps=max_steps*GRAD_ACC_STEPS, batch_size=32//GRAD_ACC_STEPS,
                        temp_increase=temp_increase, l0_lambda=lambda_val, learning_rate=learning_rate,
                        shuffle_train=True)
                del circuit
                # Check for unstable training.
                attempt_json_path = attempt_outpath.replace('.pickle', '.json')
                with codecs.open(attempt_json_path, 'rb', encoding='utf-8') as f:
                    results_dict = json.loads(f.read())
                losses = results_dict['xent_loss_per_step']
                is_stable = True if losses is None else check_loss_stability(losses)
                if is_stable:
                    # Success!
                    os.rename(attempt_outpath, outpath)
                    os.rename(attempt_json_path, json_path)
                    finished_outpaths.append(outpath)
                    break  # No more attempts needed.
                finished_outpaths.append(attempt_outpath)

        # MODIFIED: use distill correlation instead of ngram correlation.
        # Eval.
        for outpath in finished_outpaths:
            json_path = outpath.replace('.pickle', '.json')
            with codecs.open(json_path, 'rb', encoding='utf-8') as f:
                results_dict = json.loads(f.read())
            if 'distill_correlation' not in results_dict:
                _, tokenizer, model = load_circuit_model(MODEL_DIR, outpath, model_type=MODEL_TYPE, checkpoint=ckpt)
                model_surprisals, target_surprisals, xent = compute_correlation_metrics(
                        model, target_fn, EVAL_INPUTS_PATH, batch_size=32//GRAD_ACC_STEPS)
                target_surprisals[np.isinf(target_surprisals)] = np.max(target_surprisals[~np.isinf(target_surprisals)])
                model_surprisals[np.isinf(model_surprisals)] = np.max(model_surprisals[~np.isinf(model_surprisals)])
                r, p = scipy.stats.pearsonr(model_surprisals, target_surprisals)
                results_dict['distill_correlation'] = r
                results_dict['distill_xent'] = float(xent)
                del model
                # Rewrite output dict.
                with codecs.open(json_path, 'wb', encoding='utf-8') as f:
                    f.write(json.dumps(results_dict) + '\n')

        # MODIFIED: n-gram not needed until eval, so this code is moved down here.
        if MODEL_TYPE == 'pythia':
            # For bigrams, no pruning, following https://github.com/tylerachang/lm-learning-curves/blob/main/annotate_curves.py.
            ngram_model = NGramModel(NGRAM_CACHE_DIR, NGRAM_PREFIX, NGRAM_N, 50277,
                    reference_path='../tokenized_inputs/pythia_oscar_en_fortrain.txt',
                    prune_every=999999999, prune_minimum=None)
            # Or, for trigrams after running above, use pruning as defaults.
        elif MODEL_TYPE == 'gpt2':
            if MODEL_DIR.startswith('openai-community'):
                print('Using official GPT-2 implementation.')
                ngram_model = NGramModel(NGRAM_CACHE_DIR, NGRAM_PREFIX, NGRAM_N, 50257,
                        reference_path='../tokenized_inputs/gpt2_oscar_en_fortrain.txt',
                        prune_every=999999999, prune_minimum=None)
            else:
                print('Using our GPT-2 implementation.')
                ngram_model = NGramModel(NGRAM_CACHE_DIR, NGRAM_PREFIX, NGRAM_N, 50004)
        elif MODEL_TYPE == 'olmo':
            ngram_model = NGramModel(NGRAM_CACHE_DIR, NGRAM_PREFIX, NGRAM_N, 50280,
                    reference_path='../tokenized_inputs/olmo_oscar_en_fortrain.txt',
                    prune_every=999999999, prune_minimum=None)
        ngram_name = {1: 'unigram', 2: 'bigram', 3: 'trigram'}[NGRAM_N]
        ngram_target_fn = ngram_model.ngram_predict_with_backoff
        # End MODIFIED.

        # MODIFIED: use ngram_target_fn.
        # Eval.
        for outpath in finished_outpaths:
            json_path = outpath.replace('.pickle', '.json')
            with codecs.open(json_path, 'rb', encoding='utf-8') as f:
                results_dict = json.loads(f.read())
            if 'ngram_correlation' not in results_dict:
                results_dict['ngram_n'] = NGRAM_N
                _, tokenizer, model = load_circuit_model(MODEL_DIR, outpath, model_type=MODEL_TYPE, checkpoint=ckpt)
                model_surprisals, target_surprisals, xent = compute_correlation_metrics(
                        model, ngram_target_fn, EVAL_INPUTS_PATH, batch_size=32//GRAD_ACC_STEPS)
                target_surprisals[np.isinf(target_surprisals)] = np.max(target_surprisals[~np.isinf(target_surprisals)])
                model_surprisals[np.isinf(model_surprisals)] = np.max(model_surprisals[~np.isinf(model_surprisals)])
                r, p = scipy.stats.pearsonr(model_surprisals, target_surprisals)
                results_dict['ngram_correlation'] = r
                results_dict['ngram_xent'] = float(xent)
                del model
                # Rewrite output dict.
                with codecs.open(json_path, 'wb', encoding='utf-8') as f:
                    f.write(json.dumps(results_dict) + '\n')
            if 'ngram_corr_dropinf' not in results_dict:
                ngram_surprisals_outpath = os.path.join(NGRAM_CACHE_DIR,
                        EVAL_INPUTS_PATH.split('/')[-1].replace('.txt', f'_{ngram_name}_surprisals.npy'))
                if os.path.isfile(ngram_surprisals_outpath):
                    ngram_surprisals = np.load(ngram_surprisals_outpath, allow_pickle=False)
                else:
                    ngram_surprisals = get_target_surprisals(ngram_target_fn, EVAL_INPUTS_PATH)
                    np.save(ngram_surprisals_outpath, ngram_surprisals, allow_pickle=False)
                _, tokenizer, model = load_circuit_model(MODEL_DIR, outpath, model_type=MODEL_TYPE, checkpoint=ckpt)
                model_surprisals = get_model_surprisals(model, EVAL_INPUTS_PATH, batch_size=32//GRAD_ACC_STEPS)
                to_drop = np.isinf(ngram_surprisals)
                model_infs = np.isinf(model_surprisals)
                if np.sum(model_infs) > 0: print(f'Warning: {np.sum(model_infs)} inf surprisals for model.')
                model_surprisals[model_infs] = np.max(model_surprisals[~model_infs])
                r, p = scipy.stats.pearsonr(model_surprisals[~to_drop], ngram_surprisals[~to_drop])
                results_dict['ngram_corr_dropinf'] = r
                del model, ngram_surprisals, model_surprisals
                # Rewrite output dict.
                with codecs.open(json_path, 'wb', encoding='utf-8') as f:
                    f.write(json.dumps(results_dict) + '\n')
