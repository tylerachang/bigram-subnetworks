"""
Train circuits for different checkpoints and different lambdas, for n-grams.
Circuit training code builds on the continuous_sparsification code from:
https://github.com/mlepori1/NeuroSurgeon
"""
import codecs
import json
import numpy as np
import os
import scipy
from utils.ngram_utils import NGramModel
from utils.circuit_utils import train_circuit, load_circuit_model, check_loss_stability
from utils.model_utils import compute_correlation_metrics, get_model_surprisals, get_target_surprisals


# GPT2 model (ours):
# 21 checkpoints, approximately evenly spaced log10.
CKPTS = [None, 0, 101, 205, 421, 647, 1009, 1535, 2592, 4049, 6329, 9931, 15749,
         25422, 39831, 62938, 100800, 160072, 252862, 398125, 625532]
# CKPTS = [None]  # To only use final checkpoint.
NGRAM_CACHE_DIR = '../ngrams_cache'
NGRAM_PREFIX = 'orig_train'
MODEL_DIR = '../../monolingual-pretraining/models/gpt2_0'
MODEL_TYPE = 'gpt2'
INPUTS_PATH = '../tokenized_inputs/orig_train_subset_500k.txt'
OUTDIR = '../trained_circuits_gpt2_ours'
NGRAM_N = 2
# All for final checkpoint:
# LAMBDAS = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, -np.inf, np.inf]
LAMBDAS = [0.0, 1.0, 10.0, 100.0, 500.0, -np.inf, np.inf]
EVAL_INPUTS_PATH = '../tokenized_inputs/orig_eval_subset_10k.txt'
NUM_ATTEMPTS = 3
GRAD_ACC_STEPS = 1  # Should divide 32.

# Pythia.
# Checkpoints evenly spaced log10.
CKPTS = [0, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 25000, 40000, 63000, 100000, 143000, None]
# CKPTS = [None]  # To only use final checkpoint.
NGRAM_CACHE_DIR = '../ngrams_cache'
NGRAM_PREFIX = 'pythia_train'
MODEL_DIR = 'EleutherAI/pythia-160m'  # Change for different sizes.
MODEL_TYPE = 'pythia'
INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_fortrain_500k.txt'
OUTDIR = '../trained_circuits_pythia_160m'  # Change for different sizes.
NGRAM_N = 2
# All for final checkpoint:
# LAMBDAS = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, -np.inf, np.inf]
LAMBDAS = [0.0, 1.0, 10.0, 100.0, 500.0, -np.inf, np.inf]
EVAL_INPUTS_PATH = '../tokenized_inputs/pythia_oscar_en_foreval_10k.txt'
NUM_ATTEMPTS = 3
GRAD_ACC_STEPS = 1  # Should divide 32. 2 for 1b, 1 for others.

# GPT2 (official).
CKPTS = [None]  # Only have final checkpoint.
NGRAM_CACHE_DIR = '../ngrams_cache'
NGRAM_PREFIX = 'gpt2_train'
MODEL_DIR = 'openai-community/gpt2'  # Change for different sizes.
MODEL_TYPE = 'gpt2'
INPUTS_PATH = '../tokenized_inputs/gpt2_oscar_en_fortrain_500k.txt'
OUTDIR = '../trained_circuits_gpt2_small'  # Change for different sizes.
NGRAM_N = 2
LAMBDAS = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, -np.inf, np.inf]
EVAL_INPUTS_PATH = '../tokenized_inputs/gpt2_oscar_en_foreval_10k.txt'
NUM_ATTEMPTS = 3
GRAD_ACC_STEPS = 1  # Should divide 32.

# OLMo.
# CKPTS = [None]
# NGRAM_CACHE_DIR = '../ngrams_cache'
# NGRAM_PREFIX = 'olmo_train'
# MODEL_DIR = 'allenai/OLMo-1B-hf'
# MODEL_TYPE = 'olmo'
# INPUTS_PATH = '../tokenized_inputs/olmo_oscar_en_fortrain_500k.txt'
# OUTDIR = '../trained_circuits_olmo_1b'
# NGRAM_N = 2
# LAMBDAS = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
# EVAL_INPUTS_PATH = '../tokenized_inputs/olmo_oscar_en_foreval_10k.txt'
# NUM_ATTEMPTS = 3
# GRAD_ACC_STEPS = 2  # Should divide 32.


# Run circuit training and eval.
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


target_fn = ngram_model.ngram_predict_with_backoff
ngram_name = {1: 'unigram', 2: 'bigram', 3: 'trigram'}[NGRAM_N]
for ckpt in CKPTS:
    for lambda_val in LAMBDAS:
        lambda_name = {0.0: '0', 0.1: '1e-1', 0.5: '5e-1',
                       1.0: '1', 5.0: '5', 10.0: '1e1',
                       50.0: '5e1', 100.0: '1e2',
                       500.0: '5e2', 1000.0: '1e3',
                       -np.inf: 'neginf', np.inf: 'inf'}[lambda_val]
        outpath = os.path.join(OUTDIR, f'{ngram_name}_ckpt{ckpt}_t1e-3_lambda{lambda_name}_lr5e-5.pickle')
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
            print(f'Training circuit: {outpath}')
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
        # Eval.
        for outpath in finished_outpaths:
            json_path = outpath.replace('.pickle', '.json')
            with codecs.open(json_path, 'rb', encoding='utf-8') as f:
                results_dict = json.loads(f.read())
            # Eval ngram correlation and cross-entropy.
            if 'ngram_correlation' not in results_dict:
                results_dict['ngram_n'] = NGRAM_N
                _, tokenizer, model = load_circuit_model(MODEL_DIR, outpath, model_type=MODEL_TYPE, checkpoint=ckpt)
                model_surprisals, target_surprisals, xent = compute_correlation_metrics(
                        model, target_fn, EVAL_INPUTS_PATH, batch_size=32//GRAD_ACC_STEPS)
                target_surprisals[np.isinf(target_surprisals)] = np.max(target_surprisals[~np.isinf(target_surprisals)])
                model_surprisals[np.isinf(model_surprisals)] = np.max(model_surprisals[~np.isinf(model_surprisals)])
                r, p = scipy.stats.pearsonr(model_surprisals, target_surprisals)
                results_dict['ngram_correlation'] = r
                results_dict['ngram_xent'] = float(xent)
                del model, target_surprisals, model_surprisals
                # Rewrite output dict.
                with codecs.open(json_path, 'wb', encoding='utf-8') as f:
                    f.write(json.dumps(results_dict) + '\n')
            # This was added later, and it does not need to re-run the bigram
            # predictions; those can be saved (because we only need surprisals,
            # unlike above, where we need to compute cross-entropy as well).
            # This drops any infinite ngram surprisals, i.e. unobserved / zero
            # probability ngrams.
            if 'ngram_corr_dropinf' not in results_dict:
                ngram_surprisals_outpath = os.path.join(NGRAM_CACHE_DIR,
                        EVAL_INPUTS_PATH.split('/')[-1].replace('.txt', f'_{ngram_name}_surprisals.npy'))
                if os.path.isfile(ngram_surprisals_outpath):
                    ngram_surprisals = np.load(ngram_surprisals_outpath, allow_pickle=False)
                else:
                    ngram_surprisals = get_target_surprisals(target_fn, EVAL_INPUTS_PATH)
                    np.save(ngram_surprisals_outpath, ngram_surprisals, allow_pickle=False)
                _, tokenizer, model = load_circuit_model(MODEL_DIR, outpath, model_type=MODEL_TYPE, checkpoint=ckpt)
                model_surprisals = get_model_surprisals(model, EVAL_INPUTS_PATH, batch_size=32//GRAD_ACC_STEPS)
                # Drop infinity ngram surprisals, but not infinite model surprisals.
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
