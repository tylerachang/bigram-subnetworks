import codecs
import json
import numpy as np
import os
from utils.circuit_utils import load_circuit_model
from utils.model_utils import get_model_surprisals

model_dir = '../../monolingual-pretraining/models/gpt2_0'
circuit_path = '../trained_circuits_gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle'
model_type = 'gpt2'
outpath = '../trained_circuits_gpt2_ours/perplexities_ckptNone_lambda1e1.json'
inputs_path = '../tokenized_inputs/orig_eval_subset_10k.txt'
# circuit_path = '../experimental/distillation/gpt2_ours/distill_ckptNone_t1e-3_lambda5e1_lr5e-5.pickle'
# outpath = '../experimental/distillation/gpt2_ours/perplexities_ckptNone_lambda5e1.json'

model_dir = 'EleutherAI/pythia-160m'
circuit_path = '../trained_circuits_pythia_160m/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle'
model_type = 'pythia'
outpath = '../trained_circuits_pythia_160m/perplexities_ckptNone_lambda1e1.json'
inputs_path = '../tokenized_inputs/pythia_oscar_en_foreval_10k.txt'
# circuit_path = '../experimental/distillation/pythia_160m/distill_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle'
# outpath = '../experimental/distillation/pythia_160m/perplexities_ckptNone_lambda1e1.json'

model_dir = 'EleutherAI/pythia-1b'
circuit_path = '../trained_circuits_pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle'
model_type = 'pythia'
outpath = '../trained_circuits_pythia_1b/perplexities_ckptNone_lambda1e2.json'
inputs_path = '../tokenized_inputs/pythia_oscar_en_foreval_10k.txt'
# circuit_path = '../experimental/distillation/pythia_1b/distill_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle'
# outpath = '../experimental/distillation/pythia_1b/perplexities_ckptNone_lambda1e2.json'


batch_size = 8
if not os.path.isfile(outpath):
    outdict = dict()
    # For full model:
    _, tokenizer, model = load_circuit_model(model_dir, None, model_type=model_type, checkpoint=None)
    surprisals = get_model_surprisals(model, inputs_path, batch_size=batch_size)
    surprisals[np.isinf(surprisals)] = np.max(surprisals[~np.isinf(surprisals)])
    outdict['full_surprisal_mean'] = float(np.mean(surprisals))
    outdict['full_surprisal_std'] = float(np.std(surprisals))
    # For bigram circuit:
    _, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
            checkpoint=None, use_complement=False, randomize=False)
    surprisals = get_model_surprisals(model, inputs_path, batch_size=batch_size)
    surprisals[np.isinf(surprisals)] = np.max(surprisals[~np.isinf(surprisals)])
    outdict['circuit_surprisal_mean'] = float(np.mean(surprisals))
    outdict['circuit_surprisal_std'] = float(np.std(surprisals))
    # For random circuit with same structure as bigram circuit:
    _, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
            checkpoint=None, use_complement=False, randomize=True)
    surprisals = get_model_surprisals(model, inputs_path, batch_size=batch_size)
    surprisals[np.isinf(surprisals)] = np.max(surprisals[~np.isinf(surprisals)])
    outdict['rand_circuit_surprisal_mean'] = float(np.mean(surprisals))
    outdict['rand_circuit_surprisal_std'] = float(np.std(surprisals))
    # For bigram complement:
    _, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
            checkpoint=None, use_complement=True, randomize=False)
    surprisals = get_model_surprisals(model, inputs_path, batch_size=batch_size)
    surprisals[np.isinf(surprisals)] = np.max(surprisals[~np.isinf(surprisals)])
    outdict['ablate_circuit_surprisal_mean'] = float(np.mean(surprisals))
    outdict['ablate_circuit_surprisal_std'] = float(np.std(surprisals))
    # For random circuit with same structure as bigram complement:
    _, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
            checkpoint=None, use_complement=True, randomize=True)
    surprisals = get_model_surprisals(model, inputs_path, batch_size=batch_size)
    surprisals[np.isinf(surprisals)] = np.max(surprisals[~np.isinf(surprisals)])
    outdict['ablate_rand_circuit_surprisal_mean'] = float(np.mean(surprisals))
    outdict['ablate_rand_circuit_surprisal_std'] = float(np.std(surprisals))
    # Write output.
    with codecs.open(outpath, 'wb', encoding='utf-8') as f:
        f.write(json.dumps(outdict) + '\n')
