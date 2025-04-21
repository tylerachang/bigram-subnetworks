import numpy as np
import os
from utils.circuit_utils import load_circuit_model
from utils.model_utils import generate_text

model_dir = '../../monolingual-pretraining/models/gpt2_0'
circuit_path = '../trained_circuits_gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle'
model_type = 'gpt2'

model_dir = 'EleutherAI/pythia-1b'
circuit_path = '../trained_circuits_pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle'
model_type = 'pythia'


# Run.
def generate(model, tokenizer, text):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    output_text = tokenizer.decode(generate_text(model, input_ids, tokenizer, max_seq_len=128, temperature=0.3))
    return output_text
text = 'This is'
# For original model:
_, tokenizer, model = load_circuit_model(model_dir, None, model_type=model_type, checkpoint=None)
print('For original model:')
print(generate(model, tokenizer, text))
# For bigram circuit:
_, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
        checkpoint=None, use_complement=False, randomize=False)
print('For bigram circuit:')
print(generate(model, tokenizer, text))
# For random circuit with same structure as bigram circuit:
_, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
        checkpoint=None, use_complement=False, randomize=True)
print('For random circuit with same structure as bigram circuit:')
print(generate(model, tokenizer, text))
# For bigram complement:
_, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
        checkpoint=None, use_complement=True, randomize=False)
print('Ablating bigram circuit:')
print(generate(model, tokenizer, text))
# For random circuit with same structure as bigram complement:
_, tokenizer, model = load_circuit_model(model_dir, circuit_path, model_type=model_type,
        checkpoint=None, use_complement=True, randomize=True)
print('Ablating random circuit with same structure as bigram circuit:')
print(generate(model, tokenizer, text))
