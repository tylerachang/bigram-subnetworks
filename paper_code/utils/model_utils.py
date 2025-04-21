"""
Utils for Hugging Face Transformer models.
"""

import codecs
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoConfig, AutoTokenizer, AlbertTokenizer,
    AutoModelForCausalLM, AutoModelForMaskedLM, GPT2LMHeadModel,
    GPTNeoXForCausalLM, OlmoForCausalLM)

from utils.constants import OLMO_TOKENS_MAP
from utils.model_overrides import GPT2ModelOverride, GPTNeoXModelOverride, OlmoModelOverride


"""
Read tokenized examples from a txt file.
"""
def read_examples(inpath):
    ds = []
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    for l in infile:
        example = [int(token_id) for token_id in l.strip().split()]
        ds.append(example)
    infile.close()
    return ds


"""
Loads a model from a directory.
If checkpoint is provided (an integer for number of steps), loads that
checkpoint; otherwise, loads the final model.
Returns the config, tokenizer, and model.
The model_type is bert or gpt2.
Places model on cuda if available.
"""
def load_model(model_dir, model_type, checkpoint=None, tokenizer_path_override=None,
               config_path_override=None, cache_dir='hf_cache',
               override_for_hidden_states=True):
    model_type = model_type.lower()
    # Handle our custom-trained models:
    if model_type in ['gpt2', 'bert'] and not model_dir.startswith('openai-community'):
        # Our custom-trained models.
        # Load config.
        config_path = os.path.join(model_dir, 'config.json') if config_path_override is None else config_path_override
        config = AutoConfig.from_pretrained(config_path, cache_dir=cache_dir)
        # Load tokenizer.
        tokenizer_path = model_dir if tokenizer_path_override is None else tokenizer_path_override
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
        except:
            # If passing in a raw tokenizer model file, assume ALBERT sentencepiece model.
            print('WARNING: attempting to use local sentencepiece model file as tokenizer.')
            tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
        # Overwrite special token ids in the configs.
        config.bos_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.sep_token_id
        config.pad_token_id = tokenizer.pad_token_id
        if model_type == 'bert':
            max_seq_len = config.max_position_embeddings
        elif model_type == 'gpt2':
            max_seq_len = config.n_positions
        # Load model.
        if checkpoint is not None:
            model_dir = os.path.join(model_dir, 'checkpoint-' + str(checkpoint))
        print('Loading from directory: {}'.format(model_dir))
        if model_type == 'gpt2': # GPT2LMHeadModel.
            model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, cache_dir=cache_dir)
            # Override so that final layer hidden states are saved before the final
            # layer norm, to be comparable with other layers.
            if type(model) == GPT2LMHeadModel and override_for_hidden_states:
                print('Overriding model to save final layer hidden states correctly.')
                overridden_gpt2 = GPT2ModelOverride.from_pretrained(model_dir, config=config, cache_dir=cache_dir)
                model.transformer = overridden_gpt2
            else:
                print('WARNING: may need to override model to save final layer hidden states correctly.')
        elif model_type == 'bert': # BertForMaskedLM.
            model = AutoModelForMaskedLM.from_pretrained(model_dir, config=config, cache_dir=cache_dir)
        model.resize_token_embeddings(len(tokenizer))
    elif model_type in ['pythia', 'olmo', 'gpt2']:
        # Official model implementations:
        tokenizer_path = model_dir if tokenizer_path_override is None else tokenizer_path_override
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
        model_revision = 'main'
        if (model_type == 'pythia') and (checkpoint is not None):
            model_revision = f'step{checkpoint}'
        if (model_type == 'olmo') and (checkpoint is not None):
            step_str = f'step{checkpoint}'
            model_revision = step_str + '-' + OLMO_TOKENS_MAP[step_str]
        model = AutoModelForCausalLM.from_pretrained(model_dir, revision=model_revision, cache_dir=cache_dir)
        config = AutoConfig.from_pretrained(model_dir, revision=model_revision, cache_dir=cache_dir)
        # If necessary, override so that final layer hidden states are saved
        # before the final layer norm, to be comparable with other layers.
        if override_for_hidden_states:
            if type(model) == GPT2LMHeadModel:
                print('Overriding model to save final layer hidden states correctly.')
                overridden_gpt2 = GPT2ModelOverride.from_pretrained(model_dir, revision=model_revision, config=config, cache_dir=cache_dir)
                model.transformer = overridden_gpt2
            elif type(model) == GPTNeoXForCausalLM:
                print('Overriding model to save final layer hidden states correctly.')
                overridden_gptneox = GPTNeoXModelOverride.from_pretrained(model_dir, revision=model_revision, config=config, cache_dir=cache_dir)
                model.gpt_neox = overridden_gptneox
            elif type(model) == OlmoForCausalLM:
                print('Overriding model to save final layer hidden states correctly.')
                overridden_olmo = OlmoModelOverride.from_pretrained(model_dir, revision=model_revision, config=config, cache_dir=cache_dir)
                model.model = overridden_olmo
            else:
                print('WARNING: may need to override model to save final layer hidden states correctly.')
    # Load onto GPU.
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return config, tokenizer, model


"""
Convert a list of examples (token id lists) to a batch.
Inputs should already include CLS and SEP tokens. Because this function
does not know the maximum sequence length, examples should already be truncated.
All sequences will be padded to the length of the longest example, so this
should be called per batch.
Labels are set to None, assuming these examples are only used for
representation analysis.
"""
def prepare_tokenized_examples(tokenized_examples, tokenizer):
    # Convert into a tensor.
    tensor_examples = [torch.tensor(e, dtype=torch.long) for e in tokenized_examples]
    # Shape: (batch_size, sequence_len).
    input_ids = pad_sequence(tensor_examples, batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    inputs = {'input_ids': input_ids, 'labels': None}
    if torch.cuda.is_available():
        inputs['input_ids'] = inputs['input_ids'].cuda()
    return inputs


"""
DEPRECATED: this function is from older versions of this code, and has some
overhead that might not be consistent across models.
See get_model_surprisals() instead.

Output the surprisals given examples (lists of token_ids). Assumes examples
have been truncated. Outputs tensor of shape (n_tokens) (where
n_tokens concatenates tokens in all examples). The first token in each example
has no surprisal because there is no prediction for the first token.
Handles batching and example tensorizing.
"""
def get_autoregressive_surprisals(model, examples, batch_size, tokenizer):
    # Create batches.
    batches = []
    i = 0
    while i+batch_size <= len(examples):
        batches.append(examples[i:i+batch_size])
        i += batch_size
    if len(examples) % batch_size != 0:
        batches.append(examples[i:])
    # Run evaluation.
    model.eval()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=-1)
        all_surprisals = []
        for batch_i in tqdm(range(len(batches))):
            # Adds padding.
            inputs = prepare_tokenized_examples(batches[batch_i], tokenizer)
            # Run model.
            outputs = model(input_ids=inputs['input_ids'],
                            labels=None,
                            output_hidden_states=True, return_dict=True)
            # Note: logits pre-softmax.
            # Shape: (batch_size, seq_len, vocab_size).
            logits = outputs['logits'].detach()
            logits = logits[:, :-1, :] # Ignore last prediction, because no corresponding label.
            vocab_size = logits.shape[-1]
            del outputs
            # Surprisals for these labels.
            labels = inputs['input_ids'][:, 1:]
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
            labels = labels.flatten()
            labels = labels[labels != -100]
            logits = logits.reshape(-1, vocab_size)
            probs = softmax(logits)[labels != -100, :]
            label_probs = torch.gather(probs, dim=-1, index=labels.reshape(-1, 1)).flatten()
            surprisals = -1.0 * torch.log2(label_probs).cpu()
            all_surprisals.append(np.array(surprisals))
        all_surprisals = np.concatenate(all_surprisals, axis=0)
    print('Computed {} surprisals.'.format(all_surprisals.shape[0]))
    return all_surprisals


"""
Autoregressively generates token ids, given an input prompt (list of token ids).
Returns the generated token ids.
"""
def generate_text(model, input_ids, tokenizer, max_seq_len=128, temperature=0.0):
    input_ids = list(input_ids)
    if len(input_ids) > max_seq_len:
        return input_ids[:max_seq_len]
    # Iteratively fill in tokens.
    output_ids = []
    while len(input_ids) + len(output_ids) < max_seq_len:
        # Deprecated:
        # inputs = prepare_tokenized_examples([input_ids + output_ids], tokenizer)
        # curr_input_ids = inputs['input_ids']
        curr_input_ids = torch.Tensor([input_ids + output_ids]).int().cuda()
        # Note: here, labels are None (because not computing loss).
        outputs = model(input_ids=curr_input_ids,
                        labels=None,
                        output_hidden_states=False, return_dict=True)
        # Note: logits pre-softmax.
        logits = outputs.logits.detach()
        logits = logits[0] # First example only.
        del outputs
        # Logits for the last token.
        index_logits = logits[-1, :]
        softmax = torch.nn.Softmax(dim=0)
        probs = softmax(index_logits)
        if temperature > 0.0:
            probs = torch.pow(probs, 1.0 / temperature)
            fill_id = torch.multinomial(probs, 1).item() # Automatically rescales probs.
        else:
            fill_id = torch.argmax(probs).item()
        output_ids.append(fill_id)
    return output_ids


"""
Get surprisals for a target function that returns probabilities.
"""
def get_target_surprisals(target_fn, inputs_path, batch_size=32):
    # Load eval dataset.
    eval_dataset = read_examples(inputs_path)
    target_surprisals = []  # Per target token.
    n_batches = len(eval_dataset) // batch_size
    if len(eval_dataset) % batch_size != 0: n_batches += 1
    for batch_i in tqdm(range(n_batches)):
        # Get inputs. Assume sequences of same length in input examples.
        input_ids = eval_dataset[batch_size*batch_i:batch_size*(batch_i+1)]
        input_ids = torch.Tensor(input_ids).int()
        target_probs = torch.Tensor(target_fn(input_ids))
        # Get surprisals.
        vocab_size = target_probs.size()[-1]
        labels = input_ids[:, 1:]
        labels = labels.flatten().long().cpu()  # Gather requires int64.
        target_probs = target_probs[:, :-1, :].reshape(-1, vocab_size).cpu()
        target_surprisals_batch = torch.gather(target_probs, dim=-1, index=labels.reshape(-1, 1)).flatten()
        target_surprisals_batch = -1.0 * torch.log2(target_surprisals_batch)
        target_surprisals.append(np.array(target_surprisals_batch))
    target_surprisals = np.concatenate(target_surprisals, axis=0)
    return target_surprisals


"""
Get surprisals for a model.
"""
def get_model_surprisals(model, inputs_path, batch_size=32):
    # Load eval dataset.
    eval_dataset = read_examples(inputs_path)
    model_surprisals = []  # Per target token.
    softmax = torch.nn.Softmax(dim=-1)
    model.eval()
    with torch.no_grad():
        n_batches = len(eval_dataset) // batch_size
        if len(eval_dataset) % batch_size != 0: n_batches += 1
        for batch_i in tqdm(range(n_batches)):
            # Get inputs. Assume sequences of same length in input examples.
            input_ids = eval_dataset[batch_size*batch_i:batch_size*(batch_i+1)]
            input_ids = torch.Tensor(input_ids).int().cuda()
            # Compute model logits and target probs.
            logits = model(input_ids=input_ids).logits
            vocab_size = logits.size()[-1]
            # Get surprisals.
            labels = input_ids[:, 1:]
            labels = labels.flatten().long().cpu()  # Gather requires int64.
            logits = logits[:, :-1, :].reshape(-1, vocab_size)  # Ignore last token per example, because no corresponding target token.
            model_probs = softmax(logits).cpu()
            # Surprisals for model and target fn.
            model_surprisals_batch = torch.gather(model_probs, dim=-1, index=labels.reshape(-1, 1)).flatten()
            model_surprisals_batch = -1.0 * torch.log2(model_surprisals_batch)
            model_surprisals.append(np.array(model_surprisals_batch))
        model_surprisals = np.concatenate(model_surprisals, axis=0)
    return model_surprisals


"""
Get surprisals, and mean cross-entropy between model and target_fn.
"""
def compute_correlation_metrics(model, target_fn, inputs_path,
        batch_size=32):
    # Load eval dataset.
    eval_dataset = read_examples(inputs_path)
    # Run model and target_fn.
    xents = []  # Per batch.
    model_surprisals = []  # Per target token.
    target_surprisals = []  # Per target token.
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    softmax = torch.nn.Softmax(dim=-1)
    model.eval()
    with torch.no_grad():
        n_batches = len(eval_dataset) // batch_size
        if len(eval_dataset) % batch_size != 0: n_batches += 1
        for batch_i in tqdm(range(n_batches)):
            # Get inputs. Assume sequences of same length in input examples.
            # Dummy inputs:
            # input_ids = torch.Tensor(np.zeros((32, 128))).int().cuda()
            input_ids = eval_dataset[batch_size*batch_i:batch_size*(batch_i+1)]
            input_ids = torch.Tensor(input_ids).int().cuda()
            # Compute model logits and target probs.
            logits = model(input_ids=input_ids).logits
            target_probs = torch.Tensor(target_fn(input_ids))
            # May need to remove dims from logits if tokenizer has extra tokens
            # added (e.g. padding vocab to multiple of power of two). This makes
            # logits dims match target_probs.
            vocab_size = target_probs.size()[-1]
            if logits.size()[-1] != vocab_size:
                logits = logits[:, :, :vocab_size]
            # Note: transpose because the class labels/probabilities must be index 1.
            xent_loss = loss_fn(logits.transpose(-1, 1).detach(), target_probs.transpose(-1, 1).cuda())
            xents.append(xent_loss.detach().cpu())
            # Get surprisals.
            labels = input_ids[:, 1:]
            labels = labels.flatten().long().cpu()  # Gather requires int64.
            logits = logits[:, :-1, :].reshape(-1, vocab_size)  # Ignore last token per example, because no corresponding target token.
            model_probs = softmax(logits).cpu()
            # Surprisals for model and target fn.
            model_surprisals_batch = torch.gather(model_probs, dim=-1, index=labels.reshape(-1, 1)).flatten()
            model_surprisals_batch = -1.0 * torch.log2(model_surprisals_batch)
            model_surprisals.append(np.array(model_surprisals_batch))
            target_probs = target_probs[:, :-1, :].reshape(-1, vocab_size).cpu()
            target_surprisals_batch = torch.gather(target_probs, dim=-1, index=labels.reshape(-1, 1)).flatten()
            target_surprisals_batch = -1.0 * torch.log2(target_surprisals_batch)
            target_surprisals.append(np.array(target_surprisals_batch))
    target_surprisals = np.concatenate(target_surprisals, axis=0)
    model_surprisals = np.concatenate(model_surprisals, axis=0)
    return model_surprisals, target_surprisals, np.mean(xents)


"""
Get surprisals and ranks of the input token when using this model. This is
for approximations designed to just output the input token.
Assumes input examples are all the same length.
"""
def compute_input_token_probs(model, inputs_path, batch_size=32):
    # Load eval dataset.
    eval_dataset = read_examples(inputs_path)
    # Run model and target_fn.
    all_surprisals = []  # Per token.
    all_ranks = []  # Per token.
    softmax = torch.nn.Softmax(dim=-1)
    model.eval()
    with torch.no_grad():
        n_batches = len(eval_dataset) // batch_size
        if len(eval_dataset) % batch_size != 0: n_batches += 1
        for batch_i in tqdm(range(n_batches)):
            # Get inputs. Assume sequences of same length in input examples.
            input_ids = eval_dataset[batch_size*batch_i:batch_size*(batch_i+1)]
            input_ids = torch.Tensor(input_ids).int().cuda()
            # Compute model logits.
            logits = model(input_ids=input_ids).logits.detach()
            vocab_size = logits.shape[-1]
            labels = input_ids.flatten().long()  # Just have the input token as target.
            logits = logits.reshape(-1, vocab_size)
            # Compute surprisals.
            probs = softmax(logits)  # Shape: (n_inputs, vocab_size).
            label_probs = torch.gather(probs, dim=-1, index=labels.reshape(-1, 1)).flatten()
            surprisals = -1.0 * torch.log2(label_probs).cpu()
            all_surprisals.append(np.array(surprisals))
            # Compute ranks.
            # Get the token ids sorted by rank (first argsort), then get the
            # indices (i.e. rank because sorted) of increasing values (i.e. the
            # token ids; second argsort). This results in the rank of each
            # token id.
            ranks = logits.argsort(dim=-1, descending=True).argsort(dim=-1, descending=False)
            label_ranks = torch.gather(ranks, dim=-1, index=labels.reshape(-1, 1)).flatten().cpu()
            all_ranks.append(np.array(label_ranks))
            del logits, probs, ranks
    all_surprisals = np.concatenate(all_surprisals, axis=0)
    all_ranks = np.concatenate(all_ranks, axis=0)
    return all_surprisals, all_ranks


"""
DEPRECATED:
Get token bigram scores.
Output shape: (vocab_size), containing cross-entropy values
for all possible context tokens.
"""
def get_bigram_layer_scores(model, tokenizer, bigram_fn, prepend_cls=True):
    xents = -1.0 * np.ones(tokenizer.vocab_size)
    model.eval()
    with torch.no_grad():
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        for token_id in tqdm(np.arange(tokenizer.vocab_size)):
            input_ids = [[token_id]]  # One sequence with one input token.
            if prepend_cls: input_ids[0].insert(0, tokenizer.cls_token_id)
            input_ids = torch.Tensor(input_ids).int().cuda()
            # Compute (masked) model logits and target probs.
            logits = model(input_ids=input_ids).logits[:, [-1], :]
            target_probs = torch.Tensor(bigram_fn(input_ids[:, [-1]]))
            # May need to remove dims from logits if tokenizer has extra tokens
            # added (e.g. padding vocab to multiple of power of two). This makes
            # logits dims match target_probs.
            vocab_size = target_probs.size()[-1]
            if logits.size()[-1] != vocab_size:
                logits = logits[:, :, :vocab_size]
            # Note: transpose because the class labels/probabilities must be index 1.
            xent_loss = loss_fn(logits.transpose(-1, 1), target_probs.transpose(-1, 1).cuda())
            xents[token_id] = xent_loss.cpu()
    return xents


"""
Output the raw token embeddings for the input examples.
"""
def get_token_embeddings(model, examples, model_type):
    if model_type == 'gpt2':
        emb_fn = model.transformer.wte
    elif model_type == 'pythia':
        emb_fn = model.gpt_neox.embed_in
    elif model_type == 'olmo':
        emb_fn = model.model.embed_tokens
    all_token_embs = []
    for ex in examples:
        ex_tensor = torch.tensor(ex, dtype=torch.long).to(device=emb_fn.weight.device)
        token_embs = emb_fn(ex_tensor).detach().cpu().numpy()
        emb_size = token_embs.shape[-1]
        all_token_embs.append(token_embs.reshape(-1, emb_size))
    return np.concatenate(all_token_embs, axis=0)
def get_token_unembeddings_matrix(model, model_type):
    # Output shape: (n_vocab, d).
    if model_type == 'gpt2':
        emb_matrix = model.lm_head.weight.detach().cpu().numpy()
    elif model_type == 'pythia':
        emb_matrix = model.embed_out.weight.detach().cpu().numpy()
    elif model_type == 'olmo':
        emb_matrix = model.lm_head.weight.detach().cpu().numpy()
    return emb_matrix


"""
Output the hidden states given examples (lists of token_ids). Assumes examples
are all the same length. Outputs tensor of shape n_tokens x hidden_size (where
n_tokens concatenates tokens in all examples). Handles batching and example
tensorizing.
"""
def get_hidden_states(model, examples, batch_size, tokenizer, layers):
    # Run evaluation.
    model.eval()
    with torch.no_grad():
        all_hidden_states = [[] for _ in range(len(layers))]
        n_batches = len(examples) // batch_size
        if len(examples) % batch_size != 0: n_batches += 1
        for batch_i in tqdm(range(n_batches)):
            # Get inputs. Assume sequences of same length in input examples.
            input_ids = examples[batch_size*batch_i:batch_size*(batch_i+1)]
            input_ids = torch.Tensor(input_ids).int().cuda()
            # Run model.
            outputs = model(input_ids=input_ids,
                            output_hidden_states=True, return_dict=True)
            # List of n_layers+1 (embedding through last layer), each with shape
            # (n_seqs, seq_len, n_dims).
            hidden_states = outputs['hidden_states']
            del outputs # Delete before the next batch runs.
            for layer_i, layer in enumerate(layers):
                layer_states = hidden_states[layer].detach()
                hidden_size = layer_states.shape[-1]
                layer_states = layer_states.reshape(-1, hidden_size)
                # Remove pad tokens.
                # Shape: n_tokens x hidden_size
                # layer_states = layer_states[nonpad_mask.flatten(), :]
                layer_states = layer_states.detach().cpu().numpy() # Send to CPU so not all need to be held on GPU.
                all_hidden_states[layer_i].append(layer_states)
            del hidden_states, layer_states
    # Concatenate at end, instead of within the loop because this is faster.
    for layer_i in range(len(all_hidden_states)):
        all_hidden_states[layer_i] = np.concatenate(all_hidden_states[layer_i], axis=0)
    print('Extracted {} hidden states per layer.'.format(all_hidden_states[0].shape[0]))
    return all_hidden_states
