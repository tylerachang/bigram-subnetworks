"""
N-gram model class.
"""

import codecs
import gc
import numpy as np
import os
import pickle
from tqdm import tqdm

from collections import defaultdict, Counter


class NGramModel:
    # The parameters reference_path, prune_every, and
    # prune_minimum are only required if computing ngram counts from scratch
    # (not cached).
    def __init__(self, cache_dir, filename_prefix, ngram_n,
                 vocab_size, reference_path='',
                 prune_every=1000000, prune_minimum=2, array_topk=10000):
        self.cache_dir = cache_dir
        self.filename_prefix = filename_prefix
        self.ngram_n = ngram_n
        self.vocab_size = vocab_size
        # Load or compute ngram models.
        os.makedirs(self.cache_dir, exist_ok=True)
        self.ngram_models = []
        for ngram_i in range(1, self.ngram_n+1):
            self.ngram_models.append(self.load_or_compute_ngram_model(
                    ngram_i, reference_path=reference_path,
                    prune_every=prune_every,
                    prune_minimum=prune_minimum,
                    array_topk=array_topk))

    def load_or_compute_ngram_model(self, ngram_n,
            reference_path='', prune_every=1000000, prune_minimum=2,
            array_topk=10000):
        ngrams = self.load_ngram_counts(ngram_n)
        if ngrams is None:
            ngrams = self.compute_ngram_counts(
                    ngram_n, reference_path,
                    prune_every=prune_every, prune_minimum=prune_minimum)
        # Convert counts to conditional probabilities.
        # Entry i_0, ..., i_{n-1} is the probability of
        # i_{n-1} given i_0, ..., i_{n-2}.
        print('Converting counts to probabilities.')
        for context_key in tqdm(ngrams):
            # Convert the counts to probabilities.
            counts = ngrams[context_key]
            total = np.sum(list(counts.values()))
            # Note: using a dictionary here because many entries are likely
            # empty.
            probs_dict = defaultdict(lambda: 0.0)
            for target_key, count in counts.items():
                prob = count / total
                probs_dict[target_key] = prob
            ngrams[context_key] = probs_dict
        if array_topk <= 0: return ngrams
        # The top array_topk entries are saved as array rather than dict for
        # efficiency. Sort by the number of keys in the ngram count
        # distribution.
        print(f'Converting top {array_topk} to arrays.')
        array_keys = sorted(ngrams.keys(), key=lambda k: len(ngrams[k]),
                            reverse=True)[:array_topk]
        for context_key in tqdm(array_keys):
            prob_dict = ngrams[context_key]
            array = np.zeros(self.vocab_size)
            for token_id, prob in prob_dict.items():
                if token_id == self.vocab_size: continue
                array[token_id] = prob
            ngrams[context_key] = array
        return ngrams

    # Assume compute_ngram_counts has already been run.
    # Otherwise, returns None.
    def load_ngram_counts(self, ngram_n):
        # Dictionary mapping context tuples to Counters:
        # ngrams[(i-n+1, ..., i-1)][i] = count
        # Note: for unigrams, the first key is an empty tuple.
        ngrams_path = os.path.join(
                self.cache_dir,
                f'{self.filename_prefix}_{ngram_n}gram_counts.pickle')
        # Load from cache if possible.
        if not os.path.isfile(ngrams_path): return None
        print(f'Loading {ngram_n}-gram counts.')
        with open(ngrams_path, 'rb') as handle:
            ngrams = pickle.load(handle)
        ngrams.default_factory = lambda: Counter()
        return ngrams

    # Reference path should contain tokenized (space-separated int) sequences,
    # one per line.
    def compute_ngram_counts(self, ngram_n, reference_path,
                             prune_every, prune_minimum):
        print(f'Computing {ngram_n}-gram counts.')
        # Function to prune the ngrams dictionary.
        # Prunes anything with count 1.
        def prune_ngrams(ngrams, min_count=2):
            if min_count is None:
                # No pruning.
                return ngrams
            context_keys_to_remove = []
            for context, counts in ngrams.items():
                target_keys_to_remove = []
                for target, count in counts.items():
                    if count < min_count:
                        target_keys_to_remove.append(target)
                for target in target_keys_to_remove:
                    counts.pop(target)
                del target_keys_to_remove
                # If all zero, prune this entire counter.
                if len(counts) == 0:
                    context_keys_to_remove.append(context)
            for context in context_keys_to_remove:
                ngrams.pop(context)
            # To resize the dictionary in memory after the removed keys.
            ngrams = ngrams.copy()
            del context_keys_to_remove
            gc.collect()
            return ngrams
        # Count ngrams. Create dictionary mapping:
        # ngrams[(i-n+1, ..., i-1)][i] = count
        # Note: for unigrams, the first key is an empty tuple.
        ngrams = defaultdict(lambda: Counter())
        reference_file = codecs.open(reference_path, 'rb', encoding='utf-8')
        line_count = 0
        for line_i, line in tqdm(enumerate(reference_file)):
            stripped_line = line.strip()
            if stripped_line == "":
                continue
            sequence = [int(token_id) for token_id in stripped_line.split()]
            # Initialize with the extra pre-sequence tokens.
            # This represents the token_ids for the current ngram_n positions.
            curr = np.ones(ngram_n, dtype=int) * self.vocab_size
            for token_id in sequence:
                # Increment to the next token.
                curr = np.roll(curr, -1)
                curr[-1] = token_id
                # Increment the corresponding ngram:
                ngrams[tuple(curr[:-1])][curr[-1]] += 1
            # Pruning.
            line_count += 1
            if line_count % prune_every == 0:
                print('Pruning ngram counts <{}.'.format(prune_minimum))
                orig_len = len(ngrams)
                ngrams = prune_ngrams(ngrams, min_count=prune_minimum)
                print('Pruned: {0} keys -> {1} keys.'.format(orig_len, len(ngrams)))
        print('Final prune: pruning ngram counts <{}.'.format(prune_minimum))
        orig_len = len(ngrams)
        ngrams = prune_ngrams(ngrams, min_count=prune_minimum)
        print('Pruned: {0} keys -> {1} keys.'.format(orig_len, len(ngrams)))
        # To allow pickling.
        ngrams.default_factory = None
        # Save.
        ngrams_path = os.path.join(
                self.cache_dir,
                f'{self.filename_prefix}_{ngram_n}gram_counts.pickle')
        with open(ngrams_path, 'wb') as handle:
            pickle.dump(ngrams, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ngrams.default_factory = lambda: Counter()
        return ngrams

    # Given input sequences (n_sequences, n_tokens), returns the probabilities
    # with shape (n_sequences, n_tokens) (probabilities conditioned on n-gram
    # context tokens). For un-observed contexts, probability is np.nan.
    # Use get_ngram_probs_with_backoff() to ensure no np.nans.
    #
    # If prefilled is passed in, only replaces np.nan values in the input array.
    def get_ngram_probs(self, sequences, ngram_n, prefilled=None):
        assert ngram_n <= self.ngram_n
        ngrams = self.ngram_models[ngram_n-1]
        if prefilled is None:
            to_return = np.nan * np.ones(tuple(sequences.shape))
        else:
            to_return = prefilled
        for sequence_i, sequence in enumerate(sequences):
            if np.all(~np.isnan(to_return[sequence_i, :])): continue
            # Fill previous tokens with placeholder.
            curr = np.ones(ngram_n, dtype=int) * self.vocab_size
            for token_i, token_id in enumerate(sequence):
                # Increment to the next token.
                curr = np.roll(curr, -1)
                curr[-1] = token_id
                # Skip if already filled.
                if not np.isnan(to_return[sequence_i, token_i]): continue
                # Update with conditional prob.
                context_key = tuple(curr[:-1])
                if context_key in ngrams:
                    prob_dict = ngrams[context_key]
                    conditional_prob = prob_dict[curr[-1]]
                    to_return[sequence_i, token_i] = conditional_prob
        return to_return

    # Returns n-gram probabilities as in get_ngram_probs(), but using backoff
    # to ensure no np.nans. For backoff, unobserved ngram contexts use the
    # probabilities for ngram n-1.
    def get_ngram_probs_with_backoff(self, sequences):
        to_return = None
        has_nan = True
        curr_ngram_n = self.ngram_n
        while curr_ngram_n > 0 and has_nan:
            to_return = self.get_ngram_probs(sequences, curr_ngram_n, prefilled=to_return)
            has_nan = np.any(np.isnan(to_return))
            curr_ngram_n -= 1
        # Any remaining nans are unobserved contexts; uniform distribution.
        to_return[np.isnan(to_return)] = 1.0 / self.vocab_size
        return to_return

    # Given input tokens (n_sequences, n_tokens), returns the probabilities
    # of next tokens, with shape (n_sequences, n_tokens, vocab_size).
    # Probabilities conditioned on n-gram context input; for un-observed
    # contexts, probability is all np.nan.
    # Use ngram_predict_with_backoff() to ensure no np.nans.
    #
    # If prefilled is passed in, only replaces np.nan values in the input array.
    def ngram_predict(self, sequences, ngram_n, prefilled=None):
        assert ngram_n <= self.ngram_n
        ngrams = self.ngram_models[ngram_n-1]
        if prefilled is None:
            to_return = np.nan * np.ones(tuple(list(sequences.shape) + [self.vocab_size]))
        else:
            to_return = prefilled
        for sequence_i, sequence in enumerate(sequences):
            if np.all(~np.isnan(to_return[sequence_i, :, 0])): continue
            # Fill previous tokens with placeholder.
            curr = np.ones(ngram_n-1, dtype=int) * self.vocab_size
            for token_i, token_id in enumerate(sequence):
                # Increment to the next token.
                curr = np.roll(curr, -1)
                if len(curr) > 0: curr[-1] = token_id
                # Skip if already filled.
                if not np.isnan(to_return[sequence_i, token_i, 0]): continue
                # Update with conditional prob.
                context_key = tuple(curr)
                if context_key in ngrams:
                    prob_dict = ngrams[context_key]
                    # Either dictionary or array.
                    if isinstance(prob_dict, dict):
                        to_return[sequence_i, token_i, :] = 0.0
                        for token_id, prob in prob_dict.items():
                            if token_id == self.vocab_size: continue
                            to_return[sequence_i, token_i, token_id] = prob
                    else:
                        to_return[sequence_i, token_i, :] = prob_dict
        return to_return

    # Returns n-gram probabilities as in ngram_predict(), but using backoff
    # to ensure no np.nans. For backoff, unobserved ngram contexts use the
    # probabilities for ngram n-1.
    def ngram_predict_with_backoff(self, sequences):
        to_return = None
        has_nan = True
        curr_ngram_n = self.ngram_n
        while curr_ngram_n > 0 and has_nan:
            to_return = self.ngram_predict(sequences, curr_ngram_n,
                    prefilled=to_return)
            has_nan = np.any(np.isnan(to_return))
            curr_ngram_n -= 1
        # Any remaining nans are unobserved contexts; uniform distribution.
        to_return[np.isnan(to_return)] = 1.0 / self.vocab_size
        return to_return
