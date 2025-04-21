"""
Computes the overlap between two circuits, and the distribution of overlaps if
both circuits were randomly distributed within each parameter block.
"""

import codecs
import numpy as np
import os
import pickle
from tqdm import tqdm

from utils.circuit_utils import convert_mask_names


N_SAMPLES = 10000
TO_RUN = []

# Overlap between bigram and initialization (ckpt0) circuits with trained embeddings.
# TO_RUN.append(('../trained_circuits_pythia_160m/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#                '../experimental/random_init_except_embs/pythia_160m/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#                'pythia',
#                '../overlap_results/pythia_160m_bigram_init_sample.npy'))
# TO_RUN.append(('../trained_circuits_gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#               '../experimental/random_init_except_embs/gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#               'gpt2',
#               '../overlap_results/gpt2_ours_bigram_init_sample.npy'))
# TO_RUN.append(('../trained_circuits_pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
#                '../experimental/random_init_except_embs/pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
#                'pythia',
#                '../overlap_results/pythia_1b_bigram_init_sample.npy'))

# Overlap between bigram and initialization (ckpt0) circuits.
# TO_RUN.append(('../trained_circuits_pythia_160m/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#                '../trained_circuits_pythia_160m/bigram_ckpt0_t1e-3_lambda1e1_lr5e-5.pickle',
#                'pythia',
#                '../overlap_results/pythia_160m_bigram_ckpt0_sample.npy'))
# TO_RUN.append(('../trained_circuits_gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#               '../trained_circuits_gpt2_ours/bigram_ckpt0_t1e-3_lambda1e1_lr5e-5.pickle',
#               'gpt2',
#               '../overlap_results/gpt2_ours_bigram_ckpt0_sample.npy'))
# TO_RUN.append(('../trained_circuits_pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
#                '../trained_circuits_pythia_1b/bigram_ckpt0_t1e-3_lambda1e2_lr5e-5.pickle',
#                'pythia',
#                '../overlap_results/pythia_1b_bigram_ckpt0_sample.npy'))

# Overlap between bigram and distillation circuits.
TO_RUN.append(('../trained_circuits_pythia_160m/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
               '../experimental/distillation/pythia_160m/distill_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
               'pythia',
               '../overlap_results/pythia_160m_bigram_distill_sample.npy'))
TO_RUN.append(('../trained_circuits_gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
              '../experimental/distillation/gpt2_ours/distill_ckptNone_t1e-3_lambda5e1_lr5e-5.pickle',
              'gpt2',
              '../overlap_results/gpt2_ours_bigram_distill_sample.npy'))
TO_RUN.append(('../trained_circuits_gpt2_small/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
              '../experimental/distillation/gpt2_small/distill_ckptNone_t1e-3_lambda5e1_lr5e-5.pickle',
              'gpt2',
              '../overlap_results/gpt2_small_bigram_distill_sample.npy'))
TO_RUN.append(('../trained_circuits_gpt2_large/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
              '../experimental/distillation/gpt2_large/distill_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
              'gpt2',
              '../overlap_results/gpt2_large_bigram_distill_sample.npy'))
TO_RUN.append(('../trained_circuits_pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
               '../experimental/distillation/pythia_1b/distill_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
               'pythia',
               '../overlap_results/pythia_1b_bigram_distill_sample.npy'))

# Overlap between bigram and random init circuits.
# TO_RUN.append(('../trained_circuits_pythia_160m/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#                '../experimental/true_random_init_except_embs/pythia_160m/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#                'pythia',
#                '../overlap_results/pythia_160m_bigram_randinit_sample.npy'))
# TO_RUN.append(('../trained_circuits_gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#               '../experimental/true_random_init_except_embs/gpt2_ours/bigram_ckptNone_t1e-3_lambda1e1_lr5e-5.pickle',
#               'gpt2',
#               '../overlap_results/gpt2_ours_bigram_randinit_sample.npy'))
# TO_RUN.append(('../trained_circuits_pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
#                '../experimental/true_random_init_except_embs/pythia_1b/bigram_ckptNone_t1e-3_lambda1e2_lr5e-5.pickle',
#                'pythia',
#                '../overlap_results/pythia_1b_bigram_randinit_sample.npy'))


# Run.
def compute_overlap(dict0, dict1):
    n_overlap = 0
    n_total = 0
    for k, v0 in dict0.items():
        n_total += len(v0.flatten())
        v1 = dict1[k]
        n_overlap += np.sum(v0 & v1)
    return n_overlap / n_total
def compute_totals(dict0):
    n_kept = 0
    n_total = 0
    for k, v0 in dict0.items():
        n_total += len(v0.flatten())
        n_kept += np.sum(v0)
    return n_kept, n_total
for circuit_path0, circuit_path1, model_type, sample_outpath in TO_RUN:
    with open(circuit_path0, 'rb') as f:
        circuit_dict0 = pickle.load(f)
    with open(circuit_path1, 'rb') as f:
        circuit_dict1 = pickle.load(f)
    circuit_dict0 = convert_mask_names(circuit_dict0, model_type)
    circuit_dict1 = convert_mask_names(circuit_dict1, model_type)
    actual_overlap = compute_overlap(circuit_dict0, circuit_dict1)
    circuit0_total, total_mask_params = compute_totals(circuit_dict0)
    circuit1_total, _ = compute_totals(circuit_dict1)
    if os.path.isfile(sample_outpath):
        samples = np.load(sample_outpath, allow_pickle=False)
        assert len(samples) == N_SAMPLES
    else:
        # Not flattening, so shuffling is just shuffling rows. Otherwise, quite slow.
        # Flatten dict entries for shuffling.
        # for k in list(circuit_dict0.keys()):
        #     circuit_dict0[k] = circuit_dict0[k].flatten()
        #     circuit_dict1[k] = circuit_dict1[k].flatten()

        # Keep arrays in contiguous memory; this significantly improves speed
        # if they are not already kept contiguously.
        for k in list(circuit_dict0.keys()):
            circuit_dict0[k] = np.ascontiguousarray(circuit_dict0[k])
            circuit_dict1[k] = np.ascontiguousarray(circuit_dict1[k])

        # Run shuffled overlap samples.
        samples = -1.0 * np.ones(N_SAMPLES)
        for sample_i in tqdm(np.arange(N_SAMPLES)):
            for k in list(circuit_dict0.keys()):
                np.random.shuffle(circuit_dict0[k])
                np.random.shuffle(circuit_dict1[k])
            sample_overlap = compute_overlap(circuit_dict0, circuit_dict1)
            samples[sample_i] = sample_overlap
        np.save(sample_outpath, samples, allow_pickle=False)
    expected_overlap = np.mean(samples)
    # Probability that random sample overlap >= actual_overlap.
    p_greater = np.sum(samples >= actual_overlap) / N_SAMPLES
    # Then, the probability that we would get such a high overlap due to chance
    # is just p_greater.
    print(f'Overlap between:\n{circuit_path0},\n{circuit_path1}:\nOverlap: {actual_overlap} (expected: {expected_overlap}) (p={p_greater})')
    # Note that the proportion of path0 that is included in path1 is:
    # (actual_overlap * total_mask_params) / circuit0_total
    print(f'Proportion {actual_overlap*total_mask_params/circuit0_total} of circuit0 contained in circuit1 (which is {circuit1_total/total_mask_params} of total params).\n')
