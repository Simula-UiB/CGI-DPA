import numpy as np
import scipy.stats
import sys
import os
from numpy import random as rd

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})


# value of noise for which we want to run experiments
SIGMA = np.sqrt(float(sys.argv[1]))
# number of trace to process for each value of noise
N_TRACES = int(sys.argv[2])
# number of experiment to perform for the success rate
N_EXPERIMENTS = int(sys.argv[3])

if len(sys.argv) == 5:
    seed = int(sys.argv[4])
else:
    seed = int.from_bytes(os.urandom(4), sys.byteorder)

HAMMING_WEIGTH_TABLE = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
                        2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
                        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
                        3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3,
                        4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
                        5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3,
                        4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
                        5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4,
                        5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5,
                        6, 6, 7, 6, 7, 7, 8]

SBOX_TABLE = [0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b,
              0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b,
              0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9,
              0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9,
              0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d,
              0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d,
              0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd,
              0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd,
              0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29,
              0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
              0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb,
              0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb,
              0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f,
              0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f,
              0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf,
              0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff]


INV_SBOX_TABLE = [0xac, 0xe8, 0x68, 0x3c, 0x6c, 0x38, 0xa8, 0xec, 0xaa, 0xae, 0x3a, 0x3e, 0x6a, 0x6e, 0xea, 0xee,
                  0xa6, 0xa3, 0x33, 0x36, 0x66, 0x63, 0xe3, 0xe6, 0xe1, 0xa4, 0x61, 0x34, 0x31, 0x64, 0xa1, 0xe4,
                  0x8d, 0xc9, 0x49, 0x1d, 0x4d, 0x19, 0x89, 0xcd, 0x8b, 0x8f, 0x1b, 0x1f, 0x4b, 0x4f, 0xcb, 0xcf,
                  0x85, 0xc0, 0x40, 0x15, 0x45, 0x10, 0x80, 0xc5, 0x82, 0x87, 0x12, 0x17, 0x42, 0x47, 0xc2, 0xc7,
                  0x96, 0x93, 0x03, 0x06, 0x56, 0x53, 0xd3, 0xd6, 0xd1, 0x94, 0x51, 0x04, 0x01, 0x54, 0x91, 0xd4,
                  0x9c, 0xd8, 0x58, 0x0c, 0x5c, 0x08, 0x98, 0xdc, 0x9a, 0x9e, 0x0a, 0x0e, 0x5a, 0x5e, 0xda, 0xde,
                  0x95, 0xd0, 0x50, 0x05, 0x55, 0x00, 0x90, 0xd5, 0x92, 0x97, 0x02, 0x07, 0x52, 0x57, 0xd2, 0xd7,
                  0x9d, 0xd9, 0x59, 0x0d, 0x5d, 0x09, 0x99, 0xdd, 0x9b, 0x9f, 0x0b, 0x0f, 0x5b, 0x5f, 0xdb, 0xdf,
                  0x16, 0x13, 0x83, 0x86, 0x46, 0x43, 0xc3, 0xc6, 0x41, 0x14, 0xc1, 0x84, 0x11, 0x44, 0x81, 0xc4,
                  0x1c, 0x48, 0xc8, 0x8c, 0x4c, 0x18, 0x88, 0xcc, 0x1a, 0x1e, 0x8a, 0x8e, 0x4a, 0x4e, 0xca, 0xce,
                  0x35, 0x60, 0xe0, 0xa5, 0x65, 0x30, 0xa0, 0xe5, 0x32, 0x37, 0xa2, 0xa7, 0x62, 0x67, 0xe2, 0xe7,
                  0x3d, 0x69, 0xe9, 0xad, 0x6d, 0x39, 0xa9, 0xed, 0x3b, 0x3f, 0xab, 0xaf, 0x6b, 0x6f, 0xeb, 0xef,
                  0x26, 0x23, 0xb3, 0xb6, 0x76, 0x73, 0xf3, 0xf6, 0x71, 0x24, 0xf1, 0xb4, 0x21, 0x74, 0xb1, 0xf4,
                  0x2c, 0x78, 0xf8, 0xbc, 0x7c, 0x28, 0xb8, 0xfc, 0x2a, 0x2e, 0xba, 0xbe, 0x7a, 0x7e, 0xfa, 0xfe,
                  0x25, 0x70, 0xf0, 0xb5, 0x75, 0x20, 0xb0, 0xf5, 0x22, 0x27, 0xb2, 0xb7, 0x72, 0x77, 0xf2, 0xf7,
                  0x2d, 0x79, 0xf9, 0xbd, 0x7d, 0x29, 0xb9, 0xfd, 0x2b, 0x2f, 0xbb, 0xbf, 0x7b, 0x7f, 0xfb, 0xff]

# KEY BYTES MODELS


def leakage_k1(tweakeys_byte, plaintext):
    c0 = 1
    l_5 = SBOX_TABLE[plaintext[0]] ^ tweakeys_byte ^ c0
    l_13 = l_5 ^ SBOX_TABLE[plaintext[10]]
    l_1 = l_13 ^ SBOX_TABLE[plaintext[13]]
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_1]]

def leakage_k9(tweakeys_byte, ciphertext):
    l_56 = INV_SBOX_TABLE[ciphertext[6] ^
                          ciphertext[10] ^ ciphertext[14] ^ tweakeys_byte]
    return HAMMING_WEIGTH_TABLE[l_56]


# SKINNY FUNCTIONS


def skinny_lfsr_3(byte, n_application):
    for _ in range(n_application):
        x6 = (byte >> 6) & 0x01
        x0 = byte & 0x01
        top = (x0 ^ x6) << 7
        byte = (byte >> 1) ^ top
    return byte


def compute_round_tweakeys(tk1, tk2, tk3):
    permutation = np.array(
        [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7])
    round_tweakeys = [[], [], []]
    for _ in range(56):
        round_tweakeys[0].append(tk1)
        tk1 = tk1[permutation]
        round_tweakeys[1].append(tk2)
        tk2 = tk2[permutation]
        for i in range(8):
            x5 = (tk2[i] >> 5) & 0x01
            x7 = (tk2[i] >> 7) & 0x01
            bottom = x7 ^ x5
            tk2[i] = ((tk2[i] << 1) ^ bottom) % 256
        round_tweakeys[2].append(tk3)
        tk3 = tk3[permutation]
        for i in range(8):
            x6 = (tk3[i] >> 6) & 0x01
            x0 = tk3[i] & 0x01
            top = (x0 ^ x6) << 7
            tk3[i] = (tk3[i] >> 1) ^ top
    return round_tweakeys


def mix_column(state):
    new_state = np.array([0 for _ in range(16)], dtype=int)
    new_state[4:8] = state[0:4]
    new_state[8:12] = np.bitwise_xor(state[4:8], state[8:12])
    new_state[12:16] = np.bitwise_xor(state[0:4], state[8:12])
    new_state[0:4] = np.bitwise_xor(new_state[12:16], state[12:16])
    return new_state


def encrypt_skinny_with_leakages(round_tweakeys, plaintext):
    c0 = [1, 3, 7, 15, 15, 14, 13, 11, 7, 15, 14, 12, 9, 3, 7, 14, 13, 10, 5, 11, 6, 12, 8, 0, 1, 2, 5, 11, 7, 14, 12, 8,
          1, 3, 6, 13, 11, 6, 13, 10, 4, 9, 2, 4, 8, 1, 2, 4, 9, 3, 6, 12, 9, 2, 5, 10]
    c1 = [0, 0, 0, 0, 1, 3, 3, 3, 3, 2, 1, 3, 3, 3, 2, 0, 1, 3, 3, 2, 1, 2, 1, 3, 2, 0, 0, 0, 1, 2, 1, 3, 3, 2, 0, 0, 1,
          3, 2, 1, 3, 2, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 3, 2, 0]
    c2 = 0x2
    shift = np.array([0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12])
    for round in range(56):
        # leak on the SBOX input for last rounds
        if round == 55:
            round_56 = [HAMMING_WEIGTH_TABLE[x] for x in plaintext]
        plaintext = np.array([SBOX_TABLE[p] for p in plaintext])
        # leak on the SBOX output for first rounds
        if round == 1:
            round_2 = [HAMMING_WEIGTH_TABLE[x] for x in plaintext]
        plaintext[0] = plaintext[0] ^ c0[round]
        plaintext[4] = plaintext[4] ^ c1[round]
        plaintext[8] = plaintext[8] ^ c2
        plaintext[0:8] = np.bitwise_xor(
            np.bitwise_xor(
                np.bitwise_xor(plaintext[0:8], round_tweakeys[0][round][0:8]), round_tweakeys[1][round][0:8]),
            round_tweakeys[2][round][0:8])
        plaintext = plaintext[shift]
        plaintext = mix_column(plaintext)
    return (round_2, round_56, plaintext)


# CONSTANTS
pdfs = [scipy.stats.norm(
    loc=i, scale=SIGMA) for i in range(0, 9)]

last_round_k = np.arange(start=0, stop=256, step=1, dtype=int)
last_round_k = [skinny_lfsr_3(k, 28) for k in last_round_k]

# MAIN LOOP

success_rate_k1 = np.zeros(N_TRACES, dtype=float)
success_rate_k9 = np.zeros(N_TRACES, dtype=float)

for experiment in range(N_EXPERIMENTS):
    # the PRNG is reseeded with seed + experiment to allow for
    # easy inspection of individual experiment
    rd_state = rd.default_rng(seed+experiment)
    secret = rd_state.integers(0, 256, 16)
    # fixed tweaks, random, known by the attack
    tk1 = rd_state.integers(0, 256, 16)
    tk2 = rd_state.integers(0, 256, 16)
    round_tweakeys = compute_round_tweakeys(tk1, tk2, secret)

    # holder for the ranks
    rank_k1 = []
    rank_k9 = []
    
    # holder for the scores

    scores_k1 = np.array([(i, 0) for i in range(256)], dtype=[
                        ('key', int), ('score', float)])
    scores_k9 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    for t in range(N_TRACES):
        plaintext = rd_state.integers(0, 256, 16)
        (round_2, round_56, ciphertext) = encrypt_skinny_with_leakages(
            round_tweakeys, plaintext)

        # K1
        trace_k1 = rd_state.normal(
            loc=round_2[0], scale=SIGMA, size=1)
        eval_k1 = [pdf.pdf(trace_k1[0]) for pdf in pdfs]
        predicted_k1s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k1s = [leakage_k1(
            round_tweakeys[0][0][0] ^ round_tweakeys[1][0][0] ^ k, plaintext) for k in predicted_k1s]
        k1s = np.array([eval_k1[k] for k in predicted_k1s], dtype=float)
        k1s = k1s.flatten(order="C")
        k1s = np.log(k1s)
        scores_k1['score'] += k1s
        tmp_scores_k1 = np.sort(scores_k1, order='score')[::-1]
        rank_k1.append(next(x for x, y in enumerate(
            tmp_scores_k1) if y['key'] == secret[0]))

               
        # K9
        trace_k9 = rd_state.normal(
            loc=round_56[5], scale=SIGMA, size=1)
        eval_k9 = [pdf.pdf(trace_k9[0]) for pdf in pdfs]
        predicted_k9s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k9s = [leakage_k9(
            round_tweakeys[0][55][5] ^ round_tweakeys[1][55][5] ^ last_round_k[k], ciphertext) for k in predicted_k9s]
        k9s = np.array([eval_k9[k] for k in predicted_k9s], dtype=float)
        k9s = k9s.flatten(order="C")
        k9s = np.log(k9s)
        scores_k9['score'] += k9s
        tmp_scores_k9 = np.sort(scores_k9, order='score')[::-1]
        rank_k9.append(next(x for x, y in enumerate(
            tmp_scores_k9) if y['key'] == secret[8]))
        
    success_rate_k1 += (np.array(rank_k1) == 0).astype(float)
    success_rate_k9 += (np.array(rank_k9) == 0).astype(float)

success_rate_k1 = success_rate_k1/ N_EXPERIMENTS
success_rate_k1 = np.log2(success_rate_k1) * -1
success_rate_k9 = success_rate_k9/ N_EXPERIMENTS
success_rate_k9 = np.log2(success_rate_k9) * -1

success_rate = np.zeros(N_TRACES, dtype=float)
success_rate += success_rate_k1 *8
success_rate += success_rate_k9 *8
print(success_rate) 
