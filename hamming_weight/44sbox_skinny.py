import numpy as np
import scipy.stats
import sys
import os
from numpy import random as rd
import csv

# value of noise for which we want to run experiments
SIGMA = np.sqrt(float(sys.argv[1]))
# number of trace to process for each value of noise
N_TRACES = int(sys.argv[2])
# number of experiment to perform for the success rate
N_EXPERIMENTS = int(sys.argv[3])

if len(sys.argv) == 6:
    seed = int(sys.argv[5])
else:
    seed = int.from_bytes(os.urandom(4), sys.byteorder)

OUTPUT_PATH = f"out_{int(SIGMA**2)}_{seed}.csv"

if len(sys.argv) >= 5:
    var_path = sys.argv[4]
    if var_path != "default":
        OUTPUT_PATH = var_path

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
    return [HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_1]], HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_5]],
            HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_13]]]


def leakage_k2(tweakeys_byte, plaintext):
    l_6 = SBOX_TABLE[plaintext[1]] ^ tweakeys_byte
    l_14 = l_6 ^ SBOX_TABLE[plaintext[11]]
    l_2 = l_14 ^ SBOX_TABLE[plaintext[14]]
    return [HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_2]], HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_6]],
            HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_14]]]


def leakage_k3(tweakeys_byte, plaintext):
    c2 = 0x2
    l_7 = SBOX_TABLE[plaintext[2]] ^ tweakeys_byte
    l_15 = l_7 ^ SBOX_TABLE[plaintext[8]] ^ c2
    l_3 = l_15 ^ SBOX_TABLE[plaintext[15]]
    return [HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_3]], HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_7]],
            HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_15]]]


def leakage_k4(tweakeys_byte, plaintext):
    l_8 = SBOX_TABLE[plaintext[3]] ^ tweakeys_byte
    l_16 = l_8 ^ SBOX_TABLE[plaintext[9]]
    l_4 = l_16 ^ SBOX_TABLE[plaintext[12]]
    return [HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_4]], HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_8]],
            HAMMING_WEIGTH_TABLE[SBOX_TABLE[l_16]]]


def leakage_k5(tweakeys_byte, plaintext):
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[SBOX_TABLE[plaintext[4]] ^ SBOX_TABLE[plaintext[11]] ^ tweakeys_byte]]


def leakage_k6(tweakeys_byte, plaintext):
    c2 = 0x2
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[SBOX_TABLE[plaintext[5]] ^ SBOX_TABLE[plaintext[8]] ^ tweakeys_byte ^ c2]]


def leakage_k7(tweakeys_byte, plaintext):
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[SBOX_TABLE[plaintext[6]] ^ SBOX_TABLE[plaintext[9]] ^ tweakeys_byte]]


def leakage_k8(tweakeys_byte, plaintext):
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[SBOX_TABLE[plaintext[7]] ^ SBOX_TABLE[plaintext[10]] ^ tweakeys_byte]]


def leakage_k9(tweakeys_byte, ciphertext):
    l_56 = INV_SBOX_TABLE[ciphertext[6] ^
                          ciphertext[10] ^ ciphertext[14] ^ tweakeys_byte]
    l_55 = INV_SBOX_TABLE[INV_SBOX_TABLE[ciphertext[0]
                                         ^ ciphertext[12]] ^ l_56]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k10(tweakeys_byte, ciphertext):
    c2 = 0x2
    l_56 = INV_SBOX_TABLE[ciphertext[7] ^
                          ciphertext[11] ^ ciphertext[15] ^ tweakeys_byte]
    l_55 = INV_SBOX_TABLE[INV_SBOX_TABLE[ciphertext[1]
                                         ^ ciphertext[13]] ^ l_56 ^ c2]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k11(tweakeys_byte, ciphertext):
    l_56 = INV_SBOX_TABLE[ciphertext[7] ^ tweakeys_byte]
    l_55 = INV_SBOX_TABLE[INV_SBOX_TABLE[ciphertext[2]
                                         ^ ciphertext[14]] ^ l_56]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k12(tweakeys_byte, ciphertext):
    l_56 = INV_SBOX_TABLE[ciphertext[6] ^ tweakeys_byte]
    l_55 = INV_SBOX_TABLE[INV_SBOX_TABLE[ciphertext[1]
                                         ^ ciphertext[13]] ^ l_56]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k13(tweakeys_byte, ciphertext):
    l_56 = INV_SBOX_TABLE[ciphertext[4] ^
                          ciphertext[8] ^ ciphertext[12] ^ tweakeys_byte]
    l_55 = INV_SBOX_TABLE[INV_SBOX_TABLE[ciphertext[2]
                                         ^ ciphertext[14]] ^ l_56]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k14(tweakeys_byte, ciphertext):
    c0 = 0x0A
    l_56 = INV_SBOX_TABLE[ciphertext[4] ^ tweakeys_byte ^ c0]
    l_55 = INV_SBOX_TABLE[l_56 ^
                          INV_SBOX_TABLE[ciphertext[3] ^ ciphertext[15]]]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k15(tweakeys_byte, ciphertext):
    l_56 = INV_SBOX_TABLE[ciphertext[5] ^ tweakeys_byte]
    l_55 = INV_SBOX_TABLE[INV_SBOX_TABLE[ciphertext[0]
                                         ^ ciphertext[12]] ^ l_56]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k16(tweakeys_byte, ciphertext):
    l_56 = INV_SBOX_TABLE[ciphertext[5] ^
                          ciphertext[9] ^ ciphertext[13] ^ tweakeys_byte]
    l_55 = INV_SBOX_TABLE[INV_SBOX_TABLE[ciphertext[3]
                                         ^ ciphertext[15]] ^ l_56]
    return [HAMMING_WEIGTH_TABLE[l_55], HAMMING_WEIGTH_TABLE[l_56]]


def leakage_k1_k10(k1_tk_byte, k10_tk_byte, plaintext):
    c0 = [1, 3]
    lk1 = SBOX_TABLE[plaintext[0]] ^ k1_tk_byte ^ c0[0] ^ SBOX_TABLE[plaintext[10]
                                                                     ] ^ SBOX_TABLE[plaintext[13]]
    lk1 = SBOX_TABLE[lk1]
    lk = lk1 ^ k10_tk_byte ^ c0[1]
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[lk]]


def leakage_k1_k13(k1_tk_byte, k13_tk_byte, ciphertext):
    lk = INV_SBOX_TABLE[ciphertext[4] ^
                        ciphertext[8] ^ ciphertext[12] ^ k13_tk_byte]
    lk = INV_SBOX_TABLE[ciphertext[5] ^ ciphertext[13]
                        ] ^ INV_SBOX_TABLE[ciphertext[2] ^ ciphertext[14]] ^ lk ^ k1_tk_byte
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


def leakage_k2_k9(k2_tk_byte, k9_tk_byte, ciphertext):
    c1 = 2
    lk = INV_SBOX_TABLE[ciphertext[6] ^
                        ciphertext[10] ^ ciphertext[14] ^ k9_tk_byte]
    lk = INV_SBOX_TABLE[ciphertext[7] ^ ciphertext[15]
                        ] ^ INV_SBOX_TABLE[ciphertext[0] ^ ciphertext[12]] ^ lk ^ k2_tk_byte ^ c1
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


def leakage_k2_k16(k2_tk_byte, k16_tk_byte, plaintext):
    lk = SBOX_TABLE[plaintext[1]] ^ k2_tk_byte ^ SBOX_TABLE[plaintext[11]
                                                            ] ^ SBOX_TABLE[plaintext[14]]
    lk = SBOX_TABLE[lk] ^ k16_tk_byte
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[lk]]


def leakage_k3_k9(k3_tk_byte, k9_tk_byte, plaintext):
    c2 = 0x2
    lk = SBOX_TABLE[plaintext[2]] ^ k3_tk_byte ^ SBOX_TABLE[plaintext[8]
                                                            ] ^ c2 ^ SBOX_TABLE[plaintext[15]]
    lk = SBOX_TABLE[lk] ^ k9_tk_byte
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[lk]]


def leakage_k3_k10(k3_tk_byte, k10_tk_byte, ciphertext):
    lk = INV_SBOX_TABLE[ciphertext[7] ^
                        ciphertext[11] ^ ciphertext[15] ^ k10_tk_byte]
    lk = INV_SBOX_TABLE[ciphertext[4] ^ ciphertext[12]
                        ] ^ INV_SBOX_TABLE[ciphertext[1] ^ ciphertext[13]] ^ lk ^ k3_tk_byte
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


def leakage_k4_k14(k4_tk_byte, k14_tk_byte, plaintext):
    lk = SBOX_TABLE[plaintext[3]] ^ k4_tk_byte ^ SBOX_TABLE[plaintext[9]
                                                            ] ^ SBOX_TABLE[plaintext[12]]
    lk = SBOX_TABLE[lk] ^ k14_tk_byte
    return HAMMING_WEIGTH_TABLE[SBOX_TABLE[lk]]


def leakage_k4_k16(k4_tk_byte, k16_tk_byte, ciphertext):
    c0 = 5
    lk = INV_SBOX_TABLE[ciphertext[5] ^
                        ciphertext[9] ^ ciphertext[13] ^ k16_tk_byte] ^ k4_tk_byte ^ c0
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


def leakage_k5_k13(k5_tk_byte, k13_tk_byte, ciphertext):
    lk = INV_SBOX_TABLE[ciphertext[4] ^
                        ciphertext[8] ^ ciphertext[12] ^ k13_tk_byte] ^ k5_tk_byte
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


def leakage_k6_k9(k6_tk_byte, k9_tk_byte, ciphertext):
    lk = INV_SBOX_TABLE[ciphertext[6] ^
                        ciphertext[10] ^ ciphertext[14] ^ k9_tk_byte] ^ k6_tk_byte
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


def leakage_k7_k16(k7_tk_byte, k16_tk_byte, ciphertext):
    c2 = 0x2
    lk = INV_SBOX_TABLE[ciphertext[5] ^
                        ciphertext[9] ^ ciphertext[13] ^ k16_tk_byte]
    lk = INV_SBOX_TABLE[ciphertext[6] ^ ciphertext[14] ^ c2
                        ] ^ INV_SBOX_TABLE[ciphertext[3] ^ ciphertext[15]] ^ lk ^ k7_tk_byte
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


def leakage_k8_k10(k8_tk_byte, k10_tk_byte, ciphertext):
    lk = INV_SBOX_TABLE[ciphertext[7] ^
                        ciphertext[11] ^ ciphertext[15] ^ k10_tk_byte] ^ k8_tk_byte
    return HAMMING_WEIGTH_TABLE[INV_SBOX_TABLE[lk]]


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
        if round == 54:
            round_55 = [HAMMING_WEIGTH_TABLE[x] for x in plaintext]
        if round == 55:
            round_56 = [HAMMING_WEIGTH_TABLE[x] for x in plaintext]
        plaintext = np.array([SBOX_TABLE[p] for p in plaintext])
        # leak on the SBOX output for first rounds
        if round == 1:
            round_2 = [HAMMING_WEIGTH_TABLE[x] for x in plaintext]
        if round == 2:
            round_3 = [HAMMING_WEIGTH_TABLE[x] for x in plaintext]
        plaintext[0] = plaintext[0] ^ c0[round]
        plaintext[4] = plaintext[4] ^ c1[round]
        plaintext[8] = plaintext[8] ^ c2
        plaintext[0:8] = np.bitwise_xor(
            np.bitwise_xor(
                np.bitwise_xor(plaintext[0:8], round_tweakeys[0][round][0:8]), round_tweakeys[1][round][0:8]),
            round_tweakeys[2][round][0:8])
        plaintext = plaintext[shift]
        plaintext = mix_column(plaintext)
    return (round_2, round_3, round_55, round_56, plaintext)

# CLASSES BP


class Edge():
    def __init__(self, var):
        self._previous_message = np.ones(256, dtype=float)
        self._var = var

    def pass_message(self, marg_factor):
        message = np.divide(marg_factor, self._previous_message, out=np.zeros_like(
            marg_factor), where=self._previous_message != 0)
        self._previous_message = marg_factor
        return (message, self._var)


class Clique():
    def __init__(self):
        self.factor = np.ones((256, 256), dtype=float)

    # var = var that should be kept in the message => the opposite of the one being marginalized over
    def emit_message(self, var):
        return np.amax(self.factor, axis=var)

    # var = 0 or var = 1, if message.shape() = (256,256) use var = 0
    def update_factor(self, message, var):
        if var == 0:
            self.factor = np.multiply(self.factor, message)
        else:
            self.factor = np.multiply(self.factor, message[:, np.newaxis])

# BELIEF PROPAGATION

# VAR = 0 => HIGHEST KEY BYTE
# VAR = 1 => LOWEST KEY BYTE


def combine_factors(factors):
    # index of clique and edge is their id, using the graph in the paper
    # from top left to bottom right going vertically first
    # then horizontally going up to the top everytime you change column
    # cliques = (K8-K10, K1-K10, K1-K13, K5-K13, K3-K10, K3-K9, K6-K9,
    # K7-K16, K2-K16, K2-K9, K4-K16, K4-K14) and their edges accordingly
    # in the same order

    cliques = [Clique() for _ in range(12)]
    edges = [Edge(0), Edge(1), Edge(0), Edge(0), Edge(1), Edge(
        0), Edge(0), Edge(0), Edge(1), Edge(0), Edge(1)]

    # INIT TREE

    # read factors from file
    # probably turn them into probability distribution
    # factors in numerical order: K1,K2,...without K11, K12 and K15, not part of the graph
    # followed by numerical order for double bytes K1-K10, K1-K13

    # single byte factors
    # attributed to first clique who contains the variable

    cliques[0].update_factor(factors[7], 1)
    cliques[0].update_factor(factors[9], 0)
    cliques[1].update_factor(factors[0], 1)
    cliques[2].update_factor(factors[10], 0)
    cliques[3].update_factor(factors[4], 1)
    cliques[4].update_factor(factors[2], 1)
    cliques[5].update_factor(factors[8], 0)
    cliques[6].update_factor(factors[5], 1)
    cliques[7].update_factor(factors[6], 1)
    cliques[7].update_factor(factors[12], 0)
    cliques[8].update_factor(factors[1], 1)
    cliques[10].update_factor(factors[3], 1)
    cliques[11].update_factor(factors[11], 0)

    # two bytes factors

    cliques[1].update_factor(factors[13], 0)
    cliques[2].update_factor(factors[14], 0)
    cliques[9].update_factor(factors[15], 0)
    cliques[8].update_factor(factors[16], 0)
    cliques[5].update_factor(factors[17], 0)
    cliques[4].update_factor(factors[18], 0)
    cliques[11].update_factor(factors[19], 0)
    cliques[10].update_factor(factors[20], 0)
    cliques[3].update_factor(factors[21], 0)
    cliques[6].update_factor(factors[22], 0)
    cliques[7].update_factor(factors[23], 0)
    cliques[0].update_factor(factors[24], 0)

    # INIT DONE
    # UPWARD PASS STARTS
    cliques[1].update_factor(
        *edges[0].pass_message(cliques[0].emit_message(0)))
    cliques[2].update_factor(
        *edges[2].pass_message(cliques[3].emit_message(0)))
    cliques[1].update_factor(
        *edges[1].pass_message(cliques[2].emit_message(1)))
    cliques[4].update_factor(
        *edges[3].pass_message(cliques[1].emit_message(0)))
    cliques[5].update_factor(
        *edges[4].pass_message(cliques[4].emit_message(1)))
    cliques[5].update_factor(
        *edges[5].pass_message(cliques[6].emit_message(0)))
    cliques[9].update_factor(
        *edges[6].pass_message(cliques[5].emit_message(0)))
    cliques[8].update_factor(
        *edges[7].pass_message(cliques[7].emit_message(0)))
    cliques[8].update_factor(
        *edges[8].pass_message(cliques[9].emit_message(1)))
    cliques[10].update_factor(
        *edges[9].pass_message(cliques[8].emit_message(0)))
    cliques[11].update_factor(
        *edges[10].pass_message(cliques[10].emit_message(1)))

    # UPWARD PASS DONE
    # DOWNWARD PASS STARTS

    cliques[10].update_factor(
        *edges[10].pass_message(cliques[11].emit_message(1)))
    cliques[8].update_factor(
        *edges[9].pass_message(cliques[10].emit_message(0)))
    cliques[7].update_factor(
        *edges[7].pass_message(cliques[8].emit_message(0)))
    cliques[9].update_factor(
        *edges[8].pass_message(cliques[8].emit_message(1)))
    cliques[5].update_factor(
        *edges[6].pass_message(cliques[9].emit_message(0)))
    cliques[6].update_factor(
        *edges[5].pass_message(cliques[5].emit_message(0)))
    cliques[4].update_factor(
        *edges[4].pass_message(cliques[5].emit_message(1)))
    cliques[1].update_factor(
        *edges[3].pass_message(cliques[4].emit_message(0)))
    cliques[0].update_factor(
        *edges[0].pass_message(cliques[1].emit_message(0)))
    cliques[2].update_factor(
        *edges[1].pass_message(cliques[1].emit_message(1)))
    cliques[3].update_factor(
        *edges[2].pass_message(cliques[2].emit_message(0)))

    # DOWNWARD PASS ENDS
    # THE TREE IS CALIBRATED
    # EXTRACT MARGINALS IN ORDER

    marginals = []
    marginals.append(cliques[1].emit_message(1))
    marginals.append(cliques[8].emit_message(1))
    marginals.append(cliques[4].emit_message(1))
    marginals.append(cliques[10].emit_message(1))
    marginals.append(cliques[3].emit_message(1))
    marginals.append(cliques[6].emit_message(1))
    marginals.append(cliques[7].emit_message(1))
    marginals.append(cliques[0].emit_message(1))
    marginals.append(cliques[5].emit_message(0))
    marginals.append(cliques[0].emit_message(0))
    marginals.append(cliques[2].emit_message(0))
    marginals.append(cliques[11].emit_message(0))
    marginals.append(cliques[10].emit_message(0))

    return marginals

# CONSTANTS


pdfs = [scipy.stats.norm(
    loc=i, scale=SIGMA) for i in range(0, 9)]

lfsrk = np.arange(start=0, stop=256, step=1, dtype=int)
fifty_six_round_k = [skinny_lfsr_3(k, 28) for k in lfsrk]
fifty_five_round_k = [skinny_lfsr_3(k, 27) for k in lfsrk]
three_round_k = [skinny_lfsr_3(k, 1) for k in lfsrk]

# FILE HEADER

f = open(OUTPUT_PATH, 'a')
with f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["sigma", SIGMA])
    writer.writerow(["number of traces", N_TRACES])
    writer.writerow(["seed", seed])

# MAIN LOOP



for experiment in range(N_EXPERIMENTS):
    # the PRNG is reseeded with seed + experiment to allow for
    # easy inspection of individual experiment
    rd_state = rd.default_rng(seed+experiment)
    secret = rd_state.integers(0, 256, 16)
    # fixed tweaks, random, known by the attack
    tk1 = rd_state.integers(0, 256, 16)
    tk2 = rd_state.integers(0, 256, 16)
    round_tweakeys = compute_round_tweakeys(tk1, tk2, secret)

    # holder for the bytes scores

    scores_k1 = np.array([(i, 0) for i in range(256)], dtype=[
                        ('key', int), ('score', float)])
    scores_k2 = np.array([(i, 0) for i in range(256)], dtype=[
                        ('key', int), ('score', float)])
    scores_k3 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k4 = np.array([(i, 0) for i in range(256)], dtype=[
                        ('key', int), ('score', float)])
    scores_k5 = np.array([(i, 0) for i in range(256)], dtype=[
                        ('key', int), ('score', float)])
    scores_k6 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k7 = np.array([(i, 0) for i in range(256)], dtype=[
                        ('key', int), ('score', float)])
    scores_k8 = np.array([(i, 0) for i in range(256)], dtype=[
                        ('key', int), ('score', float)])
    scores_k9 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k10 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k11 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k12 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k13 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k14 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k15 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])
    scores_k16 = np.array([(i, 0) for i in range(256)], dtype=[
        ('key', int), ('score', float)])

    # holder for the double bytes scores

    scores_k1_k10 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k1_k13 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k2_k9 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k2_k16 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k3_k9 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k3_k10 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k4_k14 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k4_k16 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k5_k13 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k6_k9 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k7_k16 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])
    scores_k8_k10 = np.array([(i, 0) for i in range(65536)], dtype=[
        ('key', int), ('score', float)])

    # holders for the exp_sr

    exp_post_ranks = []    

    for t in range(N_TRACES):
        factors = []
        plaintext = rd_state.integers(0, 256, 16)
        (round_2, round_3, round_55, round_56, ciphertext) = encrypt_skinny_with_leakages(
            round_tweakeys, plaintext)

        # K1
        trace_k1 = rd_state.normal(
            loc=[round_2[0], round_2[4], round_2[12]], scale=SIGMA, size=3)
        eval_k1 = [[pdf.pdf(trace_k1[0]) for pdf in pdfs], [pdf.pdf(
            trace_k1[1]) for pdf in pdfs], [pdf.pdf(trace_k1[2]) for pdf in pdfs]]
        predicted_k1s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k1s = [leakage_k1(
            round_tweakeys[0][0][0] ^ round_tweakeys[1][0][0] ^ k, plaintext) for k in predicted_k1s]
        k1s = np.array(
            [[eval_k1[0][k[0]] for k in predicted_k1s],
             [eval_k1[1][k[1]] for k in predicted_k1s],
             [eval_k1[2][k[2]] for k in predicted_k1s]], dtype=float)
        k1s = np.prod(k1s, axis=0)
        k1s = np.log(k1s)
        scores_k1['score'] += k1s
        tmp_scores_k1 = scores_k1['score'] - np.max(scores_k1['score'])
        tmp_scores_k1 = np.exp(tmp_scores_k1)
        tmp_scores_k1 = tmp_scores_k1 / np.sum(tmp_scores_k1)
        factors.append(tmp_scores_k1)

        # K2
        trace_k2 = rd_state.normal(
            loc=[round_2[1], round_2[5], round_2[13]], scale=SIGMA, size=3)
        eval_k2 = [[pdf.pdf(trace_k2[0]) for pdf in pdfs], [pdf.pdf(
            trace_k2[1]) for pdf in pdfs], [pdf.pdf(trace_k2[2]) for pdf in pdfs]]
        predicted_k2s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k2s = [leakage_k2(
            round_tweakeys[0][0][1] ^ round_tweakeys[1][0][1] ^ k, plaintext) for k in predicted_k2s]
        k2s = np.array(
            [[eval_k2[0][k[0]] for k in predicted_k2s],
             [eval_k2[1][k[1]] for k in predicted_k2s],
             [eval_k2[2][k[2]] for k in predicted_k2s]], dtype=float)
        k2s = np.prod(k2s, axis=0)
        k2s = np.log(k2s)
        scores_k2['score'] += k2s
        tmp_scores_k2 = scores_k2['score'] - np.max(scores_k2['score'])
        tmp_scores_k2 = np.exp(tmp_scores_k2)
        tmp_scores_k2 = tmp_scores_k2 / np.sum(tmp_scores_k2)
        factors.append(tmp_scores_k2)

        # K3
        trace_k3 = rd_state.normal(
            loc=[round_2[2], round_2[6], round_2[14]], scale=SIGMA, size=3)
        eval_k3 = [[pdf.pdf(trace_k3[0]) for pdf in pdfs], [pdf.pdf(
            trace_k3[1]) for pdf in pdfs], [pdf.pdf(trace_k3[2]) for pdf in pdfs]]
        predicted_k3s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k3s = [leakage_k3(
            round_tweakeys[0][0][2] ^ round_tweakeys[1][0][2] ^ k, plaintext) for k in predicted_k3s]
        k3s = np.array(
            [[eval_k3[0][k[0]] for k in predicted_k3s],
             [eval_k3[1][k[1]] for k in predicted_k3s],
             [eval_k3[2][k[2]] for k in predicted_k3s]], dtype=float)
        k3s = np.prod(k3s, axis=0)
        k3s = np.log(k3s)
        scores_k3['score'] += k3s
        tmp_scores_k3 = scores_k3['score'] - np.max(scores_k3['score'])
        tmp_scores_k3 = np.exp(tmp_scores_k3)
        tmp_scores_k3 = tmp_scores_k3 / np.sum(tmp_scores_k3)
        factors.append(tmp_scores_k3)

        # K4
        trace_k4 = rd_state.normal(
            loc=[round_2[3], round_2[7], round_2[15]], scale=SIGMA, size=3)
        eval_k4 = [[pdf.pdf(trace_k4[0]) for pdf in pdfs], [pdf.pdf(
            trace_k4[1]) for pdf in pdfs], [pdf.pdf(trace_k4[2]) for pdf in pdfs]]
        predicted_k4s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k4s = [leakage_k4(
            round_tweakeys[0][0][3] ^ round_tweakeys[1][0][3] ^ k, plaintext) for k in predicted_k4s]
        k4s = np.array(
            [[eval_k4[0][k[0]] for k in predicted_k4s],
             [eval_k4[1][k[1]] for k in predicted_k4s],
             [eval_k4[2][k[2]] for k in predicted_k4s]], dtype=float)
        k4s = np.prod(k4s, axis=0)
        k4s = np.log(k4s)
        scores_k4['score'] += k4s
        tmp_scores_k4 = scores_k4['score'] - np.max(scores_k4['score'])
        tmp_scores_k4 = np.exp(tmp_scores_k4)
        tmp_scores_k4 = tmp_scores_k4 / np.sum(tmp_scores_k4)
        factors.append(tmp_scores_k4)

        # K5
        trace_k5 = rd_state.normal(loc=round_2[9], scale=SIGMA, size=1)
        eval_k5 = [pdf.pdf(trace_k5) for pdf in pdfs]
        predicted_k5s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k5s = [leakage_k5(
            round_tweakeys[0][0][4] ^ round_tweakeys[1][0][4] ^ k, plaintext) for k in predicted_k5s]
        k5s = np.array(
            [eval_k5[k] for k in predicted_k5s], dtype=float)
        k5s = k5s.flatten(order="C")
        k5s = np.log(k5s)
        scores_k5['score'] += k5s
        tmp_scores_k5 = scores_k5['score'] - np.max(scores_k5['score'])
        tmp_scores_k5 = np.exp(tmp_scores_k5)
        tmp_scores_k5 = tmp_scores_k5 / np.sum(tmp_scores_k5)
        factors.append(tmp_scores_k5)

        # K6
        trace_k6 = rd_state.normal(loc=round_2[10], scale=SIGMA, size=1)
        eval_k6 = [pdf.pdf(trace_k6) for pdf in pdfs]
        predicted_k6s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k6s = [leakage_k6(
            round_tweakeys[0][0][5] ^ round_tweakeys[1][0][5] ^ k, plaintext) for k in predicted_k6s]
        k6s = np.array(
            [eval_k6[k] for k in predicted_k6s], dtype=float)
        k6s = k6s.flatten(order="C")
        k6s = np.log(k6s)
        scores_k6['score'] += k6s
        tmp_scores_k6 = scores_k6['score'] - np.max(scores_k6['score'])
        tmp_scores_k6 = np.exp(tmp_scores_k6)
        tmp_scores_k6 = tmp_scores_k6 / np.sum(tmp_scores_k6)
        factors.append(tmp_scores_k6)

        # K7
        trace_k7 = rd_state.normal(loc=round_2[11], scale=SIGMA, size=1)
        eval_k7 = [pdf.pdf(trace_k7) for pdf in pdfs]
        predicted_k7s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k7s = [leakage_k7(
            round_tweakeys[0][0][6] ^ round_tweakeys[1][0][6] ^ k, plaintext) for k in predicted_k7s]
        k7s = np.array(
            [eval_k7[k] for k in predicted_k7s], dtype=float)
        k7s = k7s.flatten(order="C")
        k7s = np.log(k7s)
        scores_k7['score'] += k7s
        tmp_scores_k7 = scores_k7['score'] - np.max(scores_k7['score'])
        tmp_scores_k7 = np.exp(tmp_scores_k7)
        tmp_scores_k7 = tmp_scores_k7 / np.sum(tmp_scores_k7)
        factors.append(tmp_scores_k7)

        # K8
        trace_k8 = rd_state.normal(loc=round_2[8], scale=SIGMA, size=1)
        eval_k8 = [pdf.pdf(trace_k8) for pdf in pdfs]
        predicted_k8s = np.arange(start=0, stop=256, step=1, dtype=int)
        predicted_k8s = [leakage_k8(
            round_tweakeys[0][0][7] ^ round_tweakeys[1][0][7] ^ k, plaintext) for k in predicted_k8s]
        k8s = np.array(
            [eval_k8[k] for k in predicted_k8s], dtype=float)
        k8s = k8s.flatten(order="C")
        k8s = np.log(k8s)
        scores_k8['score'] += k8s
        tmp_scores_k8 = scores_k8['score'] - np.max(scores_k8['score'])
        tmp_scores_k8 = np.exp(tmp_scores_k8)
        tmp_scores_k8 = tmp_scores_k8 / np.sum(tmp_scores_k8)
        factors.append(tmp_scores_k8)

        # K9
        trace_k9 = rd_state.normal(
            loc=[round_55[11], round_56[5]], scale=SIGMA, size=2)
        eval_k9 = [[pdf.pdf(trace_k9[0]) for pdf in pdfs], [
            pdf.pdf(trace_k9[1]) for pdf in pdfs]]
        predicted_k9s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k9s = [leakage_k9(
            round_tweakeys[0][55][5] ^ round_tweakeys[1][55][5] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k9s]
        k9s = np.array(
            [[eval_k9[0][k[0]] for k in predicted_k9s],
             [eval_k9[1][k[1]] for k in predicted_k9s]], dtype=float)
        k9s = np.prod(k9s, axis=0)
        k9s = np.log(k9s)
        scores_k9['score'] += k9s
        tmp_scores_k9 = scores_k9['score'] - np.max(scores_k9['score'])
        tmp_scores_k9 = np.exp(tmp_scores_k9)
        tmp_scores_k9 = tmp_scores_k9 / np.sum(tmp_scores_k9)
        factors.append(tmp_scores_k9)

        # K10
        trace_k10 = rd_state.normal(
            loc=[round_55[8], round_56[6]], scale=SIGMA, size=2)
        eval_k10 = [[pdf.pdf(trace_k10[0]) for pdf in pdfs], [
            pdf.pdf(trace_k10[1]) for pdf in pdfs]]
        predicted_k10s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k10s = [leakage_k10(
            round_tweakeys[0][55][6] ^ round_tweakeys[1][55][6] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k10s]
        k10s = np.array(
            [[eval_k10[0][k[0]] for k in predicted_k10s],
             [eval_k10[1][k[1]] for k in predicted_k10s]], dtype=float)
        k10s = np.prod(k10s, axis=0)
        k10s = np.log(k10s)
        scores_k10['score'] += k10s
        tmp_scores_k10 = scores_k10['score'] - np.max(scores_k10['score'])
        tmp_scores_k10 = np.exp(tmp_scores_k10)
        tmp_scores_k10 = tmp_scores_k10 / np.sum(tmp_scores_k10)
        factors.append(tmp_scores_k10)

        # K11
        trace_k11 = rd_state.normal(
            loc=[round_55[12], round_56[3]], scale=SIGMA, size=2)
        eval_k11 = [[pdf.pdf(trace_k11[0]) for pdf in pdfs], [
            pdf.pdf(trace_k11[1]) for pdf in pdfs]]
        predicted_k11s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k11s = [leakage_k11(
            round_tweakeys[0][55][3] ^ round_tweakeys[1][55][3] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k11s]
        k11s = np.array(
            [[eval_k11[0][k[0]] for k in predicted_k11s],
             [eval_k11[1][k[1]] for k in predicted_k11s]], dtype=float)
        k11s = np.prod(k11s, axis=0)
        k11s = np.log(k11s)
        scores_k11['score'] += k11s

        # K12
        trace_k12 = rd_state.normal(
            loc=[round_55[15], round_56[2]], scale=SIGMA, size=2)
        eval_k12 = [[pdf.pdf(trace_k12[0]) for pdf in pdfs], [
            pdf.pdf(trace_k12[1]) for pdf in pdfs]]
        predicted_k12s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k12s = [leakage_k12(
            round_tweakeys[0][55][2] ^ round_tweakeys[1][55][2] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k12s]
        k12s = np.array(
            [[eval_k12[0][k[0]] for k in predicted_k12s],
             [eval_k12[1][k[1]] for k in predicted_k12s]], dtype=float)
        k12s = np.prod(k12s, axis=0)
        k12s = np.log(k12s)
        scores_k12['score'] += k12s

        # K13
        trace_k13 = rd_state.normal(
            loc=[round_55[9], round_56[7]], scale=SIGMA, size=2)
        eval_k13 = [[pdf.pdf(trace_k13[0]) for pdf in pdfs], [
            pdf.pdf(trace_k13[1]) for pdf in pdfs]]
        predicted_k13s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k13s = [leakage_k13(
            round_tweakeys[0][55][7] ^ round_tweakeys[1][55][7] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k13s]
        k13s = np.array(
            [[eval_k13[0][k[0]] for k in predicted_k13s],
             [eval_k13[1][k[1]] for k in predicted_k13s]], dtype=float)
        k13s = np.prod(k13s, axis=0)
        k13s = np.log(k13s)
        scores_k13['score'] += k13s
        tmp_scores_k13 = scores_k13['score'] - np.max(scores_k13['score'])
        tmp_scores_k13 = np.exp(tmp_scores_k13)
        tmp_scores_k13 = tmp_scores_k13 / np.sum(tmp_scores_k13)
        factors.append(tmp_scores_k13)

        # K14
        trace_k14 = rd_state.normal(
            loc=[round_55[13], round_56[0]], scale=SIGMA, size=2)
        eval_k14 = [[pdf.pdf(trace_k14[0]) for pdf in pdfs], [
            pdf.pdf(trace_k14[1]) for pdf in pdfs]]
        predicted_k14s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k14s = [leakage_k14(
            round_tweakeys[0][55][0] ^ round_tweakeys[1][55][0] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k14s]
        k14s = np.array(
            [[eval_k14[0][k[0]] for k in predicted_k14s],
             [eval_k14[1][k[1]] for k in predicted_k14s]], dtype=float)
        k14s = np.prod(k14s, axis=0)
        k14s = np.log(k14s)
        scores_k14['score'] += k14s
        tmp_scores_k14 = scores_k14['score'] - np.max(scores_k14['score'])
        tmp_scores_k14 = np.exp(tmp_scores_k14)
        tmp_scores_k14 = tmp_scores_k14 / np.sum(tmp_scores_k14)
        factors.append(tmp_scores_k14)

        # K15
        trace_k15 = rd_state.normal(
            loc=[round_55[14], round_56[1]], scale=SIGMA, size=2)
        eval_k15 = [[pdf.pdf(trace_k15[0]) for pdf in pdfs], [
            pdf.pdf(trace_k15[1]) for pdf in pdfs]]
        predicted_k15s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k15s = [leakage_k15(
            round_tweakeys[0][55][1] ^ round_tweakeys[1][55][1] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k15s]
        k15s = np.array(
            [[eval_k15[0][k[0]] for k in predicted_k15s],
             [eval_k15[1][k[1]] for k in predicted_k15s]], dtype=float)
        k15s = np.prod(k15s, axis=0)
        k15s = np.log(k15s)
        scores_k15['score'] += k15s

        # K16
        trace_k16 = rd_state.normal(
            loc=[round_55[10], round_56[4]], scale=SIGMA, size=2)
        eval_k16 = [[pdf.pdf(trace_k16[0]) for pdf in pdfs], [
            pdf.pdf(trace_k16[1]) for pdf in pdfs]]
        predicted_k16s = np.arange(
            start=0, stop=256, step=1, dtype=int)
        predicted_k16s = [leakage_k16(
            round_tweakeys[0][55][4] ^ round_tweakeys[1][55][4] ^ fifty_six_round_k[k], ciphertext) for k in predicted_k16s]
        k16s = np.array(
            [[eval_k16[0][k[0]] for k in predicted_k16s],
             [eval_k16[1][k[1]] for k in predicted_k16s]], dtype=float)
        k16s = np.prod(k16s, axis=0)
        k16s = np.log(k16s)
        scores_k16['score'] += k16s
        tmp_scores_k16 = scores_k16['score'] - np.max(scores_k16['score'])
        tmp_scores_k16 = np.exp(tmp_scores_k16)
        tmp_scores_k16 = tmp_scores_k16 / np.sum(tmp_scores_k16)
        factors.append(tmp_scores_k16)

        # K1-K10
        trace_k1_k10 = rd_state.normal(loc=round_3[4], scale=SIGMA, size=1)
        eval_k1_k10 = [pdf.pdf(trace_k1_k10) for pdf in pdfs]
        predicted_k1_k10s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k1_k10s = [leakage_k1_k10(
            round_tweakeys[0][0][0] ^ round_tweakeys[1][0][0] ^ (k >> 8),
            round_tweakeys[0][1][0] ^ round_tweakeys[1][1][0] ^ three_round_k[k & 0xff], plaintext) for k in predicted_k1_k10s]
        k1_k10s = np.array(
            [eval_k1_k10[k] for k in predicted_k1_k10s], dtype=float)
        k1_k10s = k1_k10s.flatten(order="C")
        k1_k10s = np.log(k1_k10s)
        scores_k1_k10['score'] += k1_k10s
        tmp_scores_k1_k10 = scores_k1_k10['score'] - \
            np.max(scores_k1_k10['score'])
        tmp_scores_k1_k10 = np.exp(tmp_scores_k1_k10)
        tmp_scores_k1_k10 = tmp_scores_k1_k10 / np.sum(tmp_scores_k1_k10)
        factors.append(tmp_scores_k1_k10.reshape((256, 256)))

        # K1-K13
        trace_k1_k13 = rd_state.normal(loc=round_55[6], scale=SIGMA, size=1)
        eval_k1_k13 = [pdf.pdf(trace_k1_k13) for pdf in pdfs]
        predicted_k1_k13s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k1_k13s = [leakage_k1_k13(
            round_tweakeys[0][54][6] ^ round_tweakeys[1][54][6] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][7] ^ round_tweakeys[1][55][7] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k1_k13s]
        k1_k13s = np.array(
            [eval_k1_k13[k] for k in predicted_k1_k13s], dtype=float)
        k1_k13s = k1_k13s.flatten(order="C")
        k1_k13s = np.log(k1_k13s)
        scores_k1_k13['score'] += k1_k13s
        tmp_scores_k1_k13 = scores_k1_k13['score'] - \
            np.max(scores_k1_k13['score'])
        tmp_scores_k1_k13 = np.exp(tmp_scores_k1_k13)
        tmp_scores_k1_k13 = tmp_scores_k1_k13 / np.sum(tmp_scores_k1_k13)
        factors.append(tmp_scores_k1_k13.reshape((256, 256)))

        # K2-K9
        trace_k2_k9 = rd_state.normal(loc=round_55[4], scale=SIGMA, size=1)
        eval_k2_k9 = [pdf.pdf(trace_k2_k9) for pdf in pdfs]
        predicted_k2_k9s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k2_k9s = [leakage_k2_k9(
            round_tweakeys[0][54][4] ^ round_tweakeys[1][54][4] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][5] ^ round_tweakeys[1][55][5] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k2_k9s]
        k2_k9s = np.array(
            [eval_k2_k9[k] for k in predicted_k2_k9s], dtype=float)
        k2_k9s = k2_k9s.flatten(order="C")
        k2_k9s = np.log(k2_k9s)
        scores_k2_k9['score'] += k2_k9s
        tmp_scores_k2_k9 = scores_k2_k9['score'] - \
            np.max(scores_k2_k9['score'])
        tmp_scores_k2_k9 = np.exp(tmp_scores_k2_k9)
        tmp_scores_k2_k9 = tmp_scores_k2_k9 / np.sum(tmp_scores_k2_k9)
        factors.append(tmp_scores_k2_k9.reshape((256, 256)))

        # K2-K16
        trace_k2_k16 = rd_state.normal(loc=round_3[5], scale=SIGMA, size=1)
        eval_k2_k16 = [pdf.pdf(trace_k2_k16) for pdf in pdfs]
        predicted_k2_k16s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k2_k16s = [leakage_k2_k16(
            round_tweakeys[0][0][1] ^ round_tweakeys[1][0][1] ^ (k >> 8),
            round_tweakeys[0][1][1] ^ round_tweakeys[1][1][1] ^ three_round_k[k & 0xff], plaintext) for k in predicted_k2_k16s]
        k2_k16s = np.array(
            [eval_k2_k16[k] for k in predicted_k2_k16s], dtype=float)
        k2_k16s = k2_k16s.flatten(order="C")
        k2_k16s = np.log(k2_k16s)
        scores_k2_k16['score'] += k2_k16s
        tmp_scores_k2_k16 = scores_k2_k16['score'] - \
            np.max(scores_k2_k16['score'])
        tmp_scores_k2_k16 = np.exp(tmp_scores_k2_k16)
        tmp_scores_k2_k16 = tmp_scores_k2_k16 / np.sum(tmp_scores_k2_k16)
        factors.append(tmp_scores_k2_k16.reshape((256, 256)))

        # K3-K9
        trace_k3_k9 = rd_state.normal(loc=round_3[6], scale=SIGMA, size=1)
        eval_k3_k9 = [pdf.pdf(trace_k3_k9) for pdf in pdfs]
        predicted_k3_k9s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k3_k9s = [leakage_k3_k9(
            round_tweakeys[0][0][2] ^ round_tweakeys[1][0][2] ^ (k >> 8),
            round_tweakeys[0][1][2] ^ round_tweakeys[1][1][2] ^ three_round_k[k & 0xff], plaintext) for k in predicted_k3_k9s]
        k3_k9s = np.array(
            [eval_k3_k9[k] for k in predicted_k3_k9s], dtype=float)
        k3_k9s = k3_k9s.flatten(order="C")
        k3_k9s = np.log(k3_k9s)
        scores_k3_k9['score'] += k3_k9s
        tmp_scores_k3_k9 = scores_k3_k9['score'] - \
            np.max(scores_k3_k9['score'])
        tmp_scores_k3_k9 = np.exp(tmp_scores_k3_k9)
        tmp_scores_k3_k9 = tmp_scores_k3_k9 / np.sum(tmp_scores_k3_k9)
        factors.append(tmp_scores_k3_k9.reshape((256, 256)))

        # K3-K10
        trace_k3_k10 = rd_state.normal(loc=round_55[5], scale=SIGMA, size=1)
        eval_k3_k10 = [pdf.pdf(trace_k3_k10) for pdf in pdfs]
        predicted_k3_k10s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k3_k10s = [leakage_k3_k10(
            round_tweakeys[0][54][5] ^ round_tweakeys[1][54][5] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][6] ^ round_tweakeys[1][55][6] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k3_k10s]
        k3_k10s = np.array(
            [eval_k3_k10[k] for k in predicted_k3_k10s], dtype=float)
        k3_k10s = k3_k10s.flatten(order="C")
        k3_k10s = np.log(k3_k10s)
        scores_k3_k10['score'] += k3_k10s
        tmp_scores_k3_k10 = scores_k3_k10['score'] - \
            np.max(scores_k3_k10['score'])
        tmp_scores_k3_k10 = np.exp(tmp_scores_k3_k10)
        tmp_scores_k3_k10 = tmp_scores_k3_k10 / np.sum(tmp_scores_k3_k10)
        factors.append(tmp_scores_k3_k10.reshape((256, 256)))

        # K4-K14
        trace_k4_k14 = rd_state.normal(loc=round_3[7], scale=SIGMA, size=1)
        eval_k4_k14 = [pdf.pdf(trace_k4_k14) for pdf in pdfs]
        predicted_k4_k14s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k4_k14s = [leakage_k4_k14(
            round_tweakeys[0][0][3] ^ round_tweakeys[1][0][3] ^ (k >> 8),
            round_tweakeys[0][1][3] ^ round_tweakeys[1][1][3] ^ three_round_k[k & 0xff], plaintext) for k in predicted_k4_k14s]
        k4_k14s = np.array(
            [eval_k4_k14[k] for k in predicted_k4_k14s], dtype=float)
        k4_k14s = k4_k14s.flatten(order="C")
        k4_k14s = np.log(k4_k14s)
        scores_k4_k14['score'] += k4_k14s
        tmp_scores_k4_k14 = scores_k4_k14['score'] - \
            np.max(scores_k4_k14['score'])
        tmp_scores_k4_k14 = np.exp(tmp_scores_k4_k14)
        tmp_scores_k4_k14 = tmp_scores_k4_k14 / np.sum(tmp_scores_k4_k14)
        factors.append(tmp_scores_k4_k14.reshape((256, 256)))

        # K4-K16
        trace_k4_k16 = rd_state.normal(loc=round_55[0], scale=SIGMA, size=1)
        eval_k4_k16 = [pdf.pdf(trace_k4_k16) for pdf in pdfs]
        predicted_k4_k16s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k4_k16s = [leakage_k4_k16(
            round_tweakeys[0][54][0] ^ round_tweakeys[1][54][0] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][4] ^ round_tweakeys[1][55][4] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k4_k16s]
        k4_k16s = np.array(
            [eval_k4_k16[k] for k in predicted_k4_k16s], dtype=float)
        k4_k16s = k4_k16s.flatten(order="C")
        k4_k16s = np.log(k4_k16s)
        scores_k4_k16['score'] += k4_k16s
        tmp_scores_k4_k16 = scores_k4_k16['score'] - \
            np.max(scores_k4_k16['score'])
        tmp_scores_k4_k16 = np.exp(tmp_scores_k4_k16)
        tmp_scores_k4_k16 = tmp_scores_k4_k16 / np.sum(tmp_scores_k4_k16)
        factors.append(tmp_scores_k4_k16.reshape((256, 256)))

        # K5-K13
        trace_k5_k13 = rd_state.normal(loc=round_55[3], scale=SIGMA, size=1)
        eval_k5_k13 = [pdf.pdf(trace_k5_k13) for pdf in pdfs]
        predicted_k5_k13s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k5_k13s = [leakage_k5_k13(
            round_tweakeys[0][54][3] ^ round_tweakeys[1][54][3] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][7] ^ round_tweakeys[1][55][7] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k5_k13s]
        k5_k13s = np.array(
            [eval_k5_k13[k] for k in predicted_k5_k13s], dtype=float)
        k5_k13s = k5_k13s.flatten(order="C")
        k5_k13s = np.log(k5_k13s)
        scores_k5_k13['score'] += k5_k13s
        tmp_scores_k5_k13 = scores_k5_k13['score'] - \
            np.max(scores_k5_k13['score'])
        tmp_scores_k5_k13 = np.exp(tmp_scores_k5_k13)
        tmp_scores_k5_k13 = tmp_scores_k5_k13 / np.sum(tmp_scores_k5_k13)
        factors.append(tmp_scores_k5_k13.reshape((256, 256)))

        # K6-K9
        trace_k6_k9 = rd_state.normal(loc=round_55[1], scale=SIGMA, size=1)
        eval_k6_k9 = [pdf.pdf(trace_k6_k9) for pdf in pdfs]
        predicted_k6_k9s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k6_k9s = [leakage_k6_k9(
            round_tweakeys[0][54][1] ^ round_tweakeys[1][54][1] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][5] ^ round_tweakeys[1][55][5] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k6_k9s]
        k6_k9s = np.array(
            [eval_k6_k9[k] for k in predicted_k6_k9s], dtype=float)
        k6_k9s = k6_k9s.flatten(order="C")
        k6_k9s = np.log(k6_k9s)
        scores_k6_k9['score'] += k6_k9s
        tmp_scores_k6_k9 = scores_k6_k9['score'] - \
            np.max(scores_k6_k9['score'])
        tmp_scores_k6_k9 = np.exp(tmp_scores_k6_k9)
        tmp_scores_k6_k9 = tmp_scores_k6_k9 / np.sum(tmp_scores_k6_k9)
        factors.append(tmp_scores_k6_k9.reshape((256, 256)))

        # K7-K16
        trace_k7_k16 = rd_state.normal(loc=round_55[7], scale=SIGMA, size=1)
        eval_k7_k16 = [pdf.pdf(trace_k7_k16) for pdf in pdfs]
        predicted_k7_k16s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k7_k16s = [leakage_k7_k16(
            round_tweakeys[0][54][7] ^ round_tweakeys[1][54][7] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][4] ^ round_tweakeys[1][55][4] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k7_k16s]
        k7_k16s = np.array(
            [eval_k7_k16[k] for k in predicted_k7_k16s], dtype=float)
        k7_k16s = k7_k16s.flatten(order="C")
        k7_k16s = np.log(k7_k16s)
        scores_k7_k16['score'] += k7_k16s
        tmp_scores_k7_k16 = scores_k7_k16['score'] - \
            np.max(scores_k7_k16['score'])
        tmp_scores_k7_k16 = np.exp(tmp_scores_k7_k16)
        tmp_scores_k7_k16 = tmp_scores_k7_k16 / np.sum(tmp_scores_k7_k16)
        factors.append(tmp_scores_k7_k16.reshape((256, 256)))

        # K8-K10
        trace_k8_k10 = rd_state.normal(loc=round_55[2], scale=SIGMA, size=1)
        eval_k8_k10 = [pdf.pdf(trace_k8_k10) for pdf in pdfs]
        predicted_k8_k10s = np.arange(start=0, stop=65536, step=1, dtype=int)
        predicted_k8_k10s = [leakage_k8_k10(
            round_tweakeys[0][54][2] ^ round_tweakeys[1][54][2] ^ fifty_five_round_k[k >> 8],
            round_tweakeys[0][55][6] ^ round_tweakeys[1][55][6] ^ fifty_six_round_k[k & 0xff], ciphertext) for k in predicted_k8_k10s]
        k8_k10s = np.array(
            [eval_k8_k10[k] for k in predicted_k8_k10s], dtype=float)
        k8_k10s = k8_k10s.flatten(order="C")
        k8_k10s = np.log(k8_k10s)
        scores_k8_k10['score'] += k8_k10s
        tmp_scores_k8_k10 = scores_k8_k10['score'] - \
            np.max(scores_k8_k10['score'])
        tmp_scores_k8_k10 = np.exp(tmp_scores_k8_k10)
        tmp_scores_k8_k10 = tmp_scores_k8_k10 / np.sum(tmp_scores_k8_k10)
        factors.append(tmp_scores_k8_k10.reshape((256, 256)))

        # K11, K12, K15 not part of the graph so not evaluated in the BP
        cut_secret = np.delete(secret, [10, 11, 14])
        marginals = combine_factors(factors)
        ranks_post = []
        for key_bytes in range(13):
            scores = np.array([(i, marginals[key_bytes][i]) for i in range(256)], dtype=[
                ('key', int), ('score', float)])
            scores = np.sort(scores, order='score')[::-1]
            ranks_post.append(next(x for x, y in enumerate(
                scores) if y['key'] == cut_secret[key_bytes]))

        # Now we do K11, K12, K15
        tmp_scores_k11 = np.sort(scores_k11, order='score')[::-1]
        rank_k11 = next(x for x, y in enumerate(
                tmp_scores_k11) if y['key'] == secret[10])
        ranks_post.insert(10,rank_k11)
        tmp_scores_k12 = np.sort(scores_k12, order='score')[::-1]
        rank_k12 = next(x for x, y in enumerate(
                tmp_scores_k12) if y['key'] == secret[11])
        ranks_post.insert(11,rank_k12)
        tmp_scores_k15 = np.sort(scores_k15, order='score')[::-1]
        rank_k15 = next(x for x, y in enumerate(
                tmp_scores_k15) if y['key'] == secret[14])
        ranks_post.insert(14,rank_k15)
        exp_post_ranks.append(ranks_post)

    exp_post_ranks = np.array(exp_post_ranks)
    
    # output
    f = open(OUTPUT_PATH, 'a')
    with f:
        for k in range(16):
            writer.writerow(exp_post_ranks[:,k])