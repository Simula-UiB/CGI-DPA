from threadpoolctl import threadpool_limits
import numpy as np
import argparse
import csv
from skinny import SkinnyCipher
import os
import collections
SBOX = SkinnyCipher.sbox8
SBOX_INV = SkinnyCipher.sbox8_inv

#HW

def hw(x): 
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
    return HAMMING_WEIGTH_TABLE[x]

# SKINNY FUNCTIONS


def skinny_lfsr_3(byte, n_application):
    for _ in range(n_application):
        x6 = (byte >> 6) & 0x01
        x0 = byte & 0x01
        top = (x0 ^ x6) << 7
        byte = (byte >> 1) ^ top
    return byte


# precomputation to avoid running the lfsr when computing tk_k9-16
lfsrk = np.arange(start=0, stop=256, step=1, dtype=int)
last_round_k = [skinny_lfsr_3(k, 28) for k in lfsrk]
fifty_five_round_k =  [skinny_lfsr_3(k, 27) for k in lfsrk]
third_round_k = [skinny_lfsr_3(k, 1) for k in lfsrk]

def compute_tk1_tk2(tk1, tk2):
    permutation = np.array(
            [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7])
    round_tweakeys = [[], []]
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

    return round_tweakeys

# LEAKAGE MODELS


def k1(tweakeys_byte, plaintext):
    c0 = 1
    l_5 = SBOX[plaintext[0]] ^ tweakeys_byte ^ c0
    l_13 = l_5 ^ SBOX[plaintext[10]]
    l_1 = l_13 ^ SBOX[plaintext[13]]
    return [SBOX[l_1], SBOX[l_5],
            SBOX[l_13]]


def k2(tweakeys_byte, plaintext):
    l_6 = SBOX[plaintext[1]] ^ tweakeys_byte
    l_14 = l_6 ^ SBOX[plaintext[11]]
    l_2 = l_14 ^ SBOX[plaintext[14]]
    return [SBOX[l_2], SBOX[l_6],
            SBOX[l_14]]


def k3(tweakeys_byte, plaintext):
    c2 = 0x2
    l_7 = SBOX[plaintext[2]] ^ tweakeys_byte
    l_15 = l_7 ^ SBOX[plaintext[8]] ^ c2
    l_3 = l_15 ^ SBOX[plaintext[15]]
    return [SBOX[l_3], SBOX[l_7],
            SBOX[l_15]]


def k4(tweakeys_byte, plaintext):
    l_8 = SBOX[plaintext[3]] ^ tweakeys_byte
    l_16 = l_8 ^ SBOX[plaintext[9]]
    l_4 = l_16 ^ SBOX[plaintext[12]]
    return [SBOX[l_4], SBOX[l_8],
            SBOX[l_16]]


def k5(tweakeys_byte, plaintext):
    return [SBOX[SBOX[plaintext[4]] ^ SBOX[plaintext[11]] ^ tweakeys_byte]]


def k6(tweakeys_byte, plaintext):
    c2 = 0x2
    return [SBOX[SBOX[plaintext[5]] ^ SBOX[plaintext[8]] ^ tweakeys_byte ^ c2]]


def k7(tweakeys_byte, plaintext):
    return [SBOX[SBOX[plaintext[6]] ^ SBOX[plaintext[9]] ^ tweakeys_byte]]


def k8(tweakeys_byte, plaintext):
    return [SBOX[SBOX[plaintext[7]] ^ SBOX[plaintext[10]] ^ tweakeys_byte]]


def k9(tweakeys_byte, ciphertext):
    l_56 = SBOX_INV[ciphertext[6] ^
            ciphertext[10] ^ ciphertext[14] ^ tweakeys_byte]
    l_55 = SBOX_INV[SBOX_INV[ciphertext[0]
        ^ ciphertext[12]] ^ l_56]
    return [l_56, l_55]


def k10(tweakeys_byte, ciphertext):
    c2 = 0x2
    l_56 = SBOX_INV[ciphertext[7] ^
            ciphertext[11] ^ ciphertext[15] ^ tweakeys_byte]
    l_55 = SBOX_INV[SBOX_INV[ciphertext[1]
        ^ ciphertext[13]] ^ l_56 ^ c2]
    return [l_56, l_55]


def k11(tweakeys_byte, ciphertext):
    l_56 = SBOX_INV[ciphertext[7] ^ tweakeys_byte]
    l_55 = SBOX_INV[SBOX_INV[ciphertext[2]
        ^ ciphertext[14]] ^ l_56]
    return [l_56, l_55]


def k12(tweakeys_byte, ciphertext):
    l_56 = SBOX_INV[ciphertext[6] ^ tweakeys_byte]
    l_55 = SBOX_INV[SBOX_INV[ciphertext[1]
        ^ ciphertext[13]] ^ l_56]
    return [l_56, l_55]


def k13(tweakeys_byte, ciphertext):
    l_56 = SBOX_INV[ciphertext[4] ^
            ciphertext[8] ^ ciphertext[12] ^ tweakeys_byte]
    l_55 = SBOX_INV[SBOX_INV[ciphertext[2]
        ^ ciphertext[14]] ^ l_56]
    return [l_56, l_55]


def k14(tweakeys_byte, ciphertext):
    c0 = 0x0A
    l_56 = SBOX_INV[ciphertext[4] ^ tweakeys_byte ^ c0]
    l_55 = SBOX_INV[l_56 ^
            SBOX_INV[ciphertext[3] ^ ciphertext[15]]]
    return [l_56, l_55]


def k15(tweakeys_byte, ciphertext):
    l_56 = SBOX_INV[ciphertext[5] ^ tweakeys_byte]
    l_55 = SBOX_INV[SBOX_INV[ciphertext[0]
        ^ ciphertext[12]] ^ l_56]
    return [l_56, l_55]


def k16(tweakeys_byte, ciphertext):
    l_56 = SBOX_INV[ciphertext[5] ^
            ciphertext[9] ^ ciphertext[13] ^ tweakeys_byte]
    l_55 = SBOX_INV[SBOX_INV[ciphertext[3]
        ^ ciphertext[15]] ^ l_56]
    return [l_56, l_55]


intermediates = [k1, k2, k3, k4, k5, k6, k7,
        k8, k9, k10, k11, k12, k13, k14, k15, k16]


def k1_k10(k1_tk_byte, k10_tk_byte, plaintext):
    c0 = [1, 3]
    lk1 = SBOX[plaintext[0]] ^ k1_tk_byte ^ c0[0] ^ SBOX[plaintext[10]
                                                                     ] ^ SBOX[plaintext[13]]
    lk1 = SBOX[lk1]
    lk = lk1 ^ k10_tk_byte ^ c0[1]
    return SBOX[lk]


def k1_k13(k1_tk_byte, k13_tk_byte, ciphertext):
    lk = SBOX_INV[ciphertext[4] ^
                        ciphertext[8] ^ ciphertext[12] ^ k13_tk_byte]
    lk = SBOX_INV[ciphertext[5] ^ ciphertext[13]
                        ] ^ SBOX_INV[ciphertext[2] ^ ciphertext[14]] ^ lk ^ k1_tk_byte
    return SBOX_INV[lk]


def k2_k9(k2_tk_byte, k9_tk_byte, ciphertext):
    c1 = 2
    lk = SBOX_INV[ciphertext[6] ^
                        ciphertext[10] ^ ciphertext[14] ^ k9_tk_byte]
    lk = SBOX_INV[ciphertext[7] ^ ciphertext[15]
                        ] ^ SBOX_INV[ciphertext[0] ^ ciphertext[12]] ^ lk ^ k2_tk_byte ^ c1
    return SBOX_INV[lk]


def k2_k16(k2_tk_byte, k16_tk_byte, plaintext):
    lk = SBOX[plaintext[1]] ^ k2_tk_byte ^ SBOX[plaintext[11]
                                                            ] ^ SBOX[plaintext[14]]
    lk = SBOX[lk] ^ k16_tk_byte
    return SBOX[lk]


def k3_k9(k3_tk_byte, k9_tk_byte, plaintext):
    c2 = 0x2
    lk = SBOX[plaintext[2]] ^ k3_tk_byte ^ SBOX[plaintext[8]
                                                            ] ^ c2 ^ SBOX[plaintext[15]]
    lk = SBOX[lk] ^ k9_tk_byte
    return SBOX[lk]


def k3_k10(k3_tk_byte, k10_tk_byte, ciphertext):
    lk = SBOX_INV[ciphertext[7] ^
                        ciphertext[11] ^ ciphertext[15] ^ k10_tk_byte]
    lk = SBOX_INV[ciphertext[4] ^ ciphertext[12]
                        ] ^ SBOX_INV[ciphertext[1] ^ ciphertext[13]] ^ lk ^ k3_tk_byte
    return SBOX_INV[lk]


def k4_k14(k4_tk_byte, k14_tk_byte, plaintext):
    lk = SBOX[plaintext[3]] ^ k4_tk_byte ^ SBOX[plaintext[9]
                                                            ] ^ SBOX[plaintext[12]]
    lk = SBOX[lk] ^ k14_tk_byte
    return SBOX[lk]


def k4_k16(k4_tk_byte, k16_tk_byte, ciphertext):
    c0 = 5
    lk = SBOX_INV[ciphertext[5] ^
                        ciphertext[9] ^ ciphertext[13] ^ k16_tk_byte] ^ k4_tk_byte ^ c0
    return SBOX_INV[lk]


def k5_k13(k5_tk_byte, k13_tk_byte, ciphertext):
    lk = SBOX_INV[ciphertext[4] ^
                        ciphertext[8] ^ ciphertext[12] ^ k13_tk_byte] ^ k5_tk_byte
    return SBOX_INV[lk]


def k6_k9(k6_tk_byte, k9_tk_byte, ciphertext):
    lk = SBOX_INV[ciphertext[6] ^
                        ciphertext[10] ^ ciphertext[14] ^ k9_tk_byte] ^ k6_tk_byte
    return SBOX_INV[lk]


def k7_k16(k7_tk_byte, k16_tk_byte, ciphertext):
    c2 = 0x2
    lk = SBOX_INV[ciphertext[5] ^
                        ciphertext[9] ^ ciphertext[13] ^ k16_tk_byte]
    lk = SBOX_INV[ciphertext[6] ^ ciphertext[14] ^ c2
                        ] ^ SBOX_INV[ciphertext[3] ^ ciphertext[15]] ^ lk ^ k7_tk_byte
    return SBOX_INV[lk]


def k8_k10(k8_tk_byte, k10_tk_byte, ciphertext):
    lk = SBOX_INV[ciphertext[7] ^
                        ciphertext[11] ^ ciphertext[15] ^ k10_tk_byte] ^ k8_tk_byte
    return SBOX_INV[lk]

intermediates_duo = [k1_k10, k1_k13, k2_k9, k2_k16, k3_k9, k3_k10, k4_k14,
        k4_k16, k5_k13, k6_k9, k7_k16, k8_k10]

# TK Bytes


def tk_k1(round_tweakeys, k):
    return round_tweakeys[0][0][0] ^ round_tweakeys[1][0][0] ^ k


def tk_k2(round_tweakeys, k):
    return round_tweakeys[0][0][1] ^ round_tweakeys[1][0][1] ^ k


def tk_k3(round_tweakeys, k):
    return round_tweakeys[0][0][2] ^ round_tweakeys[1][0][2] ^ k


def tk_k4(round_tweakeys, k):
    return round_tweakeys[0][0][3] ^ round_tweakeys[1][0][3] ^ k


def tk_k5(round_tweakeys, k):
    return round_tweakeys[0][0][4] ^ round_tweakeys[1][0][4] ^ k


def tk_k6(round_tweakeys, k):
    return round_tweakeys[0][0][5] ^ round_tweakeys[1][0][5] ^ k


def tk_k7(round_tweakeys, k):
    return round_tweakeys[0][0][6] ^ round_tweakeys[1][0][6] ^ k


def tk_k8(round_tweakeys, k):
    return round_tweakeys[0][0][7] ^ round_tweakeys[1][0][7] ^ k


def tk_k9(round_tweakeys, k):
    return round_tweakeys[0][55][5] ^ round_tweakeys[1][55][5] ^ last_round_k[k]


def tk_k10(round_tweakeys, k):
    return round_tweakeys[0][55][6] ^ round_tweakeys[1][55][6] ^ last_round_k[k]


def tk_k11(round_tweakeys, k):
    return round_tweakeys[0][55][3] ^ round_tweakeys[1][55][3] ^ last_round_k[k]


def tk_k12(round_tweakeys, k):
    return round_tweakeys[0][55][2] ^ round_tweakeys[1][55][2] ^ last_round_k[k]


def tk_k13(round_tweakeys, k):
    return round_tweakeys[0][55][7] ^ round_tweakeys[1][55][7] ^ last_round_k[k]


def tk_k14(round_tweakeys, k):
    return round_tweakeys[0][55][0] ^ round_tweakeys[1][55][0] ^ last_round_k[k]


def tk_k15(round_tweakeys, k):
    return round_tweakeys[0][55][1] ^ round_tweakeys[1][55][1] ^ last_round_k[k]


def tk_k16(round_tweakeys, k):
    return round_tweakeys[0][55][4] ^ round_tweakeys[1][55][4] ^ last_round_k[k]


tweakeys = [tk_k1, tk_k2, tk_k3, tk_k4, tk_k5, tk_k6, tk_k7, tk_k8,
        tk_k9, tk_k10, tk_k11, tk_k12, tk_k13, tk_k14, tk_k15, tk_k16]

def tk_k1_k10(round_tweakeys,k_1,k_10):
    tk_1 = round_tweakeys[0][0][0] ^ round_tweakeys[1][0][0] ^ k_1
    tk_2 = round_tweakeys[0][1][0] ^ round_tweakeys[1][1][0] ^ third_round_k[k_10]
    return (tk_1,tk_2)

def tk_k1_k13(round_tweakeys,k_1,k_13):
    tk_1 = round_tweakeys[0][54][6] ^ round_tweakeys[1][54][6] ^ fifty_five_round_k[k_1]
    tk_2 = round_tweakeys[0][55][7] ^ round_tweakeys[1][55][7] ^ last_round_k[k_13]
    return (tk_1,tk_2)

def tk_k2_k9(round_tweakeys,k_2,k_9):
    tk_1 = round_tweakeys[0][54][4] ^ round_tweakeys[1][54][4] ^ fifty_five_round_k[k_2] 
    tk_2 = round_tweakeys[0][55][5] ^ round_tweakeys[1][55][5] ^ last_round_k[k_9]
    return (tk_1,tk_2)

def tk_k2_k16(round_tweakeys,k_2,k_16):
    tk_1 = round_tweakeys[0][0][1] ^ round_tweakeys[1][0][1] ^ k_2
    tk_2 = round_tweakeys[0][1][1] ^ round_tweakeys[1][1][1] ^ third_round_k[k_16]
    return (tk_1,tk_2)

def tk_k3_k9(round_tweakeys,k_3,k_9):
    tk_1 = round_tweakeys[0][0][2] ^ round_tweakeys[1][0][2] ^ k_3
    tk_2 = round_tweakeys[0][1][2] ^ round_tweakeys[1][1][2] ^ third_round_k[k_9]
    return (tk_1,tk_2)

def tk_k3_k10(round_tweakeys,k_3,k_10):
    tk_1 = round_tweakeys[0][54][5] ^ round_tweakeys[1][54][5] ^ fifty_five_round_k[k_3]
    tk_2 = round_tweakeys[0][55][6] ^ round_tweakeys[1][55][6] ^ last_round_k[k_10]
    return (tk_1,tk_2)

def tk_k4_k14(round_tweakeys,k_4,k_14):
    tk_1 = round_tweakeys[0][0][3] ^ round_tweakeys[1][0][3] ^ k_4 
    tk_2 = round_tweakeys[0][1][3] ^ round_tweakeys[1][1][3] ^ third_round_k[k_14]
    return (tk_1,tk_2)

def tk_k4_k16(round_tweakeys,k_4,k_16):
    tk_1 = round_tweakeys[0][54][0] ^ round_tweakeys[1][54][0] ^ fifty_five_round_k[k_4]
    tk_2 = round_tweakeys[0][55][4] ^ round_tweakeys[1][55][4] ^ last_round_k[k_16]
    return  (tk_1,tk_2)

def tk_k5_k13(round_tweakeys,k_5,k_13):
    tk_1 = round_tweakeys[0][54][3] ^ round_tweakeys[1][54][3] ^ fifty_five_round_k[k_5]
    tk_2 = round_tweakeys[0][55][7] ^ round_tweakeys[1][55][7] ^ last_round_k[k_13]
    return (tk_1,tk_2)

def tk_k6_k9(round_tweakeys,k_6,k_9):
    tk_1 = round_tweakeys[0][54][1] ^ round_tweakeys[1][54][1] ^ fifty_five_round_k[k_6]
    tk_2 = round_tweakeys[0][55][5] ^ round_tweakeys[1][55][5] ^ last_round_k[k_9]
    return (tk_1,tk_2)

def tk_k7_k16 (round_tweakeys,k_7,k_16):
    tk_1 = round_tweakeys[0][54][7] ^ round_tweakeys[1][54][7] ^ fifty_five_round_k[k_7]
    tk_2 = round_tweakeys[0][55][4] ^ round_tweakeys[1][55][4] ^ last_round_k[k_16]
    return (tk_1,tk_2)

def tk_k8_k10(round_tweakeys,k_8,k_10):
    tk_1 = round_tweakeys[0][54][2] ^ round_tweakeys[1][54][2] ^ fifty_five_round_k[k_8]
    tk_2 = round_tweakeys[0][55][6] ^ round_tweakeys[1][55][6] ^ last_round_k[k_10]
    return (tk_1,tk_2)



tweakeys_duo = [tk_k1_k10, tk_k1_k13, tk_k2_k9, tk_k2_k16, tk_k3_k9, tk_k3_k10, tk_k4_k14,
        tk_k4_k16, tk_k5_k13, tk_k6_k9, tk_k7_k16, tk_k8_k10]

correspondance_duo = [[0, 9], [0, 12], [1, 8], [1, 15], [2, 8], [2, 9], [3, 13], [3, 15], [4, 12], [5, 8], [6, 15], [7, 9]]

# CLASSES BP


class Edge():
    def __init__(self, var):
        self._previous_message = np.zeros(256, dtype=float)
        self._var = var

    def pass_message(self, marg_factor):
        message = np.subtract(marg_factor, self._previous_message)
        self._previous_message = marg_factor
        return (message, self._var)


class Clique():
    def __init__(self):
        self.factor = np.zeros((256, 256), dtype=float)

    # var = var that should be kept in the message => the opposite of the one being marginalized over
    def emit_message(self, var):
        return np.max(self.factor, axis=var)

    # var = 0 or var = 1, if message.shape() = (256,256) use var = 0
    def update_factor(self, message, var):
        if var == 0:
            self.factor = np.add(self.factor, message)
        else:
            self.factor = np.add(self.factor, message[:, np.newaxis])

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



# LOADING
# TODO: make that faster if I have the time, cause 1min to load 6gb of data is insane

def get_traces(path):
    """Reads and processes traces from a file."""
    # Get the keys from the filename
    tk1, tk2, key = [
            np.fromiter((int(i+j, 16)
                for i, j in zip(a[::2], a[1::2])), dtype=np.int32)
            for a in os.path.splitext(os.path.basename(path))[0].split("-")[:3]
            ]
    plaintexts = []
    ciphertexts = []
    values = []
    with open(path) as file:
        for row in csv.reader(file, delimiter=";"):
            plaintexts.append(np.fromiter(
                (int(i+j, 16) for i, j in zip(row[0][::2], row[0][1::2])), dtype=np.int32))
            ciphertexts.append(np.fromiter(
                (int(i+j, 16) for i, j in zip(row[1][::2], row[1][1::2])), dtype=np.int32))
            values.append(np.fromstring(row[2], sep=","))
    # Create a vector of plaintexts
    plaintexts = np.array(plaintexts)
    ciphertexts = np.array(ciphertexts)
    # and a 2d array of traces where each row is a trace and each column is a timestamp
    traces = np.array(values)
    return ((tk1, tk2), plaintexts, ciphertexts, traces, key)

# PROCESSING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut", action="store_true")
    parser.add_argument("--offset")
    args = parser.parse_args()

    # used to split multiple experiments over different concurrently running jobs
    if args.offset == None:
        OFFSET = 0
    else:
        OFFSET = int(args.offset)
    if args.lut:
        attack_path = "traces/ae7b6797744257ec0a0df83b40d34432-c1b8c62b67497500a2e878137498321e-2b7e151628aed2a6abf7158809cf4f3c-lut.csv"
        end_attack_path = "traces/ae7b6797744257ec0a0df83b40d34432-c1b8c62b67497500a2e878137498321e-2b7e151628aed2a6abf7158809cf4f3c-lut-end.csv"
    else:
        attack_path = "traces/f909e1bd7e94a20c005fb36214366750-25bab39010bfcdf5a9977f81e50bca34-2b7e151628aed2a6abf7158809cf4f3c.csv"
        end_attack_path = "traces/f909e1bd7e94a20c005fb36214366750-25bab39010bfcdf5a9977f81e50bca34-2b7e151628aed2a6abf7158809cf4f3c-end.csv"
    
    if args.lut:
        # round 1 S-Boxes
        front_windows_starts = np.array([450, 450, 450, 480, 500, 520, 520, 540,600,600,620,620,700,700,700,700,700])
        front_windows_ends = np.array([600, 700, 700, 720, 750, 800, 800, 820,800,850,850,880,900,950,950,950,980])
        front_windows = [[0, 4, 12], [1, 5, 13], [2, 6, 14],
                [3, 7, 15], [9], [10], [11], [8]]
        front_windows_starts_duo = np.array([822,842,862,882])
        front_windows_ends_duo = np.array([1272,1292,1312,1332])

    else:
        # round 1 S-Boxes
        front_windows_starts = np.array([900, 1050, 1200, 1400])
        front_windows_ends = np.array([1200, 1400, 1600, 1900])
        front_windows = [[0, 2, 3], [0, 2, 3], [0, 2, 3],
                [0, 2, 3], [2], [2], [2], [2]]
        front_windows_starts_duo = np.array([1700,1862,1814,1870])
        front_windows_ends_duo = np.array([2280,2312,2264,2320])

    
    if args.lut:
        # round 56-55 S-Boxes
        end_windows_starts = np.array([3170, 3190, 3210, 3230, 3250, 3270, 3290, 3310, 2926, 2946, 2966, 2986, 3006, 3026, 3046, 3066])
        end_windows_ends = np.array([3570, 3590, 3610, 3630, 3650, 3670, 3690, 3710, 3326, 3346, 3366, 3386, 3406, 3426, 3446, 3466])
        end_windows = [[5, 11], [6, 8], [3, 12],
                [2, 15], [7, 9], [0, 13], [1, 14], [4, 10]]
        end_windows_starts_duo = np.array([2886,2846,2866,2766,2826,2786,2906,2806])
        end_windows_ends_duo = np.array([3336,3296,3316,3216,3276,3236,3356,3256])


    else:
        # round 56-55 S-Boxes
        end_windows_starts = np.array([3240,3220,3248,3272,3168,3474,3382,3402,2822,2934,2892,2912,3056,3048,3076,3100])
        end_windows_ends = np.array([3890,3870,3898,3922,4068,4124,4032,4052,3272,3384,3342,3362,3506,3498,3526,3556])
        end_windows = [[5, 11], [6, 8], [3, 12],
                [2, 15], [7, 9], [0, 13], [1, 14], [4, 10]]
        end_windows_starts_duo = np.array([2658,2400,2700,2516,2500,2446,2682,2474])
        end_windows_ends_duo = np.array([3108,3150,3100,2916,2950,2900,3132,2924])

    (tk1, tk2), _, ciphertexts, end_traces, TRUE_KEY = get_traces(end_attack_path)
    _, plaintexts, _, front_traces, _ = get_traces(attack_path)
    rtk = compute_tk1_tk2(tk1, tk2)
    tk_bytes =  collections.defaultdict(list)
    for key_id in range(16):
        tk_bytes[key_id] =[tweakeys[key_id](rtk, k) for k in range(256)]
    for key_id in range(16,28):
        tk_bytes[key_id] = [tweakeys_duo[key_id - 16](rtk, k>>8, k & 0xff)
                for k in range(65536)]
    
    # Could go into the argparse but why bother :)
    if args.lut:
        NUMBER_OF_TRACES = 50
        NUMBER_OF_EXP = 10
    else:
        NUMBER_OF_TRACES = 100
        NUMBER_OF_EXP = 1
    
    #cutting down on the memory no need to keep everything
    #we could also not LOAD everything in the first place but again no time to fix it
    scoped_front_traces = front_traces[OFFSET*NUMBER_OF_TRACES:(NUMBER_OF_EXP + OFFSET)*NUMBER_OF_TRACES]
    scoped_end_traces = end_traces[OFFSET*NUMBER_OF_TRACES:(NUMBER_OF_EXP + OFFSET)*NUMBER_OF_TRACES]
    scoped_plaintexts = plaintexts[OFFSET*NUMBER_OF_TRACES:(NUMBER_OF_EXP + OFFSET)*NUMBER_OF_TRACES]
    scoped_ciphertexts = ciphertexts[OFFSET*NUMBER_OF_TRACES:(NUMBER_OF_EXP + OFFSET)*NUMBER_OF_TRACES]
    del front_traces, end_traces, plaintexts, ciphertexts
    
    # START
    # The threadpool_limits was added due to numpy defaulting
    # to occupy as much core as possible and this was running on a shared server

    # Considering that those experiments are trivially parallel it is also more effective
    # to dedicate a single core per execution and run multiple scripts in parallel
    # than parallelizing the CPA computation.
    with threadpool_limits(limits=1, user_api="blas"):
        with threadpool_limits(limits=1, user_api="openmp"):
            for experiment in range(NUMBER_OF_EXP):
                ranks =  collections.defaultdict(list)
                ranks_mono =  collections.defaultdict(list)
                ranks_single = collections.defaultdict(list)
                scores = collections.defaultdict(list)
                scores_single =  collections.defaultdict(list)
                for key_id in range(16):
                    scores[key_id] = [0 for _ in range(256)]
                for key_id in range(16,28):
                    scores[key_id] = [0 for _ in range(65536)]
                exp_range_down = experiment * NUMBER_OF_TRACES
                exp_range_up = exp_range_down + NUMBER_OF_TRACES
                exp_front_traces = scoped_front_traces[exp_range_down:exp_range_up]
                exp_end_traces = scoped_end_traces[exp_range_down:exp_range_up]
                exp_plaintexts = scoped_plaintexts[exp_range_down:exp_range_up]
                exp_ciphertexts =  scoped_ciphertexts[exp_range_down:exp_range_up]
                full_predicted_values_mono = np.empty((256,3,16,NUMBER_OF_TRACES),dtype=int)
                full_predicted_values_duo = np.empty((65536,12,NUMBER_OF_TRACES),dtype=int)
                for trace_id in range(NUMBER_OF_TRACES):
                    factors = []
                    for key_id in range(8): 
                        predicted_values = np.array([intermediates[key_id](tk_byte, exp_plaintexts[trace_id]) for tk_byte in tk_bytes[key_id]])
                        full_predicted_values_mono[:,:,key_id,trace_id] = predicted_values
                        scores[key_id] = [0 for _ in range(256)]
                        scores_single[key_id] = [0 for _ in range(256)]
                        for j,_ in enumerate(front_windows[key_id]):
                            traces = exp_front_traces[:trace_id+1,
                                front_windows_starts[front_windows[key_id][j]]:
                                front_windows_ends[front_windows[key_id][j]]]
                            trace_div_arr = np.empty((traces.shape[1], traces.shape[0]))
                            for i in range(traces.shape[1]):
                                trace = traces[:, i]
                                trace_m = trace - trace.mean()
                                norm_trace_m = np.linalg.norm(trace_m)
                                if norm_trace_m == 0:
                                    trace_div_arr[i] = np.nan
                                    continue
                                trace_div_arr[i] = trace_m / norm_trace_m
                            for key in range(256):
                                hws = np.array([hw(x) for x in full_predicted_values_mono[key,j,key_id,:trace_id+1]])
                                hws_m = hws - hws.mean()
                                norm_hws_m = np.linalg.norm(hws_m)
                                if norm_hws_m == 0:
                                    continue
                                hws_div = hws_m / norm_hws_m
                                s = np.max(np.abs(np.dot(trace_div_arr, hws_div)))
                                scores[key_id][key] += s
                                if j == 0:
                                    scores_single[key_id][key] = s

                    for windows_id, key_id in enumerate([16,19,20,22]):
                        predicted_values = np.array([intermediates_duo[key_id-16](tk_byte[0],tk_byte[1],exp_plaintexts[trace_id]) for tk_byte in tk_bytes[key_id]])
                        full_predicted_values_duo[:,key_id-16,trace_id] = predicted_values
                        scores[key_id] = [0 for _ in range(65536)]
                        traces = exp_front_traces[:trace_id+1,
                            front_windows_starts_duo[windows_id]:
                            front_windows_ends_duo[windows_id]]
                        trace_div_arr = np.empty((traces.shape[1], traces.shape[0]))
                        for i in range(traces.shape[1]):
                            trace = traces[:, i]
                            trace_m = trace - trace.mean()
                            norm_trace_m = np.linalg.norm(trace_m)
                            if norm_trace_m == 0:
                                trace_div_arr[i] = np.nan
                                continue
                            trace_div_arr[i] = trace_m / norm_trace_m
                        for key in range(65536):
                            hws = np.array([hw(x) for x in full_predicted_values_duo[key,key_id-16,:trace_id+1]])
                            hws_m = hws - hws.mean()
                            norm_hws_m = np.linalg.norm(hws_m)
                            if norm_hws_m == 0:
                                continue
                            hws_div = hws_m / norm_hws_m
                            scores[key_id][key] += np.max(np.abs(np.dot(trace_div_arr, hws_div)))


                    for key_id in range(8,16):
                        predicted_values = np.array([intermediates[key_id](tk_byte, exp_ciphertexts[trace_id]) for tk_byte in tk_bytes[key_id]])
                        full_predicted_values_mono[:,:2,key_id,trace_id] = predicted_values
                        scores[key_id] = [0 for _ in range(256)]
                        scores_single[key_id] = [0 for _ in range(256)]
                        for j,_ in enumerate(end_windows[key_id-8]):
                            traces = exp_end_traces[:trace_id+1,
                                end_windows_starts[end_windows[key_id-8][j]]:
                                end_windows_ends[end_windows[key_id-8][j]]]
                            trace_div_arr = np.empty((traces.shape[1], traces.shape[0]))
                            for i in range(traces.shape[1]):
                                trace = traces[:, i]
                                trace_m = trace - trace.mean()
                                norm_trace_m = np.linalg.norm(trace_m)
                                if norm_trace_m == 0:
                                    trace_div_arr[i] = np.nan
                                    continue
                                trace_div_arr[i] = trace_m / norm_trace_m
                            for key in range(256):
                                hws = np.array([hw(x) for x in full_predicted_values_mono[key,j,key_id,:trace_id+1]])
                                hws_m = hws - hws.mean()
                                norm_hws_m = np.linalg.norm(hws_m)
                                if norm_hws_m == 0:
                                    continue
                                hws_div = hws_m / norm_hws_m
                                s = np.max(np.abs(np.dot(trace_div_arr, hws_div)))
                                scores[key_id][key] += s
                                if j == 0:
                                    scores_single[key_id][key] = s

                    for windows_id, key_id in enumerate([17,18,21,23,24,25,26,27]):
                        predicted_values = np.array([intermediates_duo[key_id-16](tk_byte[0],tk_byte[1],exp_ciphertexts[trace_id]) for tk_byte in tk_bytes[key_id]])
                        full_predicted_values_duo[:,key_id-16,trace_id] = predicted_values
                        scores[key_id] = [0 for _ in range(65536)]
                        traces = exp_end_traces[:trace_id+1,
                            end_windows_starts_duo[windows_id]:
                            end_windows_ends_duo[windows_id]]
                        trace_div_arr = np.empty((traces.shape[1], traces.shape[0]))
                        for i in range(traces.shape[1]):
                            trace = traces[:, i]
                            trace_m = trace - trace.mean()
                            norm_trace_m = np.linalg.norm(trace_m)
                            if norm_trace_m == 0:
                                trace_div_arr[i] = np.nan
                                continue
                            trace_div_arr[i] = trace_m / norm_trace_m
                        for key in range(65536):
                            hws = np.array([hw(x) for x in full_predicted_values_duo[key,key_id-16,:trace_id+1]])
                            hws_m = hws - hws.mean()
                            norm_hws_m = np.linalg.norm(hws_m)
                            if norm_hws_m == 0:
                                continue
                            hws_div = hws_m / norm_hws_m
                            scores[key_id][key] += np.max(np.abs(np.dot(trace_div_arr, hws_div)))

                    for key_id in range(16):
                        s = np.array(scores_single[key_id])
                        true = s[TRUE_KEY[key_id]]
                        s = np.sort(s)[::-1]
                        if np.where(s == true)[0].size > 0:
                            ranks_single[key_id].append(np.where(s == true)[0][0])
                        else:
                            ranks_single[key_id].append(255)

                    #k11, k12, k15 are not in the graph
                    for key_id in [0,1,2,3,4,5,6,7,8,9,12,13,15]:
                        s = np.array(scores[key_id])
                        true = s[TRUE_KEY[key_id]]
                        ss = np.sort(s)[::-1]
                        if np.where(ss == true)[0].size > 0:
                            ranks_mono[key_id].append(np.where(ss == true)[0][0])
                        else:
                            ranks_mono[key_id].append(255)
                        factors.append(s)

                    for key_id in [10,11,14]:
                        s = np.array(scores[key_id])
                        true = s[TRUE_KEY[key_id]]
                        s = np.sort(s)[::-1]
                        if np.where(s == true)[0].size > 0:
                            ranks_mono[key_id].append(np.where(s == true)[0][0])
                            ranks[key_id].append(np.where(s == true)[0][0])
                        else:
                            ranks_mono[key_id].append(255)
                            ranks[key_id].append(255)

                    for key_id in range(16,28):
                        s = np.array(scores[key_id])
                        factors.append(s.reshape(256,256))
                    marginals = combine_factors(factors)

                    for marg_id, key_id in enumerate([0,1,2,3,4,5,6,7,8,9,12,13,15]):
                        true = marginals[marg_id][TRUE_KEY[key_id]]
                        s = np.sort(marginals[marg_id])[::-1]
                        if np.where(s == true)[0].size > 0:
                            ranks[key_id].append(np.where(s == true)[0][0])
                        else:
                            ranks[key_id].append(255)

                if args.lut:
                    name_single = "16sbox_lut_cpa"
                    name_mono = "32sbox_lut_cpa"
                    name_duo = "44sbox_lut_cpa"
                else:
                    name_single = "16sbox_circuit_cpa"
                    name_mono = "32sbox_circuit_cpa"
                    name_duo = "44sbox_circuit_cpa"

                # output results as .csv with rows key_id;rank_at_trace_1;rank_at_trace_2;...;rank_at_max_trace;
                with open(
                    f"{name_single}.csv", "a", newline=""
                ) as f:
                    writer = csv.writer(f, delimiter=";")
                    for key, v in ranks_single.items():
                        writer.writerow([key] + v)

                with open(
                    f"{name_mono}.csv", "a", newline=""
                ) as f:
                    writer = csv.writer(f, delimiter=";")
                    for key, v in ranks_mono.items():
                        writer.writerow([key] + v)

                with open(
                    f"{name_duo}.csv", "a", newline=""
                ) as f:
                    writer = csv.writer(f, delimiter=";")
                    for key, v in ranks.items():
                        writer.writerow([key] + v)
if __name__ == "__main__":
    main()
