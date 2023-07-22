import numpy as np
import argparse
import csv
from skinny import SkinnyCipher
import os
from scipy.stats import multivariate_normal
import collections
SBOX = SkinnyCipher.sbox8
SBOX_INV = SkinnyCipher.sbox8_inv

# SKINNY FUNCTIONS


def skinny_lfsr_3(byte, n_application):
    for _ in range(n_application):
        x6 = (byte >> 6) & 0x01
        x0 = byte & 0x01
        top = (x0 ^ x6) << 7
        byte = (byte >> 1) ^ top
    return byte


# precomputation to avoid running the lfsr when computing tk_k9-16
last_round_k = np.arange(start=0, stop=256, step=1, dtype=int)
last_round_k = [skinny_lfsr_3(k, 28) for k in last_round_k]


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

# LOADING
# TODO: make that faster if I have the time, cause 1min to load 6gb of data is insane

def get_profile_traces(path):
    """Reads and processes traces from a file."""
    # Get the keys from the filename
    tk1, tk2 = [
            np.fromiter((int(i+j, 16)
                for i, j in zip(a[::2], a[1::2])), dtype=np.int32)
            for a in os.path.splitext(os.path.basename(path))[0].split("-")[:2]
            ]
    keys = []
    plaintexts = []
    ciphertexts = []
    values = []
    i = 0
    with open(path) as file:
        for row in csv.reader(file, delimiter=";"):
            keys.append(np.fromiter((int(i+j, 16)
                for i, j in zip(row[0][::2], row[0][1::2])), dtype=np.int32))
            plaintexts.append(np.fromiter(
                (int(i+j, 16) for i, j in zip(row[1][::2], row[1][1::2])), dtype=np.int32))
            ciphertexts.append(np.fromiter(
                (int(i+j, 16) for i, j in zip(row[2][::2], row[2][1::2])), dtype=np.int32))
            values.append(np.fromstring(row[3], sep=","))
            i += 1
    # Create a vector of keys and plaintexts
    keys = np.array(keys)
    plaintexts = np.array(plaintexts)
    ciphertexts = np.array(ciphertexts)
    # and a 2d array of traces where each row is a trace and each column is a timestamp
    traces = np.array(values)
    return ((tk1, tk2), plaintexts, ciphertexts, traces, keys)


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


# build n*256 multivariate_gaussian for n intermediate values dependending on a key_id
# windows_starts and windows_ends indicates the part of the traces relevant to each intermediate value
# inputs should be either plaintexts or ciphertexts matching the profile traces
# tk_bytes is tk1 ^ tk2 ^ tk3 matching the intermediates values we are targetting

def make_templates(windows_starts, windows_ends, profile_traces, key_id, inputs,tk_bytes,single=False):
    profile_traces = [np.array(
        profile_traces[:, windows_starts[x]:windows_ends[x]]) for x in range(len(windows_starts))]
    if single:
        intermediate_values = np.array([[intermediates[key_id](tk_bytes[i], inputs[i])[0]] for i in range(profile_traces[0].shape[0])],dtype=int)
    else:
        intermediate_values = np.array([intermediates[key_id](tk_bytes[i], inputs[i]) for i in range(profile_traces[0].shape[0])],dtype=int)
    traces = [[[] for _ in range(256)] for _ in range(len(windows_starts))]
    for i,iv in enumerate(intermediate_values):
        for j, v in enumerate(iv):
            traces[j][v].append(
                profile_traces[j][i]
            )
    traces = [[np.array(i) for i in traces[j]]
              for j in range(len(windows_starts))]
   
    meanMatrices = []
    covMatrices = []
    fullPOIs = []
    for variable_idx, t in enumerate(traces):
        tempMeans = np.zeros((256, profile_traces[variable_idx].shape[1]))
        for i in range(256):
            tempMeans[i] = np.average(t[i], 0)

        tempSumDiff = np.zeros(profile_traces[variable_idx].shape[1])
        for i in range(256):
            for j in range(i):
                tempSumDiff += np.abs(tempMeans[i] - tempMeans[j])
        numPOIs = 4
        POIs = []
        for _ in range(numPOIs):
            nextPOI = tempSumDiff.argmax()
            POIs.append(nextPOI)
            tempSumDiff[nextPOI] = 0
        fullPOIs.append(POIs)

        meanMatrix = np.zeros((256, numPOIs))
        for value in range(256):
            for i in range(numPOIs):
                meanMatrix[value][i] = tempMeans[value][POIs[i]]
        meanMatrices.append(meanMatrix)

        covMatrix = np.zeros((256, numPOIs, numPOIs))
        for value in range(256):
            covMatrix[value] = np.cov(t[value][:, POIs], rowvar=False)
        covMatrices.append(covMatrix)
        
    templates = [[multivariate_normal(meanMatrices[numIV][iv],covMatrices[numIV][iv]) for iv in range(256)] for numIV in range(len(windows_starts))]
    return templates,fullPOIs

# computes log-MLE on n_traces for every value of k (256 since this is mono there are no 2^16 key pair)
# add to the data dictionnary the rank of the correct key after each trace

def experiment(data, n_traces, traces, inputs, key_id, tk_bytes, key, windows_starts, windows_ends, POIs, templates, single=False):
    P_k = np.zeros(256)
    for i in range(n_traces):
        for k in range(256):
            if single:
                predicted_values = [intermediates[key_id](tk_bytes[k],inputs[i])[0]]
            else:
                predicted_values = intermediates[key_id](tk_bytes[k], inputs[i])
            s = 1
            for j, v in enumerate(predicted_values):
                tmp_trace = traces[i, windows_starts[j]:windows_ends[j]]
                tmp_trace = tmp_trace[POIs[j]]
                s *= templates[j][v].pdf(tmp_trace)
            P_k[k] += -np.log(s)
        data[(key_id, i)].append(
            np.where(P_k.argsort() == key[key_id])[0][0]
        )
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut", action="store_true")
    parser.add_argument("--single",action="store_true")
    
    #outdated argument used to skip the first round sboxes
    parser.add_argument("--skip",action="store_true")

    args = parser.parse_args()
    if args.lut:
        profile_path = "traces/2718eccbf079ded29eb5835f55db27ae-aab1146232391beb7df7aa359c2e26cd-lut.csv"
        attack_path = "traces/ae7b6797744257ec0a0df83b40d34432-c1b8c62b67497500a2e878137498321e-2b7e151628aed2a6abf7158809cf4f3c-lut.csv"
        end_profile_path = "traces/2718eccbf079ded29eb5835f55db27ae-aab1146232391beb7df7aa359c2e26cd-lut-end.csv"
        end_attack_path = "traces/ae7b6797744257ec0a0df83b40d34432-c1b8c62b67497500a2e878137498321e-2b7e151628aed2a6abf7158809cf4f3c-lut-end.csv"
    else:
        profile_path = "traces/4e4508e137815ef1bcbfc22ec93dbd55-b40df81a11aaf1490b5834b2a1d6866a.csv"
        attack_path = "traces/f909e1bd7e94a20c005fb36214366750-25bab39010bfcdf5a9977f81e50bca34-2b7e151628aed2a6abf7158809cf4f3c.csv"
        end_profile_path = "traces/4e4508e137815ef1bcbfc22ec93dbd55-b40df81a11aaf1490b5834b2a1d6866a-end.csv"
        end_attack_path = "traces/f909e1bd7e94a20c005fb36214366750-25bab39010bfcdf5a9977f81e50bca34-2b7e151628aed2a6abf7158809cf4f3c-end.csv"
    

    if not args.skip:
        # First 8 KeyBytes from the first round
        (
                (profile_tk1, profile_tk2),
                profile_plaintexts,
                _,
                profile_traces,
                profile_keys,
                ) = get_profile_traces(profile_path)
        (tk1, tk2), plaintexts, _, traces, key = get_traces(attack_path)
        rtk = compute_tk1_tk2(tk1, tk2)
        if args.lut:
            # round 1 S-Boxes
            windows_starts = np.array([450, 450, 450, 480, 500, 520, 520, 540,600,600,620,620,700,700,700,700,700])
            windows_ends = np.array([600, 700, 700, 720, 750, 800, 800, 820,800,850,850,880,900,950,950,950,980])
            if args.single:
                windows = [[0], [1], [2],
                    [3], [9], [10], [11], [8]]
     
            else:
                windows = [[0, 4, 12], [1, 5, 13], [2, 6, 14],
                    [3, 7, 15], [9], [10], [11], [8]]
        else:
            # round 1 S-Boxes
            windows_starts = np.array([900, 1050, 1200, 1400])
            windows_ends = np.array([1200, 1400, 1600, 1900])
            if args.single:
                windows = [[0], [0], [0],
                    [0], [2], [2], [2], [2]] 
            else:
                windows = [[0, 2, 3], [0, 2, 3], [0, 2, 3],
                    [0, 2, 3], [2], [2], [2], [2]]
        
        for key_id in range(8):
            round_tweakeys = compute_tk1_tk2(profile_tk1,profile_tk2)
            profile_tks = np.array([tweakeys[key_id](round_tweakeys, profile_keys[i][key_id]) for i in range(len(profile_keys))])
     
            templates,POIs = make_templates(windows_starts[windows[key_id]], windows_ends[windows[key_id]], profile_traces, key_id, profile_plaintexts, profile_tks,args.single)
            # attack
            data = collections.defaultdict(list)
            n_traces = 100
            tks = [tweakeys[key_id](rtk, k) for k in range(256)]
            for i in range(int(len(traces)/n_traces)):
                experiment(data, n_traces, traces[i*n_traces:(i+1)*n_traces], plaintexts[i*n_traces:(i+1)*n_traces],
                           key_id, tks, key, windows_starts[windows[key_id]], windows_ends[windows[key_id]], POIs, templates,args.single)
            if args.lut:
                if args.single:
                    name = "16sbox_lut"
                else:
                    name = "32sbox_lut"
            else:
                if args.single:
                    name = "16sbox_circuit"
                else:
                    name = "32sbox_circuit"
            
            with open(
                f"results/{name}.csv", "a", newline=""
            ) as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["Window", windows_starts, windows_ends])
                writer.writerow(["KeyByte", "Trace Number", "Keyranks"])
                for (key_index, trace_number), v in data.items():
                    writer.writerow([key_index, trace_number] + v)
        
        # Just to avoid a memory spike
        del profile_traces
        del traces
        
    # Last 8 KeyBytes from the last 2 rounds       
    (
            (profile_tk1, profile_tk2),
            _,
            profile_ciphertexts,
            profile_traces,
            profile_keys,
            ) = get_profile_traces(end_profile_path)
    (tk1, tk2), _, ciphertexts, traces, key = get_traces(end_attack_path)
    rtk = compute_tk1_tk2(tk1, tk2)
    if args.lut:
        # round 56-55 S-Boxes
        windows_starts = np.array([3170, 3190, 3210, 3230, 3250, 3270, 3290, 3310, 2926, 2946, 2966, 2986, 3006, 3026, 3046, 3066])
        windows_ends = np.array([3570, 3590, 3610, 3630, 3650, 3670, 3690, 3710, 3326, 3346, 3366, 3386, 3406, 3426, 3446, 3466])
        if args.single:
            windows = [[5], [6], [3],
                [2], [7], [0], [1], [4]]
 
        else:
            windows = [[5, 11], [6, 8], [3, 12],
                [2, 15], [7, 9], [0, 13], [1, 14], [4, 10]]
    else:
        # round 56-55 S-Boxes
        windows_starts = np.array([3240,3220,3248,3272,3168,3474,3382,3402,2822,2934,2892,2912,3056,3048,3076,3100])
        windows_ends = np.array([3890,3870,3898,3922,4068,4124,4032,4052,3272,3384,3342,3362,3506,3498,3526,3556])
        if args.single:
            windows = [[5], [6], [3],
                [2], [7], [0], [1], [4]]
 
        else:
            windows = [[5, 11], [6, 8], [3, 12],
                [2, 15], [7, 9], [0, 13], [1, 14], [4, 10]]
  
    for windows_key_id in range(8):
        key_id = windows_key_id + 8
        round_tweakeys = compute_tk1_tk2(profile_tk1,profile_tk2)
        profile_tks = np.array([tweakeys[key_id](round_tweakeys, profile_keys[i][key_id]) for i in range(len(profile_keys))])
 
        templates,POIs = make_templates(windows_starts[windows[windows_key_id]], windows_ends[windows[windows_key_id]], profile_traces, key_id, profile_ciphertexts, profile_tks, args.single)
        # attack
        data = collections.defaultdict(list)
        n_traces = 100
        tks = [tweakeys[key_id](rtk, k) for k in range(256)]
        for i in range(int(len(traces)/n_traces)):
            experiment(data, n_traces, traces[i*n_traces:(i+1)*n_traces], ciphertexts[i*n_traces:(i+1)*n_traces],
                       key_id, tks, key, windows_starts[windows[windows_key_id]], windows_ends[windows[windows_key_id]], POIs, templates,args.single)
        if args.lut:
            if args.single:
                name = "16sbox_lut"
            else:
                name = "32sbox_lut"
        else:
            if args.single:
                name = "16sbox_circuit"
            else:
                name = "32sbox_circuit"
        
        with open(
            f"results/{name}.csv", "a", newline=""
        ) as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Window", windows_starts, windows_ends])
            writer.writerow(["KeyByte", "Trace Number", "Keyranks"])
            for (key_index, trace_number), v in data.items():
                writer.writerow([key_index, trace_number] + v)
 
if __name__ == "__main__":
    main()
