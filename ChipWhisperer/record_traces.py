import chipwhisperer as cw
from chipwhisperer.common.utils.util import hexStrToByteArray
import csv
import os, binascii
from binascii import hexlify


# programming the Chipwhisperer
PLATFORM = "CWLITEARM"
SCOPETYPE = "OPENADC"
scope = cw.scope()
target = cw.target(scope, cw.targets.SimpleSerial)
scope.default_setup()
if "STM" in PLATFORM or PLATFORM == "CWLITEARM" or PLATFORM == "CWNANO":
    prog = cw.programmers.STM32FProgrammer
elif PLATFORM == "CW303" or PLATFORM == "CWLITEXMEGA":
    prog = cw.programmers.XMEGAProgrammer
else:
    prog = None

# for this path to work either place this script in the relevant folder of ChipWhisperer project
# or change the path

fw_path = '../hardware/victims/firmware/simpleserial-SKINNY/simpleserial-skinny-{}.hex'.format(PLATFORM)
cw.program_target(scope, prog, fw_path)


num_traces = 500
keys = []
plaintexts = []
traces = []

ktp = cw.ktp.Basic()

k1 = hexStrToByteArray("4e4508e137815ef1bcbfc22ec93dbd55")
k2 = hexStrToByteArray("b40df81a11aaf1490b5834b2a1d6866a")
target.simpleserial_write('1', k1)
target.simpleserial_wait_ack()
target.simpleserial_write('2', k2)
target.simpleserial_wait_ack()

# recording the end of the trace
# comment out to record the beginning
target.simpleserial_write('e', b'')
target.simpleserial_wait_ack()

for i in range(num_traces):
    target.simpleserial_write('1', k1)
    target.simpleserial_wait_ack()
    target.simpleserial_write('2', k2)
    target.simpleserial_wait_ack()

    # generate a random key from urandom
    # writting this key in TK3 forces a recomputation of the
    # round tweakeys
    key = binascii.hexlify(os.urandom(16)).decode("utf-8")
    keys.append(key)
    target.simpleserial_write('k', hexStrToByteArray(key))
    target.simpleserial_wait_ack()

    # generate a random plaintext from urandom
    # and record the trace of its encryption
    plaintext = binascii.hexlify(os.urandom(16)).decode("utf-8")
    plaintexts.append(plaintext)
    trace = cw.capture_trace(scope, target, hexStrToByteArray(plaintext))
    if trace is None:
        continue
    traces.append(trace)

    # simple display to follow the recording
    if i % 100 == 0 and i > 0:
        print(i)

scope.dis()
target.dis()

strk1 = hexlify(k1).decode("utf-8")
strk2 = hexlify(k2).decode("utf-8")

#replace with your output path of choice
OUTPUT_PATH = f"{strk1}-{strk2}-end.csv"

# outputs a csv with rows shaped
# key;plaintext;ciphertext;timestamp_1,timestamp_2,...,timestamp_5000

f = open(OUTPUT_PATH, 'a')
with f:
    fields = ['key','plaintext', 'ciphertext', 'trace']
    writer = csv.DictWriter(f, fieldnames=fields,delimiter=';')
    for i,trace in enumerate(traces):
        textpt = hexlify(trace.textin)
        textpt = textpt.decode("utf-8")
        textct = hexlify(trace.textout)
        textct = textct.decode("utf-8")
        writer.writerow({"key":keys[i],"plaintext":textpt,"ciphertext":textct,"trace":','.join([str(_) for _ in trace.wave])})
