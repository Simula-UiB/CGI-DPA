import chipwhisperer as cw
from chipwhisperer.common.utils.util import hexStrToByteArray
import csv
import os
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

fw_path = '../hardware/victims/firmware/simpleserial-SKINNY/simpleserial-skinny-{}.hex'.format(PLATFORM)
cw.program_target(scope, prog, fw_path)


# read from the existing csv for the value of key and plaintext
# using the filename for tk1 and tk2

INPUT_PATH = os.path.expanduser('~')+"/simula/VIOLETFELONY/sca/traces/4e4508e137815ef1bcbfc22ec93dbd55-b40df81a11aaf1490b5834b2a1d6866a.csv"
keys = []
plaintexts = []
with open(INPUT_PATH) as file:
    for row in csv.reader(file,delimiter=";"):
        keys.append(row[0])
        plaintexts.append(row[1])
traces = []
num_traces = 50000

ktp = cw.ktp.Basic()

k1 = hexStrToByteArray("4e4508e137815ef1bcbfc22ec93dbd55")
k2 = hexStrToByteArray("b40df81a11aaf1490b5834b2a1d6866a")
target.simpleserial_write('1', k1)
target.simpleserial_wait_ack()
target.simpleserial_write('2', k2)
target.simpleserial_wait_ack()

# recording the end of the trace
target.simpleserial_write('e', b'')
target.simpleserial_wait_ack()

for i in range(num_traces):
    target.simpleserial_write('1', k1)
    target.simpleserial_wait_ack()
    target.simpleserial_write('2', k2)
    target.simpleserial_wait_ack()
    target.simpleserial_write('k', hexStrToByteArray(keys[i]))
    target.simpleserial_wait_ack()
    trace = cw.capture_trace(scope, target, hexStrToByteArray(plaintexts[i]))
    if trace is None:
        continue
    traces.append(trace)
    if i % 1000 == 0 and i > 0:
        print(i)

scope.dis()
target.dis()

textk1 = hexlify(k1)
textk1 = textk1.decode("utf-8")
textk2 = hexlify(k2)
textk2 = textk2.decode("utf-8")

OUTPUT_PATH = os.path.expanduser('~') + f"/simula/VIOLETFELONY/sca/traces/{textk1}-{textk2}-end.csv"

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
