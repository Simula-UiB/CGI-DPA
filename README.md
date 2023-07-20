<h1 align="center">Pincering SKINNY by Exploiting Slow Diffusion: Enhancing Differential Power Analysis with Cluster Graph Inference</h1>

<p align="center">
    <a href="https://github.com/Simula-UiB/CGI-DPA/blob/master/AUTHORS"><img src="https://img.shields.io/badge/authors-SimulaUIB-orange.svg"></a>
    <a href="https://github.com/Simula-UiB/CGI-DPA/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

This repository hosts the codebase that was used for the paper `Pincering SKINNY by Exploiting Slow Diffusion: Enhancing Differential Power Analysis with Cluster Graph Inference` scheduled to appear in [TCHES](https://tches.iacr.org/) 2023,
issue 4. It contains implementations of SKINNY integrated with the chipwhisperer code (with an example on how to use
them to collect traces) and the implementation of the CGI-DPA attack on real traces and Hamming Weight model. Furthermore,
the dataset of traces we used for the paper will be available in the artifact of the paper (currently under submission).

**WARNING:** This repository was developed in an academic context and no part of this code should be used in any production system. In particular the implementations of cryptosystems in this tool are not safe for any real world usage.

**Acknowledgement** Our implementations of SKINNY leaned heavily on the [FELICS](https://csrc.nist.gov/csrc/media/events/lightweight-cryptography-workshop-2015/documents/papers/session7-dinu-paper.pdf) project (our LUT implementation is theirs integrated with the Chipwhisperer code) and on the [Skinny-C](https://github.com/rweather/skinny-c) implementation by Rhys Weatherley (for the S-Box circuit).

## License

This repositery is licensed under multiple licenses.

The code derived from the [ChipWhisperer](https://github.com/newaetech/chipwhisperer) project is licensed under GPLv3, as denoted in their files.
The rest of the code is licensed under the MIT license.

* GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.html)
* MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT)


## Overview

### ChipWhisperer

Under the folder [chipwhisperer](chipwhisperer/) are two implementations of SKINNY-128-384 (with a 384-bit tweakey state and 56 rounds) integrated with the chipwhisperer code. One implementation (LUT) uses lookup tables for the S-Box computation while the other uses a circuit. For more detail on each implementation we refer to Section 5 of our paper.

This folder is not self-sufficient, and each implementation needs to be integrated in the chipwhisperer project. Specifically, the folders [simpleserial-SKINNY](chipwhisperer/simpleserial-SKINNY) and [simpleserial-SKINNY-LUT]((chipwhisperer/simpleserial-SKINNY-LUT)) should be placed in the folder [chipwhisperer/hardware/victims/firmware/](https://github.com/newaetech/chipwhisperer/tree/develop/hardware/victims/firmware) and used similarly to the other examples (such as [simpleserial-AES](https://github.com/newaetech/chipwhisperer/tree/develop/hardware/victims/firmware/simpleserial-aes) for which the chipwhisperer project provides [example jupyter notebooks](https://github.com/newaetech/chipwhisperer-jupyter)). 

Compared to this example, our SKINNY implementation has several extra flags that must be used to record traces correctly. Hence we give a self-contained [Python script](chipwhisperer/record_traces.py) that showcases how to flash the chipwhisperer with the firmware (that can be compiled using the [provided bash script](chipwhisperer/compile_firmware.sh)), set the tweakey state and record traces.
Technical details:
 * The implementation precomputes the round-tweakeys (RTK) upon receiving TK3. Our traces assumed fixed public TK1 and TK2, and TK3 as the key.
 * We need to record the first and last few rounds for our CGI-DPA attacks. However, the chipwhisperer only records 5000 timestamps which is insufficient to cover the full encryption. We, therefore, made a flag `e` that we can set to record the last rounds. When this flag is set, the implementation executes the first 47 rounds of encryption before setting the trigger to record the trace.
 * We discovered that if you were to record profiling traces with a rotating key (where we recompute the RTK between each trace) and attack traces with a fixed key (with no recomputation of the RTK), the resulting datasets are misaligned. Our solution was to force a recomputation of the RTK for the attack dataset. We are still unsure why the traces are misaligned despite the computation happening outside of the triggers and ack signals cleanly segregating the RTK computation and the encryption (most likely a pipeline issue).



