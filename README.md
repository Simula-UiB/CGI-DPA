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

### Hamming Weight Experiments

Under the folder [hamming_weight](hamming_weight/) are three Python scripts that we used to perform the Hamming Weight simulation experiments (for more information see Section 5 of the paper).

The [16sbox](hamming_weight/16sbox_skinny.py) script exploits exactly one S-Box per key byte; we used a simplification and only computed scores for one S-Box of the first round and one S-Box of the last round (the ones for K1 and K9). We then multiply the resulting success rates to obtain the success rate for the full key. It outputs a final success rate in log2 scale (a success rate of x translates to $1/{2}^{x}\%$ chance of recovering the key).

The [32sbox](hamming_weight/32sbox_skinny.py)  script exploits all S-Boxes that depend on a single key byte. Each key byte is still handled separately, with the success rate of each key byte being multiplied at the end, but this time we compute the success rate for all of the 32 S-Boxes. It also outputs a log2 success rate.

The [44sbox](hamming_weight/44sbox_skinny.py)  script uses all 44 S-Boxes and CGI-DPA, as explained in Section 4 of the paper. The cluster graph is hardcoded in the form of edges and nodes with a predefined order of the message transmitted. It outputs a `.csv` file with rows shaped (`key_id;rank;ranks;...;rank;`) where the rank is the position of the true key at a given trace.

All those scripts take as a mandatory command line argument (in this order) `SIGMA` (`float`, $\sigma^2$ the variance of the noise), `N_TRACES` (`int`, number of traces per experiment), `N_EXPERIMENT`(`int`, number of experiments to perform before averaging). Additionally, 44sbox takes two extra optional arguments, `OUTPUT_PATH` (`string`, the output path for the `.csv` file, a default one is given otherwise), `SEED` (`int`, the seed to use for the PRNG, if none provided 4 bytes are taken from `/dev/urandom`). As a disclaimer, the code is not exactly "well written". There is a lot of code duplication that could be improved, but we decided to minimize the changes from the version we used for the submission.