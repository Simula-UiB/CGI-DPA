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

**Acknowledgement** 

## License

This repositery is licensed under multiple licenses.

The code derived from the [ChipWhisperer](https://github.com/newaetech/chipwhisperer) project is licensed under GPLv3, as denoted in their files.
The rest of the code is licensed under the MIT license.

* GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.html)
* MIT license ([LICENSE](../LICENSE) or http://opensource.org/licenses/MIT)


## Overview

