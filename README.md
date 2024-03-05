# DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators
In this work, we build a differentiable analytical model to enable mapping-first design space exploration of deep learning accelerator designs. We also apply deep learning to adapt this model to the Gemmini accelerator's RTL implementation.

For more details, please refer to:
- [MICRO'23 DOSA Paper](https://people.eecs.berkeley.edu/~ysshao/assets/papers/dosa-micro2023.pdf)
```BibTex
@inproceedings{
  hong2023dosa,
  title={DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators},
  author={Charles Hong and Qijing Huang and Grace Dinh and Mahesh Subedar and Yakun Sophia Shao},
  booktitle={IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2023},
  url={https://people.eecs.berkeley.edu/~ysshao/assets/papers/dosa-micro2023.pdf}
}
```

## Installation

(modified by Zhehai and Anish)

`conda create --name dosa python=3.10`

`conda activate dosa`

Requires `python>=3.10.0`.

To install Python dependencies:

```
pip install -e .
```
