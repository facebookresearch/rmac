# Robust Multi-Agent Counterfactual Prediction (RMAC)

This repository contains the research code for [Robust Multi-Agent Counterfactual Prediction](https://arxiv.org/abs/1904.02235), presented at NeurIPS 2019.

RMAC is a method for using logged data from a game (mechanism) to make predictions about what would happen if we "changed the rules of the game". RMAC computes estimates that are robust to agents that are not perfectly rational, or reward functions that are imperfectly specified.

For details of the RMAC algorithm, please rrefer to the paper.

To reference this work, please use:


```
@inproceedings{peysakhovich2019robust,
  title={Robust Multi-agent Counterfactual Prediction},
  author={Peysakhovich, Alexander and Kroer, Christian and Lerer, Adam},
  booktitle={NeurIPS 2019: Conference on Neural Information Processing Systems},
  year={2019}
}
```

## Requirements
RMAC requires Python3 and PyTorch.

## Running RMAC
* Clone this repo.
* To reproduce the experiments in the paper, run one of the run_XXX.sh scripts, or
* Write your own example similar to auction_exact.py, using the rmac.py library.


See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
RMAC is CC-by-NC licensed, as found in the LICENSE file.
