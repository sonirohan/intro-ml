# Project 3: Language modelling and Transformers

## Description

In this project, we will implement the core layers of a transformer-based language model: single head causal attention, multi head attention, the feed forward layer and layer norm. The `TransformerLayer` and `MiniGPT` classes that put these layers together, and the training script, are provided. Everything uses the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Since we are resource constrained we cannot train a large enough model to convergence to see very good results, so we will provide you with model weights for a 20M parameter model trained on the same dataset. You will load these weights into the model built from your layers and use it to generate a mini story.

Training the model yourself is optional and not graded. Note that this will be extremely slow on CPU but should be fun if you have access to a GPU.

Happy Coding!

## Installation

To install the required packages, run the following command:

```bash
pip install numpy torch tiktoken wandb einops
```

where:

1. `numpy` is a library for numerical computing. **Required**.
2. `torch` is the PyTorch library. **Required**.
3. `tiktoken` is a library for tokenizing text data. **Required**.
4. `wandb` is the Weights and Biases library for tracking experiments, it is optional but recommended.
5. `einops` is a library for tensor manipulation, it is fantastic and I highly recommend it. **Required**, since the provided code in `model.py` and `train.py` uses it.

You are not allowed to use any other external libraries (like `transformers` or `torchtext`) for this project.
Please come talk to us if there is a particular library you would like to try out.

 Also please do not import any other functionality from torch than is already imported in the starter code. Minor imports (like torch.var or torch.std) are allowed but not big ones like (nn.TransformerEncoderLayer).

## Dataset

Dataset is downloaded and split into train and validation for you check the `data` folder.

## Pretrained Models

Note: Pretrained models will be provided in the zipped file, if not you can download them from the link below.

Download all pretrained models from [this link](https://drive.google.com/file/d/1g09qUM9WibdfQVgkj6IAj8K2S3SGwc91/view?usp=sharing)

And add the folder to the root directory. That is all the pretrained models should be in the `pretrained_models` folder in the MiniGPT folder.
