# Context-Aware Flow Matching

Authors: Lars Kühmichel

## This site is a work in progress. Please check back later.

[GitHub Repository](https://github.com/LarsKue/context-aware-flow-matching)

[GitHub Pages Project Page](https://larskue.github.io/context-aware-flow-matching/)

This repository contains parts of the code used in my master's thesis, titled
"Advancements in Context-Aware Learning and Generative Modeling".

## 1. Introduction

Our approach to context-aware learning is defined in the thesis as
deep learning using an embedding from a set of context inputs:

<div align="center">
    <img src="docs/context-aware-learning.webp" alt="Context-Aware Learning">
</div>


In this repo, we use [Optimal Transport Flow Matching](https://arxiv.org/abs/2302.00482) to leverage this
embedding and learn a  generative model that can be conditioned on sampled
context embeddings, thus enabling interpolation between contexts:

<div align="center">
    <img src="docs/context-aware-flow-matching.webp" width=50% alt="Interpolation">
</div>

## 2. Install

Create a new conda environment with the required dependencies:
```bash
conda env create -f env.yaml
```

Activate the environment:
```bash
conda activate context-aware-flow-matching
```

Verify your install by running pytest:
```bash
pytest tests -m "not slow"
```

If you want to plot samples using blender, install the blender env instead:

```bash
conda env create -f blender.yaml
```

Activate and verify as above.

Note that these environments are incompatible with each other,
because they each require different python versions.

## 3. Experiments

Experiment notebooks can be found in the `experiments` folder. We use
[Lightning-Trainable](https://github.com/LarsKue/lightning-trainable)
to train our models. Each notebook contains the hyperparameters used for training.

### 3.1 ModelNet10

**Dataset:** [ModelNet10](https://3dvision.princeton.edu/projects/2014/3DShapeNets/)

<div align="center">
    <img src="docs/modelnet10/data_samples.webp" width=25% alt="ModelNet10 Dataset Samples">
    <img src="docs/modelnet10/reconstructions.webp" width=25% alt="ModelNet10 Model Reconstructions">
    <img src="docs/modelnet10/model_samples.webp" width=25% alt="ModelNet10 Model Samples">

    Left: Samples from the dataset. Middle: Model Reconstructions. Right: Samples from the trained model.
</div>

<div align="center">
    <img src="docs/modelnet10/interpolation.webp" width=50% alt="ModelNet10 Context Interpolation">

    Linear interpolation between randomly sampled contextual embeddings.
</div>

<div align="center">
    <img src="docs/modelnet10/rotation.webp" width=50% alt="ModelNet10 Rotation">

    Rotating points in the latent space of the flow helps visualize the shape manifold.
</div>

<div align="center">
    <img src="docs/modelnet10/metrics.webp" alt="ModelNet10 Evaluation Metrics">

    Evaluation metrics on the test set. The model is competitive with other state-of-the-art approaches.
</div>


### 3.2 LIDAR-CS (WIP)

**Dataset:** [LIDAR-CS](https://github.com/LiDAR-Perception/LiDAR-CS)

Left: Samples from the dataset. Right: Samples from the trained model.

<div align="center">
    <img src="docs/lidar-cs/data_samples.webp" width=25% alt="LIDAR-CS Dataset Samples">
    <img src="docs/lidar-cs/reconstructions.webp" width=25% alt="LIDAR-CS Model Reconstructions">
    <img src="docs/lidar-cs/model_samples.webp" width=25% alt="LIDAR-CS Model Samples">
    
    Left: Samples from the dataset. Middle: Model Reconstructions. Right: Samples from the trained model.
</div>


<div align="center">
    <img src="docs/lidar-cs/out-of-distribution.webp" width=50% alt="LIDAR-CS Out-of-Distribution Samples">

    Samples from the test set of the LIDAR-CS dataset, which were marked as out-of-distribution by the model.
</div>


<div class="row">
    <img src="docs/lidar-cs/metrics.webp" alt="LIDAR-CS Evaluation Metrics">
    
    Evaluation metrics on the test set. The model is competitive with other state-of-the-art approaches.
</div>


## 4. References

See my thesis: [Advancements in Context-Aware Learning and Generative Modeling](docs/thesis.pdf)
