# Context-Aware Flow Matching

## This site is a work in progress. Please check back later.

[Repository](https://github.com/LarsKue/context-aware-flow-matching)

[Pages Website](https://larskue.github.io/context-aware-flow-matching/)

This repository contains parts of the code used in my master's thesis, titled
"Advancements in Context-Aware Learning and Generative Modeling".

## 1. Introduction

Our approach to context-aware learning is defined in the thesis as
deep learning using an embedding from a set of context inputs:

<div class="row">
    <div class="col">
        <img src="docs/context-aware-learning.webp" alt="Context-Aware Learning">
    </div>
</div>


In this repo, we use [Optimal Transport Flow Matching](https://arxiv.org/abs/2302.00482) to leverage this
embedding and learn a  generative model that can be conditioned on sampled
context embeddings, thus enabling interpolation between contexts:

<div class="row">
    <div class="col">
        <img src="docs/context-aware-flow-matching.webp" alt="Interpolation">
    </div>
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



Dataset: [ModelNet10](https://3dvision.princeton.edu/projects/2014/3DShapeNets/)

Left: Samples from the dataset. Right: Samples from the trained model.

<div class="row">
    <div class="col">
        <img src="docs/modelnet10/data_samples.webp" alt="ModelNet10 Dataset Samples">
    </div>
    <div class="col">
        <img src="docs/modelnet10/model_samples.webp" alt="ModelNet10 Model Samples">
    </div>
</div>

We can interpolate between contexts:

<div class="row">
    <div class="col">
        <img src="docs/modelnet10/interpolation.webp" alt="ModelNet10 Context Interpolation">
    </div>
</div>

The shape manifold is particularly visible in this video,
where we rotate points in the latent space of the flow:

<div class="row">
    <div class="col">
        <img src="docs/modelnet10/rotation.webp" alt="ModelNet10 Rotation">
    </div>
</div>

The model is competitive with other state-of-the-art models:

<div class="row">
    <div class="col">
        <img src="docs/modelnet10/metrics.webp" alt="ModelNet10 Accuracy">
    </div>
</div>


### 3.2 LIDAR-CS

(Work in progress)

Dataset: [LIDAR-CS](https://github.com/LiDAR-Perception/LiDAR-CS)

Left: Samples from the dataset. Right: Samples from the trained model.

<div class="row">
    <div class="col">
        <img src="docs/lidar-cs/data_samples.png" alt="LIDAR-CS Dataset Samples">
    </div>
    <div class="col">
        <img src="docs/lidar-cs/model_samples.png" alt="LIDAR-CS Model Samples">
    </div>
</div>



