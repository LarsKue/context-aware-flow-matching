# Context-Aware Flow Matching

Authors: Lars Kühmichel

## This site is a work in progress. Please check back later.

[GitHub Repository](https://github.com/LarsKue/context-aware-flow-matching)

[GitHub Pages Project Page](https://larskue.github.io/context-aware-flow-matching/)

This repository contains parts of the code used in my master's thesis, titled
[Advancements in Context-Aware Learning and Generative Modeling](docs/thesis.pdf).

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

Experiment notebooks can be found in the `experiments` folder. We use
[Lightning-Trainable](https://github.com/LarsKue/lightning-trainable)
to train our models. Each notebook contains the hyperparameters used for training.

## 3. ModelNet10

**Dataset:** [ModelNet10](https://3dvision.princeton.edu/projects/2014/3DShapeNets/)

<div style="text-align: center;">
    <img src="docs/modelnet10/data_samples.png" alt="ModelNet10 Dataset Samples" style="width: 49.5%;">
    <img src="docs/modelnet10/reconstructions.png" alt="ModelNet10 Model Reconstructions" style="width: 49.5%;">
</div>
<div style="text-align: center; margin-bottom: 1cm;">
    Left: Samples from the dataset. Right: Model Reconstructions.
</div>

<img src="docs/modelnet10/model_samples.png" alt="ModelNet10 Model Samples" style="width: 100%; display: block; margin: auto;">
<div style="text-align: center; margin-bottom: 1cm;">
    Random samples from the trained model.
</div>

<img src="docs/modelnet10/interpolation.png" alt="ModelNet10 Context Interpolation" style="width: 100%; display: block; margin: auto;">
<div style="text-align: center; margin-bottom: 1cm;">
    Linear interpolation between randomly sampled contextual embeddings.
</div>

<video loop autoplay style="width: 75%; display: block; margin: auto;">
    <source src="docs/modelnet10/transition.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<div style="text-align: center; margin-bottom: 1cm;">
    Transition between the data and latent space.
</div>


<video loop autoplay style="width: 75%; display: block; margin: auto;">
    <source src="docs/modelnet10/rotation.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<div style="text-align: center; margin-bottom: 1cm;">
    Rotating points in the latent space of the flow helps visualize the shape manifold.
</div>

<img src="docs/modelnet10/metrics.png" alt="ModelNet10 Evaluation Metrics" style="width: 50%; display: block; margin: auto;">
<div style="text-align: center; margin-bottom: 1cm;">
    Evaluation metrics on the test set. The model is competitive with other state-of-the-art approaches.
</div>


## 4. LIDAR-CS

This is not part of the thesis, but I may revisit this dataset in the future.

**Dataset:** [LIDAR-CS](https://github.com/LiDAR-Perception/LiDAR-CS)

## 5. References

See my thesis: [Advancements in Context-Aware Learning and Generative Modeling](docs/thesis.pdf)

## 6. Citation

If this repo is useful to you in your research, please cite my thesis and related work:

```
@mastersthesis{kuehmichel2024advancements,
    author={Lars Kühmichel},
    title={Advancements in Context-Aware Learning and Generative Modeling},
    school={Heidelberg University},
    year={2024},
    month={01},
    day={22},
}

@misc{müller2023contextaware,
      title={Towards Context-Aware Domain Generalization: Representing Environments with Permutation-Invariant Networks}, 
      author={Jens Müller and Lars Kühmichel and Martin Rohbeck and Stefan T. Radev and Ullrich Köthe},
      year={2023},
      eprint={2312.10107},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
