# Minecraft Build Generator


## Overview

**Minecraft Build Generator** is a personal research-oriented project focused on 3D discrete voxel generation applied to Minecraft structures.

Current objectives:
- Generate Minecraft builds from scratch
- Enable **conditional generation** (e.g. applying themes or styles)
- Experiment with **voxel-based generative models**

> Status: Work in progress

## Data

The data is sourced from: https://www.kaggle.com/datasets/shauncomino/minecraft-builds-dataset

It contains both `.schem` (28k) and `.h5` (11k) files. For now this project focuses on [schematic files](https://www.minecraft-schematics.com/).  

## Pipeline

All parameters are centralized in `conf/parameters.yaml`.

The pipeline is divided into 4 main stages:

### 1. Extraction

- Locate and index raw `.schem` files
- Extract metadata of `.schem` files

### 2. Processing

Transforms raw `.schem` files into clean, standardized voxel tensors.

#### 2.1 Build Filtering
- Size constraints (min/max dimensions)
- Removes statistical outliers using MAD (e.g. extreme volumes)
- Removes builds with no relevant blocks (i.e. removes builds without player-made structures)
- Optionally limits dataset size

#### 2.2 Block Filtering
Reduces noise in block vocabulary
- Removes rare blocks
- Merges rare block variants into main variant
- Noramlizes block representation 
- Analyze block usage (e.g. *cake[bite=3]"* -> *cake[bite=0]*)

#### 2.3 Tensor Conversion
Each build is:
- Converted into a **3D tensor (voxel grid)** of indexed blocks (block -> integer mapping)
- **Cropped** to remove empty regions (air-only borders) 

### 3. Training

Train a baseline **3D Variational Autoencoder** (VAE) on voxel data.

#### 3.1 Model
- Input: discrete voxel grids, padded if needed
- Embedding layer for block tokens
- Encoder with 3D convolutions and GroupNorm with leakyReLU
- Latent Space with 3D convolutions (to keep spatial size)
- Decoder with 3D transposed convolutions with leakyReLU

#### 3.2 Loss Function
Combination of
- **Cross-Entropy loss** to predict block types (discrete) 
-**KL Divergence** to regularize latent space
A **KL annealing schedule** is used to stabilize training

> [!NOTE]
> Current limitation: small batch sizes due to 3D memory cost

### 4. Validation / Generation

Generates `.schem` files for qualitative evaluation in Minecraft.

Two modes:
- **Generation**: sample new builds from random latent space
- **Reconstruction**: encode + decode existing builds

Outputs are exported as `.schem` files, [loadable in Minecraft](https://modrinth.com/mod/litematica)

## Installation

This project uses [__uv__](https://docs.astral.sh/uv/) as a package and environment manager.

#### Install uv
```
curl -Ls https://astral.sh/uv/install.sh | sh
```

#### Setup environment
```
uv init
uv sync
```

## Usage
If using Visual Studio Code Run preconfigured setup at `vscode/launch.json` (All/Extraction/Processing/Training/Validation).

#### Run full pipeline
```
python src/mcbuild_generator/pipeline.py --config conf/parameters.yaml
```

#### Run individual stages
```
python src/mcbuild_generator/<stage>/pipeline.py --config conf/parameters.yaml
```
Where `<stage>` is one of:
- `<extraction>`
- `<processing>`
- `<training>`
- `<validadtion>`

> [!NOTE]
> You can also run the training pipeline on a notebook (eg: Kaggle notebook for GPU) using `notebooks/kaggle_run.ipynb`. This will install the project as a library.
