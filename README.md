# PruneGNN
This appendix describes the process of acquiring a subset of experimental results represented in the paper "PruneGNN: An Optimized Algorithm-Hardware
Framework for Graph Neural Network Pruning (HPCA 2024)". The artifacts include the implementations of the proposed sparse training algorithm and the LASSO regression-based pruning algorithm for GNN sparsification during training for various GNN models on real-world graphs, as well as the performance evaluation of pruned GNN inference using cuSPARSE and Prune-SpMM kernels.


## Requirements
### Hardware
- GPU: NVIDIA A100
- Disk Space: 15 GB

### Software
- Linux
- zip
- wget
- Docker
- NVIDIA A100 GPU drivers supporting CUDA 11.7

## Download and Extract PruneGNN Archive
Download and extract the PruneGNN archive:

    git clone https://github.com/grvndnz/PruneGNN
    cd PruneGNN
    

## Download and Extract Dynamic Datasets
Download and extract the dynamic datasets evaluated in PruneGNN:

    ./get_dataset.sh

## PruneGNN Docker Container Setup
### Create and Attach to Container
Create a new PruneGNN Docker container and install its dependencies using the following script:

    ./docker_setup.sh

Attach to the PruneGNN container as follows:

    docker attach prune-gnn

### Build PruneGNN PyTorch Extension 
Build the PruneGNN kernels and PyTorch extension within the PruneGNN container:

    cd /PruneGNN
    ./prune_gnn_setup.sh

## Run Experiments

### 1. Sparsity & Accuracy Analysis

To run compare the irregular pruning, lasso structured pruning and sparse training based structured pruning, run the following commands:

    cd /PruneGNN/experiments/pruning/
    python training_driver.py -r 5 --config train_config

This will run the training for all configurations in "config/train_config.py" for 5 times and generate the accuracy and sparsity results achieved by each algorithm. The average of the 5 runs are reported in "logs/train_config_summary.csv" for each dataset and model. The 3 evaluated algorithms should generate the same model accuracy (+/- 2%) under their corresponding sparsity rates. 

### 2. Inference Performance Analysis

To compare 4 systems for inference (i.e., (1) Irregular Pruning and cuSPARSE kernels, (2) LASSO Structured Pruning and cuSPARSE SpMM kernels, (3) LASSO Structured Pruning and Prune-SpMM kernels, and (4) Structured Sparse Training and Prune-SpMM kernels), run the following commands:

    cd /PruneGNN/experiments/inference
    python performance_driver.py -r 5 --config inference_config

This will measure the completion times of GPU kernels used during inference for the configurations given in "config/inference_config.py" (using the same sparsity rates tested in the previous experiment) using NVIDIA Nsight Compute profiler. The average completion time for each system per dataset and model are saved in "logs/inference_config_summary.csv".
