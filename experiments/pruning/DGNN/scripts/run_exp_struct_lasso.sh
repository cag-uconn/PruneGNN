#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export DATASET="elliptic_temporal" #autonomous_syst, reddit or elliptic_temporal
export TRAINING="sparse" 
export PRUNING="column"
export SPARSITY="0.9"

# Run the Python script
python run_exp_sparse_lasso.py \
  --dataset_name $DATASET \
  --training_type $TRAINING \
  --pruning_type $PRUNING \
  --sparsity_rate $SPARSITY



