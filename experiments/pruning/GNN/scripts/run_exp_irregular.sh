#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export DATASET="Cora" #Cora, CiteSeer, Pubmed, NELL, Flickr, Yelp
export MODEL="GCN" #GCN, GIN, GAT
export TRAINING="sparse" 
export PRUNING="irregular"
export SPARSITY="0.9"
export HIDDEN=16

# Run the Python script
python main.py \
        --training_type $TRAINING \
        --pruning_type $PRUNING \
        --device cuda \
        --report_freq 10 \
        --dataset_name $DATASET \
        --model_type $MODEL \
        --hidden_channels $HIDDEN \
        --epochs 600 \
        --sparsity_rate $SPARSITY 


