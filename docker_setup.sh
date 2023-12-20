#!/bin/bash

docker run --name prune-gnn -it -d --ipc=host --cap-add=SYS_ADMIN --gpus all -v $PWD:/PruneGNN pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

docker exec prune-gnn bash -c "apt update && DEBIAN_FRONTEND=noninteractive apt install -y wget unzip nsight-compute-2022.4.1 nsight-systems-2023.2.3 && conda install -y -c pyg -c conda-forge -c nvidia pyg=2.3.1 pytorch-sparse nvtx nsight-compute && pip install ogb matplotlib && pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html && pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html"