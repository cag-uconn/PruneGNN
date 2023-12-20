
from argparse import ArgumentParser, FileType
from collections import OrderedDict
from csv import DictReader
import dataclasses
from dataclasses import dataclass
import itertools
from io import StringIO
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

# Configuration Dataclass
@dataclass
class Configuration:
    # Add new configuration fields here
    # Must be of form "field_name: list[...]"
    dataset_name: list[str]
    model_type: list[str]
    hidden_channels: list[int]
    num_cols: list[int]
    sparsity_rate: list[int]
    training_type: list[str]
    pruning_type: list[str]
    algorithm: list[str]

    @property
    def instances(self):
        self_dict = OrderedDict(dataclasses.asdict(self))
        for x in itertools.product(*self_dict.values()):
            instance = SimpleNamespace(**dict(zip(self_dict.keys(), x)))
            yield instance

# Configurations
# Define configurations to run
# For each configuration field, provide a list of possible values
# At runtime, all possible combinations of inputs will be generated


configurations = [


    ######### GCN MODEL ##########
    Configuration(
        dataset_name=["Cora"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[16],
        sparsity_rate=[0.90],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[15],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[7],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),


    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[16],
        sparsity_rate=[0.99],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[14],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[7],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[16],
        sparsity_rate=[0.98],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[12],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GCN"],
        hidden_channels=[16],
        num_cols=[3],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.70],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[62],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[24],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Flickr"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.97],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[26],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[3],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.99],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Yelp"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[60],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Yelp"],
        model_type=["GCN"],
        hidden_channels=[64],
        num_cols=[15],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),


    ########## GIN MODEL ##########
    Configuration(
        dataset_name=["Cora"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[16],
        sparsity_rate=[0.97],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[14],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[11],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),


    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[16],
        sparsity_rate=[0.77],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[12],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[6],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[16],
        sparsity_rate=[0.95],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[12],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GIN"],
        hidden_channels=[16],
        num_cols=[5],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.98],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[60],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[40],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Flickr"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.97],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[57],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[16],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.99],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Yelp"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[40],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Yelp"],
        model_type=["GIN"],
        hidden_channels=[64],
        num_cols=[2],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),


    ########## GAT MODEL ##########
    Configuration(
        dataset_name=["Cora"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.97],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[54],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[16],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),


    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.99],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[39],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[17],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.95],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[33],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[4],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.80],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[56],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[31],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Flickr"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.97],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[28],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[4],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[64],
        sparsity_rate=[0.95],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["Yelp"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[62],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["Yelp"],
        model_type=["GAT"],
        hidden_channels=[64],
        num_cols=[7],
        sparsity_rate=[0],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),


    ######### DGNN MODEL ##########
    Configuration(
        dataset_name=["autonomous_syst"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.92],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
        
    ),
    Configuration(
        dataset_name=["autonomous_syst"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.05],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["autonomous_syst"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.5],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["elliptic_temporal"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.97],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["elliptic_temporal"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.3],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["elliptic_temporal"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.8],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

    Configuration(
        dataset_name=["reddit"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.99],
        training_type=["sparse"],
        pruning_type=["irregular"],
        algorithm=["ST"]
    ),
    Configuration(
        dataset_name=["reddit"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.35],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["lasso"]
    ),
    Configuration(
        dataset_name=["reddit"],
        model_type=["DGNN"],
        hidden_channels=[-1],
        num_cols=[-1],
        sparsity_rate=[0.95],
        training_type=["sparse"],
        pruning_type=["column"],
        algorithm=["ST"]
    ),

  ]