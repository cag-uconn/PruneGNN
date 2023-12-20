
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
    compressed_dim: list[int]
    sparsity_type: list[str]
    algorithm: list[str]
    mode: list[str]
    kernel_type: list[str]
    sparsity_rate: list[int]

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

    ########## GCN MODEL ##########
    
    Configuration(
        dataset_name=["Cora"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.9]
    ),

    Configuration(
        dataset_name=["Cora"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[15],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[7],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.99]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[14],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[7],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.98]
    ),

    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[12],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GCN"],
        hidden_channels=[16],
        compressed_dim=[3],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.7]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[62],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[24],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),
    
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.97]
    ),

    Configuration(
        dataset_name=["Flickr"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[26],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),


    Configuration(
        dataset_name=["Flickr"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[3],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.99]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[60],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),


    Configuration(
        dataset_name=["Yelp"],
        model_type=["GCN"],
        hidden_channels=[64],
        compressed_dim=[15],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),


    ########## GIN MODEL ##########

    Configuration(
        dataset_name=["Cora"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.97]
    ),

    Configuration(
        dataset_name=["Cora"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[14],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[11],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.77]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[12],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[6],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

   Configuration(
        dataset_name=["Pubmed"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.95]
    ),

    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[12],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GIN"],
        hidden_channels=[16],
        compressed_dim=[5],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.98]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[60],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
  Configuration(
        dataset_name=["NELL"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[40],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

   Configuration(
        dataset_name=["Flickr"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.97]
    ),

    Configuration(
        dataset_name=["Flickr"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[57],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[16],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.99]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[40],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GIN"],
        hidden_channels=[64],
        compressed_dim=[2],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),


#    ########## GAT MODEL ##########

    Configuration(
        dataset_name=["Cora"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.97]
    ),

    Configuration(
        dataset_name=["Cora"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[54],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["Cora"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[16],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.99]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[39],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["CiteSeer"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[17],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),
    
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.95]
    ),

    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[33],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["Pubmed"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[4],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["NELL"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.80]
    ),
    Configuration(
        dataset_name=["NELL"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[56],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
  Configuration(
        dataset_name=["NELL"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[31],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["Flickr"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.97]
    ),

    Configuration(
        dataset_name=["Flickr"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[28],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),
    Configuration(
        dataset_name=["Flickr"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[4],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),

   Configuration(
        dataset_name=["Yelp"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.95]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[62],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0]
    ),

    Configuration(
        dataset_name=["Yelp"],
        model_type=["GAT"],
        hidden_channels=[64],
        compressed_dim=[7],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0]
    ),


     ########## DGNN MODEL ##########
    Configuration(
        dataset_name=["autonomous_syst"],
        model_type=["DGNN"],
        hidden_channels=[30],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.92]
    ),

    Configuration(
        dataset_name=["autonomous_syst"],
        model_type=["DGNN"],
        hidden_channels=[30],
        compressed_dim=[-1],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0.05]
    ),
    Configuration(
        dataset_name=["autonomous_syst"],
        model_type=["DGNN"],
        hidden_channels=[30],
        compressed_dim=[-1],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0.5]
    ),

    Configuration(
        dataset_name=["elliptic_temporal"],
        model_type=["DGNN"],
        hidden_channels=[76],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.97]
    ),

    Configuration(
        dataset_name=["elliptic_temporal"],
        model_type=["DGNN"],
        hidden_channels=[76],
        compressed_dim=[-1],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0.3]
    ),

    Configuration(
        dataset_name=["elliptic_temporal"],
        model_type=["DGNN"],
        hidden_channels=[76],
        compressed_dim=[-1],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0.8]
    ),




    Configuration(
        dataset_name=["reddit"],
        model_type=["DGNN"],
        hidden_channels=[152],
        compressed_dim=[-1],
        sparsity_type=["irregular"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["cusparse"],
        sparsity_rate=[0.99]
    ),

    Configuration(
        dataset_name=["reddit"],
        model_type=["DGNN"],
        hidden_channels=[152],
        compressed_dim=[-1],
        sparsity_type=["structured"],
        algorithm=["lasso"],
        mode=["inference"],
        kernel_type=["cusparse", "pruneSp"],
        sparsity_rate=[0.35]
    ),
    Configuration(
        dataset_name=["reddit"],
        model_type=["DGNN"],
        hidden_channels=[152],
        compressed_dim=[-1],
        sparsity_type=["structured"],
        algorithm=["ST"],
        mode=["inference"],
        kernel_type=["pruneSp"],
        sparsity_rate=[0.95]
    ),

]