#!/usr/bin/env python

from argparse import ArgumentParser, FileType
from collections import OrderedDict
from csv import DictReader
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import dataclasses
import importlib
import itertools
import logging
import os
import re
import subprocess
import sys

import pandas

from config.train_config import Configuration

pandas.options.mode.chained_assignment = None

### Don't worry about much below here ###

# Argument parsing
parser = ArgumentParser()
parser.add_argument("-l", "--log-file", type=Path, default=Path("logs/pruning_summary.log"), help="Log file")
parser.add_argument("-s", "--summary", type=Path, default=Path("logs/pruning_summary.csv"), help="Summary output file")
parser.add_argument("-r", "--repeats", type=int, default=5, help="Repeats")
parser.add_argument("-e", "--epochs", type=int, default=600, help="Epochs")
parser.add_argument("--config", type=str, help="Name of the configuration file")


args = parser.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))


config_file_path = "config." + args.config

# Use importlib to import the specified module
config_module = importlib.import_module(config_file_path)

# Access the configurations variable from the module
configurations = config_module.configurations

# Create directories
args.log_file = Path("logs/" + args.config + "_summary.log")
args.summary = Path("logs/" + args.config + "_summary.csv")
args.log_file.parent.mkdir(parents=True, exist_ok=True)
args.summary.parent.mkdir(parents=True, exist_ok=True)

# Logging setup
## Main logger
logger = logging.getLogger(sys.argv[0])
logger.setLevel(logging.INFO)

## Formatter
formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] %(message)s')

## Console handler
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

## File handler
fh = logging.FileHandler(args.log_file)
fh.setFormatter(formatter)
logger.addHandler(fh)

## Initial log line
logger.info(f"Started {sys.argv[0]}")
logger.info("Arguments: " + ', '.join(map(lambda x: f"{x[0]} = {x[1]}", vars(args).items())))

# Summary
summary_columns = list(map(lambda x: x.name, dataclasses.fields(Configuration))) + ["accuracy", "sparsity"]
summary_file = open(args.summary, "w", buffering=1)
print(f"{','.join(summary_columns)}", file=summary_file)

test = None

def print_failed_rows(instance):
    instance = vars(instance)
    instance_values = list(map(str, instance.values()))

    fail_row_results = ",".join(instance_values + ["FAILED"])
    fail_row_summary = ",".join(instance_values + ["FAILED"] * (len(summary_columns) - len(instance_values)))

    print(fail_row_summary, file=summary_file)

def run_subprocess(command, log_stdout=True):
    # Environment overrides to sort out some weird errors...
    env = {"MKL_SERVICE_FORCE_INTEL": "1"}
    env.update(os.environ)

    try:
        completed_proc = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, env=env)
        if log_stdout:
            logger.info(completed_proc.stdout)
        return completed_proc
    except subprocess.CalledProcessError as e:
        if e.stdout:
            logger.error(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        return None
    except KeyboardInterrupt:
        sys.exit(2)

def process_results(results_stdout, instance):
    # Strip leading and trailing space
    summary = list(map(str.strip, results_stdout))
    
    # Get accuracy and sparsity
    summary = list(map(lambda x: re.match(r'.*Accuracy Result:\s+(?P<accuracy>(\d+\.\d+)|\d+)\s*Sparsity Result:\s+(?P<sparsity>(\d+\.\d+)|\d+)', x, re.DOTALL).groupdict(), summary))
    
    # Convert to float
    summary = list(map(lambda x: {k: float(v) for k, v in x.items()}, summary))

    # Convert to dataframe
    summary: pandas.DataFrame = pandas.DataFrame(summary)

    # Compute summary (mean)
    summary = summary.mean().to_frame().transpose()

    # Append instance fields
    summary[list(vars(instance).keys())] = tuple(vars(instance).values())

    # Order columns
    summary = summary[summary_columns]

    # Write summary to file
    summary.to_csv(summary_file, index=False, header=False)

def build_command(instance):

    if instance.model_type == "DGNN":
        command_args = f" ".join([f"--{k} {v}" for k, v in vars(instance).items() if k not in ['algorithm', 
                                                                                               'model_type', 
                                                                                               'hidden_channels',
                                                                                               'num_cols']])
        if instance.algorithm == "ST":
            command = f"python DGNN/run_exp_sparse.py {command_args}"
        elif instance.algorithm == "lasso":
            command = f"python DGNN/run_exp_sparse_lasso.py {command_args}"

    else:
        command_args = f"--epochs {args.epochs} " + " ".join([f"--{k} {v}" for k, v in vars(instance).items() if k not in ['algorithm']])

        if instance.algorithm == "ST":
            command = f"python GNN/main.py {command_args}"
        elif instance.algorithm == "lasso":
            command = f"python GNN/main_lasso.py {command_args}"

    return command

def run(instance):
    # Build command
    command = build_command(instance)

    results_stdout: list[str] = []

    for repeat in range(args.repeats):    
        logger.info(f"{command}")

        # Run command
        completed_proc = run_subprocess(command)
        if not completed_proc:
            print_failed_rows(instance)
            return
        results_stdout.append(completed_proc.stdout)

    # Process results
    process_results(results_stdout, instance)

def main():   
    for configuration in configurations:
        for instance in configuration.instances:
            logger.info("Running " + ', '.join(map(lambda x: f"{x[1]}", vars(instance).items())))
            run(instance)

if __name__ == "__main__":
    main()