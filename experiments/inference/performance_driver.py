#!/usr/bin/env python

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
import importlib

import pandas


# Import analyze script
from analyze import analyze

pandas.options.mode.chained_assignment = None

# from config.gcn_lasso_st_comparison_training import configurations
# exp_name = "gcn_lasso_st_comparison_training"

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

### Don't worry about much below here ###

# Argument parsing
parser = ArgumentParser()
parser.add_argument("-l", "--log-file", type=Path, default=Path("logs") / sys.argv[0].replace(".py", ".log"), help="Log file")
parser.add_argument("-n", "--no-profile", action="store_true", help="Do not profile")
# parser.add_argument("-k", "--kernels-directory", type=Path, default=Path("../build"), help="Kernel binaries path")
# parser.add_argument("-d", "--datasets-directory", type=Path, default=Path("../../datasets"), help="Datasets path")
parser.add_argument("-r", "--repeats", type=int, default=5, help="Repeats")
parser.add_argument("-o", "--output", type=Path, default=Path("logs/results.csv"), help="Detailed results output file")
parser.add_argument("-s", "--summary", type=Path, default=Path("logs/summary.csv"), help="Summary output file")
parser.add_argument("--config", type=str, help="Name of the configuration file")


args = parser.parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the configurations from the specified file


config_file_path = "config." + args.config

# Use importlib to import the specified module
config_module = importlib.import_module(config_file_path)

# Access the configurations variable from the module
configurations = config_module.configurations



args.output = Path("logs/" + args.config + "_" + "results" + ".csv")
args.summary = Path("logs/" + args.config + "_" + "summary" + ".csv")
args.log_file = Path("logs/" + args.config + ".log")

# Create directories
args.log_file.parent.mkdir(parents=True, exist_ok=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
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

# Results
results_columns = list(map(lambda x: x.name, dataclasses.fields(Configuration))) + ["Layer"]
results_file = open(args.output, "w", buffering=1)
is_results_header_written = False

# Summary
summary_columns = list(map(lambda x: x.name, dataclasses.fields(Configuration)))
summary_file = open(args.summary, "w", buffering=1)
is_summary_header_written = False

test = None

def print_failed_rows(instance):
    instance = vars(instance)
    instance_values = list(map(str, instance.values()))

    fail_row_results = ",".join(instance_values + ["FAILED"])
    fail_row_summary = ",".join(instance_values + ["FAILED"])

    print(fail_row_results, file=results_file)
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

def process_profiler_results(profiler_output_path, instance):
    instance = vars(instance)

    # Run profiler again to parse results as csv
    completed_proc = run_subprocess(f"ncu -i {profiler_output_path.with_suffix('.ncu-rep')} --csv --kernel-name-base demangled --print-units base --print-summary per-nvtx", False)
    if not completed_proc:
        print_failed_rows(instance)
        return
    
    # Read profiler results into rows
    rows = [row for row in DictReader(completed_proc.stdout.splitlines())]
    if instance["kernel_type"] == "pruneSp":
        ranged_kernel = {
            r".*agg.*": [
                r".*at::native.*"
            ],
            r".*d_input.*": [
                r".*at::native.*"
            ],
            r".*d_weight.*": [
                r".*at::native.*"
            ],
           r".*comb.*": [
                r".*at::native.*"
            ]
        }
    else:
        ranged_kernel = {
            r".*d_weight.*": [
                r".*at::native.*"
            ],
           r".*comb.*": [
                r".*at::native.*"
            ]
        }
   
    # Analyze results
    # print(rows)
    profiler_results = analyze(rows, ranged_kernel)
    # print(profiler_results)

    # ## Average across epochs
    profiler_results /= args.repeats

    ## Calculate totals
    profiler_results["Total"] = profiler_results.sum(axis=1)
    profiler_results.loc["Total"] = profiler_results.sum(axis=0)

    # Summarized results
    summary = profiler_results.copy()

    ## Append instance fields
    summary[list(instance.keys())] = tuple(instance.values())

    ## Write summary header
    global is_summary_header_written, summary_columns
    if not is_summary_header_written:
        is_summary_header_written = True
        summary_columns += profiler_results.columns.to_list()
        print(f"{','.join(summary_columns)}", file=summary_file)

    ## Order columns
    summary = summary[summary_columns]

    ## Write summary to file
    summary.loc["Total"].to_frame().transpose().to_csv(summary_file, index=False, header=False)

    # Detailed results
    details = profiler_results.copy()

    ## Append instance fields
    details[list(instance.keys())] = tuple(instance.values())

    ## Write detailed header
    global is_results_header_written, results_columns
    if not is_results_header_written:
        is_results_header_written = True
        results_columns += profiler_results.columns.to_list()
        print(f"{','.join(results_columns)}", file=results_file)

    ## Convert index to layer column
    details.reset_index(inplace=True, names="Layer")

    ## Order columns
    details = details[results_columns]

    ## Write results to file
    details.to_csv(results_file, index=False, header=False)
         
def build_profile_command(command, profiler_output_path):
    return f"ncu --nvtx --target-processes all --metrics gpu__time_duration.avg -f -o '{profiler_output_path}' {command}"

def build_command(instance):
    # Do a really neat thing to expand arguments
    command = "python main.py " + " ".join([f"--{k} {v}" for k, v in vars(instance).items()]) + " --epochs " + str(args.repeats)

    return command

def run_and_profile(instance):
    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        profiler_output = Path(temp_dir) / "profile"

        # Build profiler command
        command = build_command(instance)
        if not args.no_profile:
            command = build_profile_command(command, profiler_output)
        
        logger.info(f"{command}")

        # Run command
        completed_proc = run_subprocess(command)
        if not completed_proc:
            print_failed_rows(instance)
            return

        # Process profiler results
        if not args.no_profile:
            process_profiler_results(profiler_output, instance)

def main():   
    for configuration in configurations:
        for instance in configuration.instances:
            logger.info("Running " + ', '.join(map(lambda x: f"{x[1]}", vars(instance).items())))
            run_and_profile(instance)

if __name__ == "__main__":
    main()