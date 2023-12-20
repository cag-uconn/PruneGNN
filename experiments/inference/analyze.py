import argparse
import csv
from pathlib import Path
from pprint import pprint
import re
import sys

import pandas as pd


# ranged_kernel = {

#     r".*agg.*": [
#         r".*FillFunctor.*",
#         r".*direct_copy_kernel_cuda.*"
#     ],
#     r".*comb.*": [
#         r".*FillFunctor.*",
#         r".*direct_copy_kernel_cuda.*"
#     ],
# }


def analyze(rows, ranged_kernel):

    ## gpu__time_duration.avg only
    rows = list(filter(lambda x: x["Metric Name"] == "gpu__time_duration.avg", rows))

    # Transform
    ## Remove fields we don't care about
    rows = list(map(lambda x: {k: v for k, v in x.items() if k in ["Kernel Name", "Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg", "Invocations", "Metric Name", "Metric Unit", "Average"]}, rows))

    ## Compute NVTX Range
    for row in rows:
        row["Range"] = ".".join(list(map(lambda x: x.split(":")[0].replace('"', ''), row["Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].strip().split())))
        del row["Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"]

    ## Make numbers numbers
    for row in rows:
        row.update({"Average": float(row["Average"]), "Invocations": int(row["Invocations"])})
        if row["Metric Unit"] == "nsecond":
            row["Average"] /= 1000
            row["Metric Unit"] = "usecond"

    ## Merge rows with same kernel and range
    def merge_rows(r1, r2):
        invocations = r1["Invocations"] + r2["Invocations"]
        average = (r1["Invocations"] * r1["Average"] + r2["Invocations"] * r2["Average"]) / invocations
        return {"Invocations": invocations, "Average": average}

    new_rows = {}
    for row in rows:
        key = (row["Kernel Name"], row["Range"])
        if key not in new_rows:
            new_rows[key] = row
        else:
            new_rows[key].update(merge_rows(new_rows[key], row))
    rows = list(new_rows.values())

    # Sum by range
    def is_excluded(row):
        for range, kernels in ranged_kernel.items():
            if re.search(range, row["Range"]):
                if any([re.search(p, row["Kernel Name"]) for p in kernels]):
                    return True
        return False

    output = {}
    for row in rows:
        if is_excluded(row):
            continue

        key = row["Range"]

        if key not in output:
            output[key] = {
                "Sum": row["Average"] * row["Invocations"], 
                "Total Kernel Invocations": row["Invocations"], 
                "Metric Name": row["Metric Name"],
                "Metric Unit": row["Metric Unit"]
                }
        else:
            output[key]["Sum"] += row["Average"] * row["Invocations"]
            output[key]["Total Kernel Invocations"] += row["Invocations"]

    # pprint(output)

    indices = sorted(list(set([range.split(".")[1] for range in output.keys()])))
    columns = list(set([range.split(".")[-1] for range in output.keys()]))

    df = pd.DataFrame(0.0, index=indices, columns=columns)

    for k, v in output.items():
        range = k.split(".")
        layer_name = range[1]
        op = range[-1]
        df.at[layer_name, op] = v["Sum"]

    # print(df.to_csv(index_label="layer"))
   
    df = df.reindex(sorted(df.columns), axis=1)
    # print(df)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)

    args = parser.parse_args()

    csv_file: Path = args.csv_file

    # Load CSV
    if not csv_file.is_file():
        raise FileNotFoundError()
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    print(analyze(rows, ranged_kernel))

    