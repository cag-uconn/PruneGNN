# Profile
    ncu --nvtx --target-processes all --metrics gpu__time_duration.avg -f -o profile python main.py --epochs 1 --dataset_name Cora --model_type GIN
    ncu -i profile.ncu-rep --csv --kernel-name-base demangled --print-units base --print-summary per-nvtx > profile.csv
    python analyze.py profile.csv