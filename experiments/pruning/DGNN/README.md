# Run Experiments

Base training code is retrieved from https://github.com/IBM/EvolveGCN and extended with pruning.

For sparse training-based irregular pruning:
    
    ./scripts/run_exp_irregular.sh

For lasso-based structured pruning:

    ./scripts/run_exp_struct_lasso.sh
    
For sparse training-based structured pruning:

    ./scripts/run_exp_struct.sh