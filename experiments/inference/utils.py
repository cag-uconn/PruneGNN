import torch
import math

def generate_random_sparse_matrix(num_rows, num_cols, sparsity):
    # Generate a dense matrix
    dense_matrix = torch.randn(num_rows, num_cols)

    # Determine the number of elements to keep based on sparsity
    num_non_zero_elements = int(math.ceil(sparsity * num_rows * num_cols))

    # Randomly set elements to zero to achieve the desired sparsity
    mask = torch.randperm(num_rows * num_cols)[:num_non_zero_elements]
    dense_matrix.view(-1)[mask] = 0.0

    return dense_matrix.to_sparse_csr()


def prune_irr(args, input, sparsity):
    return generate_random_sparse_matrix(input.shape[0], input.shape[1], sparsity).to(args.device)