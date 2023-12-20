#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
torch::Tensor prune_spmm(torch::Tensor input,
                          torch::Tensor row_pointer,
                          torch::Tensor column_index,
                          torch::Tensor degrees,
                          int threads_per_warp,
                          int gin,
                          float epsilon);

torch::Tensor cusparse_spmm_row(torch::Tensor input,
                          torch::Tensor row_pointer,
                          torch::Tensor column_index,
                          torch::Tensor degrees);

torch::Tensor cublas_gemm(torch::Tensor A,
                          torch::Tensor B);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor prune_spmm_wrap(torch::Tensor input,
                               torch::Tensor row_pointer,
                               torch::Tensor column_index,
                               torch::Tensor degrees,
                               int threads_per_warp,
                               int gin,
                               float epsilon)
{
    CHECK_INPUT(input);
    CHECK_INPUT(row_pointer);
    CHECK_INPUT(column_index);
    CHECK_INPUT(degrees);

    // Cast tensors
    input = input.to(torch::kFloat32);
    row_pointer = row_pointer.to(torch::kInt32);
    column_index = column_index.to(torch::kInt32);
    degrees = degrees.to(torch::kFloat32);

    return prune_spmm(input, row_pointer, column_index, degrees, threads_per_warp, gin, epsilon);
}


torch::Tensor cusparse_spmm_row_wrap(torch::Tensor input,
                               torch::Tensor row_pointer,
                               torch::Tensor column_index,
                               torch::Tensor degrees)
{
    CHECK_INPUT(input);
    CHECK_INPUT(row_pointer);
    CHECK_INPUT(column_index);
    CHECK_INPUT(degrees);

    // Cast tensors
    input = input.to(torch::kFloat32);
    row_pointer = row_pointer.to(torch::kInt32);
    column_index = column_index.to(torch::kInt32);
    degrees = degrees.to(torch::kFloat32);

    return cusparse_spmm_row(input, row_pointer, column_index, degrees);
}


torch::Tensor cublas_gemm_wrap(torch::Tensor A,
                               torch::Tensor B)
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    // Cast tensors
    A = A.to(torch::kFloat32);
    B = B.to(torch::kFloat32);
 
    return cublas_gemm(A, B);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("prune_spmm", &prune_spmm_wrap, "Prune SPMM");
    m.def("cusparse_spmm_row", &cusparse_spmm_row_wrap, "CUSPARSE SPMM Row");
    m.def("cublas_gemm", &cublas_gemm_wrap, "CUBLAS GEMM");
}