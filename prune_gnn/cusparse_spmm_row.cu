#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdexcept>
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <torch/extension.h>

using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        throw std::runtime_error("error");                                     \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        throw std::runtime_error("error");                                     \
    }                                                                          \
}

torch::Tensor cusparse_spmm_row(torch::Tensor input,
                          torch::Tensor row_pointer,
                          torch::Tensor column_index,
                          torch::Tensor degrees)
{
    const int NODE_NUM = input.size(0);
    const int dim = input.size(1);
    const int nnz = column_index.size(0);
    const int R_PTR_NUM = NODE_NUM + 1;

    input = input.reshape({NODE_NUM * dim, 1});

    // Create output tensor
    torch::Tensor output = torch::zeros(NODE_NUM * dim, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    
    int   A_num_rows = NODE_NUM;
    int   A_num_cols = NODE_NUM;
    int   A_nnz = nnz;
    int   B_num_rows = NODE_NUM;
    int   B_num_cols = dim;

    int   A_num_off       = R_PTR_NUM;
    int   ldb             = NODE_NUM;
    int   ldc             = NODE_NUM;
    int   B_size          = ldb * dim;
    int   C_size          = ldc * dim;


    float alpha           = 1.0f;
    float beta            = 0.0f;

    // Get GPU data pointers
    auto d_row_pointer = row_pointer.data_ptr();
    auto d_column_index = column_index.data_ptr();
    auto d_degrees = degrees.data_ptr();

    auto d_output = output.data_ptr();
    auto d_input = input.data_ptr();

   // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      d_row_pointer, d_column_index, d_degrees,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, dim, d_input,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, dim, d_output,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )



   // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )


  
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

  
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------


    return output.reshape({NODE_NUM, dim});

}
