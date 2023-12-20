#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>

// #define FP16MM

#include <torch/extension.h>


const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	int a=1;

    for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = (float)rand()/(float)(RAND_MAX/a);
	}
}

torch::Tensor cublas_gemm(torch::Tensor A,
                          torch::Tensor B)
{
    // using namespace std;
    
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    A = A.reshape({m * k, 1});
    B = B.reshape({k * n, 1});


    // Create output tensor
    torch::Tensor output = torch::zeros(m * n, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Get GPU data pointers

    auto d_A = (float*) A.data_ptr();
    auto d_B = (float*) B.data_ptr();
    auto d_output = (float*) output.data_ptr();


    cublasStatus_t stat;
    cublasHandle_t handle;

    checkCublas(cublasCreate(&handle));


    int lda, ldb, ldc;
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;


	  lda = m;
	  ldb = k;
	  ldc = m;

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_output, ldc); 

    // cudaEventRecord(stop,0);
    // cudaEventSynchronize(stop);
    if(stat != CUBLAS_STATUS_SUCCESS){
        // cerr << "cublasSgemmBatched failed" << std::endl;
        exit(1);
    }
  
    assert(!cudaGetLastError());
  
    // //Free GPU memory
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_output);

    return output.reshape({m, n});

}