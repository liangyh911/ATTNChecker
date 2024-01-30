#include <iostream>
#include <cstdint>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDADataType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Export.h>
#include <c10/util/irange.h>

namespace {
    static cublasOperation_t _cublasOpFromChar(char op) {
    switch (op) {
        case 'n':
        case 'N':
        return CUBLAS_OP_N;
        
        case 't':
        case 'T':
        return CUBLAS_OP_T;
        
        case 'c':
        case 'C':
        return CUBLAS_OP_C;
    }
    AT_ERROR(
        "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
    }
}

void outputMatrixChk(at::Half *A, int64_t ld, int64_t stride, int64_t num_batches, int64_t row, int64_t col){
  size_t size = num_batches * (row * col) * sizeof(at::Half);
  at::Half *tensor;
  tensor = (at::Half *)malloc(size);
  cudaMemcpy(tensor, A, size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < num_batches; i++){
    std::cout << "[ " << std::endl;
    for(int m = 0; m < row; m++){
      for(int n = 0; n < col; n++){
        std::cout << tensor[i*stride + n*ld + m] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << " ]" << std::endl;
  }
  free(tensor);
}

void col_chk_enc(char transa, char transb, int64_t m, int64_t n,
                 at::Half *A, int64_t lda, int64_t stridea,
                 at::Half * chk_v, int64_t ld_chk_v,
                 at::Half * dcolchk, int64_t ld_dcolchk,
                 int64_t num_batches) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasOperation_t transA = _cublasOpFromChar(transa);
    cublasOperation_t transB = _cublasOpFromChar(transb);
    float falpha = at::opmath_type<at::Half>(1);
    float fbeta = at::opmath_type<at::Half>(0);

    cublasGemmStridedBatchedEx(
            handle, transA, transB, 2, n, m,
            (void*)(&falpha), chk_v, CUDA_R_16F, ld_chk_v, 0,
            A, CUDA_R_16F, lda, stridea, (void*)(&fbeta),
            dcolchk, CUDA_R_16F, ld_dcolchk, 2*n,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    outputMatrixChk(dcolchk, ld_dcolchk, 2*n, num_batches, 2, n);
}

void row_chk_enc(char transa, char transb, int64_t m, int64_t n,
                 at::Half * A, int64_t lda, int64_t stridea,
                 at::Half * chk_v, int64_t ld_chk_v,
                 at::Half * drowchk, int64_t ld_drowchk,
                 int64_t num_batches) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasOperation_t transA = _cublasOpFromChar(transa);
    cublasOperation_t transB = CUBLAS_OP_T;
    float falpha = at::opmath_type<at::Half>(1);
    float fbeta = at::opmath_type<at::Half>(0);

    cublasGemmStridedBatchedEx(
            handle, transA, transB, m, 2, n,
            (void*)(&falpha), A, CUDA_R_16F, lda, stridea,
            chk_v, CUDA_R_16F, ld_chk_v, 0, (void*)(&fbeta),
            drowchk, CUDA_R_16F, ld_drowchk, 2*m,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    outputMatrixChk(drowchk, ld_drowchk, 2*m, num_batches, m, 2);
}
