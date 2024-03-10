#include <iostream>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename Dtype>
void outputMatrixChk(Dtype *A, int64_t ld, int64_t stride, int64_t num_batches, int64_t row, int64_t col){
  size_t size = num_batches * (row * col) * sizeof(Dtype);
  Dtype *tensor;
  tensor = (Dtype *)malloc(size);
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

void col_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
                 float *A, int64_t lda, int64_t stridea,
                 float * chk_v, int64_t ld_chk_v,
                 float * dcolchk, int64_t ld_dcolchk,
                 int64_t num_batches) {
    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    // cublasOperation_t transA = _cublasOpFromChar(transa);
    // cublasOperation_t transB = _cublasOpFromChar(transb);
    // cublasOperation_t transA = CUBLAS_OP_N;
    // cublasOperation_t transB = _cublasOpFromChar(transb); 

    float falpha = 1;
    float fbeta = 0;

    // std::cout << "m: " << m << ", n: " << n << std::endl;
    // std::cout << "alpha: " << falpha << ", beta: " << fbeta << std::endl;
    // std::cout << "ld_chk: " << ld_chk_v << std::endl;
    // std::cout << "ldA: " << lda << "; stridea: " << stridea << std::endl;
    // std::cout << "ld_colchk: " << ld_dcolchk << std::endl;
    // std::cout << "Before dcolchk_r: " << std::endl;
    // outputMatrixChk(dcolchk, ld_dcolchk, 2*n, num_batches, 2, n);
    cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, m,
            &falpha, chk_v, ld_chk_v, 0,
            A, lda, stridea, &fbeta,
            dcolchk, ld_dcolchk, 2*n,
            num_batches);
    // std::cout << "dcolchk_r: " << std::endl;
    // outputMatrixChk(dcolchk, ld_dcolchk, 2*n, num_batches, 2, n);
}


void row_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
                 float * A, int64_t lda, int64_t stridea,
                 float * chk_v, int64_t ld_chk_v,
                 float * drowchk, int64_t ld_drowchk,
                 int64_t num_batches) {
    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    // cublasOperation_t transA = _cublasOpFromChar(transa);
    // cublasOperation_t transB = CUBLAS_OP_T;
    
    float falpha = 1;
    float fbeta = 0;
    
    // std::cout << "m: " << m << ", n: " << n << std::endl;
    // std::cout << "alpha: " << falpha << ", beta: " << fbeta << std::endl;
    // std::cout << "ld_chk: " << ld_chk_v << std::endl;
    // std::cout << "ldA: " << lda << "; stridea: " << stridea << std::endl;
    // std::cout << "ld_rowchk: " << ld_drowchk << std::endl;
    // std::cout << "Before drowchk_r: " << std::endl;
    // outputMatrixChk(drowchk, ld_drowchk, 2*m, num_batches, m, 2);
    cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, m, 2, n,
            &falpha, A, lda, stridea,
            chk_v, ld_chk_v, 0, &fbeta,
            drowchk, ld_drowchk, 2*m,
            num_batches);
//     std::cout << "drowchk_r: " << std::endl;
//     outputMatrixChk(drowchk, ld_drowchk, 2*m, num_batches, m, 2);
}

void col_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
                 half *A, int64_t lda, int64_t stridea,
                 half * chk_v, int64_t ld_chk_v,
                 half * dcolchk, int64_t ld_dcolchk,
                 int64_t num_batches) {
    
    float falpha = 1;
    float fbeta = 0;

    cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, m,
            (void*)(&falpha), chk_v, CUDA_R_16F, ld_chk_v, 0,
            A, CUDA_R_16F, lda, stridea, (void*)(&fbeta),
            dcolchk, CUDA_R_16F, ld_dcolchk, 2*n,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // std::cout << "dcolchk_r: " << std::endl;
    // outputMatrixChk(dcolchk, ld_dcolchk, 2*n, num_batches, 2, n);
}

void row_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
                 half * A, int64_t lda, int64_t stridea,
                 half * chk_v, int64_t ld_chk_v,
                 half * drowchk, int64_t ld_drowchk,
                 int64_t num_batches) {
    float falpha = 1;
    float fbeta = 0;

    cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, m, 2, n,
            (void*)(&falpha), A, CUDA_R_16F, lda, stridea,
            chk_v, CUDA_R_16F, ld_chk_v, 0, (void*)(&fbeta),
            drowchk, CUDA_R_16F, ld_drowchk, 2*m,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // std::cout << "drowchk_r: " << std::endl;
    // outputMatrixChk(drowchk, ld_drowchk, 2*m, num_batches, m, 2);
}
