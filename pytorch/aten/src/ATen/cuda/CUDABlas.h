#pragma once
/*
  Provides a subset of CUDA BLAS functions as templates:

    gemm<Dtype>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
  ldc)

    gemv<Dtype>(transa, m, n, alpha, a, lda, x, incx, beta, y, incy)

    dot<Dtype>(n, x, incx, y, incy, result)

  where Dtype is double, float, at::Half or at::BFloat16 (ROCm, NOT for dot).
  The functions are available in at::cuda::blas namespace.
 */
#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
using namespace std::chrono;

#include <ATen/cuda/CUDAContext.h>
#include <ATen/OpMathType.h>
#include "opt_kernels.cu"

namespace at::cuda::blas {

// RAII guard that sets the CuBLAS pointer mode and restores it to
// its previous value when the guard is destroyed
class PointerModeGuard {
public:
  PointerModeGuard(cublasHandle_t handle, cublasPointerMode_t mode) :
      handle(handle) {
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &previous_mode));
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, mode));
  }

  ~PointerModeGuard() {
    cublasSetPointerMode(handle, previous_mode);
  }

private:
  cublasHandle_t handle;
  cublasPointerMode_t previous_mode;
};

/* LEVEL 3 BLAS FUNCTIONS */

#define CUDABLAS_GEMM_ARGTYPES(Dtype)                                                       \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,  \
      const Dtype *a, int64_t lda, const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta,\
      Dtype *c, int64_t ldc

template <typename Dtype>
inline void gemm(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemm: not implemented for ", typeid(Dtype).name());
}

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double));
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));
template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));
template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));

#if (!defined(USE_ROCM) && !defined(_MSC_VER)) || (defined(USE_ROCM) && ROCM_VERSION >= 50700)
enum GEMMAndBiasActivationEpilogue {
  None,
  RELU,
  GELU,
};

// NOTE: GELU activation is not supported prior to CUDA 11.4 and will
// do nothing if passed in that case.
/*
template <typename T>
void myGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,
            const Dtype *a, int64_t lda, const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta,
            Dtype *c, int64_t ldc
);

template <typename T, int64_t M, int64_t N, int64_t K>
void abftGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, 
              at::opmath_type<T> alpha, const T *a, int64_t lda, 
              const T *b, int64_t ldb, at::opmath_type<T> beta,\
              T *c, int64_t ldc,
              T *dA_colchk, int64_t ldda_colchk, T *dA_rowchk, int64_t ldda_rowchk,             
              T *dA_colchk_r, int64_t ldda_colchk_r, T *dA_rowchk_r, int64_t ldda_rowchk_r,      
              T *dB_colchk, int64_t lddb_colchk, T *dB_rowchk, int64_t lddb_rowchk,            
              T *dB_colchk_r, int64_t lddb_colchk_r, T *dB_rowchk_r, int64_t lddb_rowchk_r,      
              T *dC_colchk, int64_t lddc_colchk, T *dC_rowchk, int64_t lddc_rowchk,           
              T *dC_colchk_r, int64_t lddc_colchk_r, T *dC_rowchk_r, int64_t lddc_rowchk_r,   
              T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,                                      
              bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER);

template <typename Dtype>
void myGemmBias(
  bool transpose_mat1, bool transpose_mat2,
  int64_t m, int64_t n, int64_t k,
  at::opmath_type<Dtype> alpha_val,
  const Dtype* mat1_ptr,
  int64_t mat1_ld,
  const Dtype* mat2_ptr,
  int64_t mat2_ld,
  const Dtype* bias,
  Dtype* result_ptr,
  int64_t result_ld,
  GEMMAndBiasActivationEpilogue activation
);

template <typename Dtype, int64_t M, int64_t N, int64_t K>
void abftGemmBias(
    bool transpose_mat1, bool transpose_mat2,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<Dtype> alpha_val, const Dtype* mat1_ptr, int64_t mat1_ld,
    const Dtype* mat2_ptr, int64_t mat2_ld, const Dtype* bias,
    Dtype* result_ptr, int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation,
    Dtype *dA_colchk, int64_t ldda_colchk, Dtype *dA_rowchk, int64_t ldda_rowchk,              \
    Dtype *dA_colchk_r, int64_t ldda_colchk_r, Dtype *dA_rowchk_r, int64_t ldda_rowchk_r,      \
    Dtype *dB_colchk, int64_t lddb_colchk, Dtype *dB_rowchk, int64_t lddb_rowchk,            \
    Dtype *dB_colchk_r, int64_t lddb_colchk_r, Dtype *dB_rowchk_r, int64_t lddb_rowchk_r,      \
    Dtype *dC_colchk, int64_t lddc_colchk, Dtype *dC_rowchk, int64_t lddc_rowchk,           \
    Dtype *dC_colchk_r, int64_t lddc_colchk_r, Dtype *dC_rowchk_r, int64_t lddc_rowchk_r,   \
    Dtype *chk_v_a, Dtype *chk_v_b, int64_t ld_chk_v,                                      \
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER
);
*/
template <typename Dtype>
void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<Dtype> alpha_val,
    const Dtype* mat1_ptr,
    int64_t mat1_ld,
    const Dtype* mat2_ptr,
    int64_t mat2_ld,
    const Dtype* bias,
    Dtype* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation = GEMMAndBiasActivationEpilogue::None);

void int8_gemm(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    const int8_t* mat1_ptr,
    int64_t mat1_ld,
    const int8_t* mat2_ptr,
    int64_t mat2_ld,
    int32_t* result_ptr,
    int64_t result_ld);

void scaled_gemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const void* mat1_ptr,
    const void* mat1_scale_ptr,
    int64_t mat1_ld,
    ScalarType mat1_dtype,
    const void* mat2_ptr,
    const void* mat2_scale_ptr,
    int64_t mat2_ld,
    ScalarType mat2_dtype,
    const void* bias_ptr,
    ScalarType bias_dtype,
    void* result_ptr,
    const void* result_scale_ptr,
    int64_t result_ld,
    ScalarType result_dtype,
    void* amax_ptr,
    bool use_fast_accum);
#endif

#define CUDABLAS_BGEMM_ARGTYPES(Dtype)                                                        \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,    \
      const Dtype *a, int64_t lda, int64_t stridea,                                           \
      const Dtype *b, int64_t ldb, int64_t strideb,                                           \
      at::opmath_type<Dtype> beta, Dtype *c, int64_t ldc, int64_t stridec, int64_t num_batches

template <typename Dtype>
inline void bgemm(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::bgemm: not implemented for ", typeid(Dtype).name());
}

template <>
void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double));
template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>));
template <>
void bgemm<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>));
template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half));
template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));


#define CUDABLAS_ABFTGEMM_ARGTYPES(Dtype)                                                       \
    char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,  \
    Dtype *dA, int64_t ldda, int64_t stridea,                                           \
    Dtype *dB, int64_t lddb, int64_t strideb, at::opmath_type<Dtype> beta,             \
    Dtype *dC, int64_t lddc, int64_t stridec,                                                 \
    Dtype *dA_colchk, int64_t ldda_colchk, Dtype *dA_rowchk, int64_t ldda_rowchk,              \
    Dtype *dA_colchk_r, int64_t ldda_colchk_r, Dtype *dA_rowchk_r, int64_t ldda_rowchk_r,      \
    Dtype *dB_colchk, int64_t lddb_colchk, Dtype *dB_rowchk, int64_t lddb_rowchk,            \
    Dtype *dB_colchk_r, int64_t lddb_colchk_r, Dtype *dB_rowchk_r, int64_t lddb_rowchk_r,      \
    Dtype *dC_colchk, int64_t lddc_colchk, Dtype *dC_rowchk, int64_t lddc_rowchk,           \
    Dtype *dC_colchk_r, int64_t lddc_colchk_r, Dtype *dC_rowchk_r, int64_t lddc_rowchk_r,   \
    Dtype *chk_v_a, Dtype *chk_v_b, int64_t ld_chk_v,                                      \
    int64_t num_batches,                                                                    \
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER
/*
template <typename Dtype>
inline void abftgemm(CUDABLAS_ABFTGEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::abftgemm: not implemented for ", typeid(Dtype).name());
}
template <int64_t M, int64_t N, int64_t K>
void abftgemm<float>(CUDABLAS_ABFTGEMM_ARGTYPES(float));
*/
template <typename T, int64_t M, int64_t N, int64_t K>
void abftbgemm(CUDABLAS_ABFTGEMM_ARGTYPES(T));

template <typename T, int64_t M, int64_t N, int64_t K>
void abftbgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha,  
              T *dA, int64_t ldda, int64_t stridea,                                           
              T *dB, int64_t lddb, int64_t strideb, at::opmath_type<T> beta,             
              T *dC, int64_t lddc, int64_t stridec,                                                 
              T *dA_colchk, int64_t ldda_colchk, T *dA_rowchk, int64_t ldda_rowchk,              
              T *dA_colchk_r, int64_t ldda_colchk_r, T *dA_rowchk_r, int64_t ldda_rowchk_r,      
              T *dB_colchk, int64_t lddb_colchk, T *dB_rowchk, int64_t lddb_rowchk,            
              T *dB_colchk_r, int64_t lddb_colchk_r, T *dB_rowchk_r, int64_t lddb_rowchk_r,      
              T *dC_colchk, int64_t lddc_colchk, T *dC_rowchk, int64_t lddc_rowchk,           
              T *dC_colchk_r, int64_t lddc_colchk_r, T *dC_rowchk_r, int64_t lddc_rowchk_r,   
              T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,                                      
              int64_t num_batches,
              bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER ){
  std::cout << "Using abftbgemm-T function." << std::endl;
  // See Note [Writing Nondeterministic Operations]
  // std::cout << "globalContext. \n";
  globalContext().alertCuBLASConfigNotDeterministic();
  // std::cout << "handle. \n";
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // std::cout << "transA & transB. \n";
  cublasOperation_t transA = CUBLAS_OP_N;
  cublasOperation_t transB = CUBLAS_OP_N;
  switch (transa) {
    case 'n':
    case 'N':
      transA = CUBLAS_OP_N;
    case 't':
    case 'T':
      transA = CUBLAS_OP_T;
    case 'c':
    case 'C':
      transA = CUBLAS_OP_C;
  }
  switch (transb) {
    case 'n':
    case 'N':
      transB = CUBLAS_OP_N;
    case 't':
    case 'T':
      transB = CUBLAS_OP_T;
    case 'c':
    case 'C':
      transB = CUBLAS_OP_C;
  }
  // cublasOperation_t transA = _cublasOpFromChar(transa);
  // cublasOperation_t transB = _cublasOpFromChar(transb);

  // std::cout << "alpha & beta. \n";
  float falpha = at::opmath_type<T>(1);
  float fbeta = at::opmath_type<T>(0);
  // std::cout << "alpha: " << falpha << "; beta: " << fbeta << std::endl;
  std::cout << "transa: " << transa << "; transb: " << transb << std::endl;
  // get the col_chk and row_chk of A and B
  // std::cout << "  Get dA_chk: " << std::endl;

  // CUDA Stream
  cudaStream_t stream_main, stream_colchk, stream_rowchk;
  cudaStreamCreate(&stream_main);
  cudaStreamCreate(&stream_colchk);
  cudaStreamCreate(&stream_rowchk);
  cublasSetStream(handle, stream_main);

  cudaEvent_t main_compute_done;
  cudaEventCreate(&main_compute_done);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t, t1, t_Achk, t_Bchk;

  if (DEBUG) cudaEventRecord(start, stream_colchk);
  if(transA == CUBLAS_OP_N){
    // encode_col_lancher<T, M, K, 4>(num_batches,
    //               dA, ldda, stridea, 
    //               dA_colchk, ldda_colchk, (2*k), stream_colchk);
    encode_col_v5<T, M, K, 4><<<num_batches, dim3(M*4, 1), (M+1)*K*sizeof(T), stream_colchk>>>(num_batches,
                  dA, ldda, stridea, 
                  dA_colchk, ldda_colchk, (2*k));
  }
  else{
    // cublasSgemmStridedBatched(
    //   handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, m,
    //   &falpha, dA, ldda, stridea,
    //   chk_v_a, ld_chk_v, 0, &fbeta,
    //   dA_rowchk, ldda_rowchk, (2*k),
    //   num_batches);
    // std::cout << "  Output dA_rowchk: " << std::endl;
    // outputMatrixChk(dA_rowchk, ldda_rowchk, (2*k), num_batches, k, 2);
  }
  if (DEBUG) {
    cudaEventRecord(stop, stream_colchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_Achk, start, stop);
    // printf("dA_chk_gemm: %f (%f)(%f)\n", t, (double)num_batches*m*2*k*2/t/1e6, (double)num_batches*(2*k+2*m+k*m)/t/1e6);
  }
  
  //std::cout << "  Get dB_chk: " << std::endl;
  if (DEBUG) cudaEventRecord(start, stream_rowchk);
  if (transB == CUBLAS_OP_N){
    // encode_row_lancher<T, K, N>(num_batches,
    //               dB, lddb, strideb, 
    //               dB_rowchk, lddb_rowchk, (2*k), stream_rowchk);
    encode_row_v5<T, K, N><<<num_batches, dim3(K*2, 1, 1), 0, stream_rowchk>>>(num_batches,
                  dB, lddb, strideb, 
                  dB_rowchk, lddb_rowchk, (2*k));
  }
  else{
    // cublasSgemmStridedBatched(
    //   handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, n,
    //   &falpha, chk_v_b, ld_chk_v, 0,
    //   dB, lddb, strideb, &fbeta,
    //   dB_colchk, lddb_colchk, (2*k),
    //   num_batches);
    // std::cout << " Output dB_colchk: " << std::endl;
    // outputMatrixChk(dB_colchk, lddb_colchk, (2*k), num_batches, 2, k);
  }
  if (DEBUG) {
    cudaEventRecord(stop, stream_rowchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_Bchk, start, stop);
    t_Bchk /= 1.0;
    // printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*2*n*k*2/t1/1e6, (double)num_batches*(2*k+k*n+2*n)/t1/1e6);
  }

  falpha = alpha;
  fbeta = beta;

  // number of row and col of B stored in memory(no trans operation)
  int64_t mem_row = 0;
	int64_t mem_col = 0;

  // --begin-- //
  // calculate check-sum
  std::cout << "-----Begin.------" << std::endl;
  if (DEBUG) cudaEventRecord(start, stream_main);
  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  if constexpr (std::is_same<T, float>::value) {
    cublasSgemmStridedBatched(
        handle, transA, transB, m, n, k,
        &falpha, dA, ldda, stridea,
        dB, lddb, strideb, &fbeta,
        dC, lddc, stridec,
        num_batches);
  } else if constexpr(std::is_same<T, at::Half>::value) {
    cublasGemmStridedBatchedEx(
    handle, transA, transB, m, n, k,
    &falpha, dA, CUDA_R_16F, ldda, stridea,
    dB, CUDA_R_16F, lddb, strideb, &fbeta,
    dC, CUDA_R_16F, lddc, stridec,
    num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaStreamSynchronize(stream_main);
  // std::cout << "Output dC: " << std::endl;
  // outputMatrix(dC, lddc, stridec, num_batches, m, n);
  
  if (DEBUG) {
    cudaEventRecord(stop, stream_main);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);
    printf("  gemm: %f (%f)(%f)\n", t, (double)num_batches*m*n*k*2/t/1e6, (double)num_batches*(m*k+k*n+m*n)/t/1e6);
    printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)num_batches*m*2*k*2/t_Achk/1e6, (double)num_batches*(2*k+2*m+k*m)/t_Achk/1e6);
    printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t_Bchk, t_Bchk/t, (double)num_batches*2*n*k*2/t_Bchk/1e6, (double)num_batches*(2*k+k*n+2*n)/t_Bchk/1e6);
  }

  if (DEBUG) cudaEventRecord(start, stream_colchk);
  if(COL_FT){
    //std::cout << "  COL_FT" << std::endl;
    if (transA == CUBLAS_OP_N) {
       if (DEBUG) std::cout << "dA_colchk * dB = dC_colchk" << std::endl;;
        // K*4 must be greater then 2 * N
        // update_col_lancher<T, K, N, 4>(num_batches,
        //             dA_colchk, ldda_colchk, k*2, 
        //             dB, lddb, strideb, 
        //             dC_colchk, lddc_colchk, n*2, stream_colchk);
        update_col_v5<T, K, N, 4><<<num_batches, dim3(K*4, 1, 1), ((K+1)*N+2*K) * sizeof(T), stream_colchk>>>(num_batches,
                    dA_colchk, ldda_colchk, k*2, 
                    dB, lddb, strideb, 
                    dC_colchk, lddc_colchk, n*2);
    }
    else{
      if (DEBUG) std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;
    }
    // std::cout << "Output dC_colchk: " << std::endl;
    // outputMatrixChk(dC_colchk, ldda_colchk, n*2, num_batches, 2, n);
  }
  if (DEBUG) {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*2*n*k*2/t1/1e6, (double)num_batches*(2*k+k*n+2*n)/t1/1e6);
  }

  if (DEBUG) cudaEventRecord(start, stream_rowchk);
  if (ROW_FT) {
      //std::cout << "  ROW_FT" << std::endl;
      if (transB == CUBLAS_OP_N) {
        if (DEBUG) std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
        //we can further work on this to support trans A.
        // update_row_lancher<T, M, K>(num_batches,
        //             dA, ldda, stridea, 
        //             dB_rowchk, lddb_rowchk, k*2, 
        //             dC_rowchk, lddc_rowchk, m*2, stream_rowchk);
        update_row_v5<T, M, K><<<num_batches, dim3(M*2, 1, 1), (2*K) * sizeof(T), stream_rowchk>>>(num_batches,
                    dA, ldda, stridea, 
                    dB_rowchk, lddb_rowchk, k*2, 
                    dC_rowchk, lddc_rowchk, m*2);
      }
      else{
        if (DEBUG) std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
        
      }
      // std::cout << "Output dC_rowchk: " << std::endl;
      // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
  }
  if (DEBUG) {
    cudaEventRecord(stop, stream_rowchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t1, start, stop);
    printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*m*2*k*2/t1/1e6, (double)num_batches*(m*k+k*2+m*2)/t1/1e6);
  }

  // --- check check-sum of C---//
  if (DEBUG) std::cout << "------Check check-sum-------" << std::endl;
  if (DEBUG) cudaEventRecord(start, stream_colchk);
  if (COL_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-col\n");
    // encode_col_lancher<T, M, N, 4>(num_batches,
    //                 dC, lddc, stridec, 
    //                 dC_colchk_r, lddc_colchk_r, (2*n),stream_colchk);
    encode_col_v5<T, M, N, 4><<<num_batches, dim3(M*4, 1), (M+1)*N*sizeof(T), stream_colchk>>>(num_batches,
                   dC, lddc, stridec, 
                    dC_colchk_r, lddc_colchk_r, (2*n));

    T E = 1e-2;
    // detect_correct_col_lancher(dC, lddc, E, stridec,
    //                                         dC_colchk,      lddc_colchk,    (2*n),
    //                                         dC_colchk_r,    lddc_colchk_r,  (2*n),
    //                                         num_batches, n, stream_colchk);
    detect_correct_col<T><<<dim3(num_batches), dim3(n), 0, stream_colchk>>>(dC, lddc, E, stridec,
                                            dC_colchk,      lddc_colchk,    (2*n),
                                            dC_colchk_r,    lddc_colchk_r,  (2*n));
  }

  if (DEBUG) {
    cudaEventRecord(stop, stream_colchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t1, start, stop);
    printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*2*n*m*2/t1/1e6, (double)num_batches*(m*n+2*m+2*n)/t1/1e6);
  }

  if (DEBUG) cudaEventRecord(start, stream_rowchk);
  if (ROW_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-row\n");
    // encode_row_lancher<T, M, N>(num_batches,
    //                 dC, lddc, stridec,
    //                 dC_rowchk_r, lddc_rowchk_r, (2*m), stream_rowchk);
    encode_row_v5<T, M, N><<<num_batches, dim3(M*2, 1, 1), 0, stream_rowchk>>>(num_batches,
                    dC, lddc, stridec,
                    dC_rowchk_r, lddc_rowchk_r, (2*m));
    T E = 1e-2;
    // detect_correct_row_lancher(dC, lddc, E, stridec,
    //                                       dC_rowchk, lddc_rowchk,     (2*m),
    //                                       dC_rowchk_r, lddc_rowchk_r, (2*m), 
    //                                       num_batches, m, stream_rowchk);
    detect_correct_row<T><<<dim3(num_batches), dim3(m), 0, stream_rowchk>>>(dC, lddc, E, stridec,
                                          dC_rowchk, lddc_rowchk,     (2*m),
                                          dC_rowchk_r, lddc_rowchk_r, (2*m));
  }

  if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*m*2*n*2/t1/1e6, (double)num_batches*(m*n+2*n+2*m)/t1/1e6);
  }
}

#define CUDABLAS_MYBGEMM_ARGTYPES(Dtype)                                                       \
      char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha, \
      const Dtype *dA, int64_t ldda, int64_t stridea,                                          \
      const Dtype *dB, int64_t lddb, int64_t strideb, at::opmath_type<Dtype> beta,             \
      Dtype *dC, int64_t lddc, int64_t stridec,                                                \
      int64_t num_batches  
/*                                                                                                                          
template <typename Dtype>
inline void mybgemm(CUDABLAS_MYBGEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::mybgemm: not implemented for ", typeid(Dtype).name());
}
template <>
void mybgemm<float>(CUDABLAS_MYBGEMM_ARGTYPES(float));
*/
template <typename T>
void mybgemm(CUDABLAS_MYBGEMM_ARGTYPES(T));

template <typename T>
void mybgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha, \
              const T *dA, int64_t ldda, int64_t stridea,                                          \
              const T *dB, int64_t lddb, int64_t strideb, at::opmath_type<T> beta,             \
              T *dC, int64_t lddc, int64_t stridec,                                                \
              int64_t num_batches) {
  std::cout << "Using mybgemm-T function." << std::endl;

  T *dA_ = const_cast<T*>(dA);
  T *dB_ = const_cast<T*>(dB);
  // std::cout << "C: " << std::endl;
  // outputMatrix(dC, lddc, stridec, num_batches, m, n);
  if(beta == at::opmath_type<T>(0)){
    cudaMemset(dC, 0, (num_batches * m * n * sizeof(T)));
  }
  
  // std::cout << "_A:" << std::endl;
  // outputMatrix(dA_, ldda, stridea, num_batches, k, n);
  // std::cout << "stridea: " << stridea  << "; lda: " << ldda << std::endl;
  // std::cout << "_B: " << std::endl;
  // outputMatrix(dB_, lddb, strideb, num_batches, m, k);
  // std::cout << "strideb: " << strideb << "; ldb: " << lddb << std::endl;
  std::cout << "m: " << m << "; n: " << n << "; k: " << k << std::endl;
  std::cout << "num_batches: " << num_batches << std::endl;

  // std::cout << "leading dimension." << std::endl;
  

  int64_t ldda_colchk = 2;
  int64_t ldda_colchk_r = 2;
  int64_t ldda_rowchk = k;
  int64_t ldda_rowchk_r = k;

  int64_t lddb_rowchk = k;
  int64_t lddb_rowchk_r = k;
  int64_t lddb_colchk = 2;
  int64_t lddb_colchk_r = 2;

  int64_t lddc_colchk = 2;
  int64_t lddc_colchk_r = 2;
  int64_t lddc_rowchk = m;
  int64_t lddc_rowchk_r = m;
  int64_t ld_chk_v = 2;

  //std::cout << "alloc chk vectors" << std::endl;

  T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
  T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
  T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
  T *chk_v_a;
  T *chk_v_b;

  size_t size = (2*num_batches) * k * sizeof(T);
  cudaMalloc((void**)&dA_colchk, size);
  cudaMemset(dA_colchk, 0, size);
  cudaMalloc((void**)&dA_colchk_r, size);
  cudaMemset(dA_colchk_r, 0, size);

  cudaMalloc((void**)&dA_rowchk, size);
  cudaMemset(dA_rowchk, 0, size);
  cudaMalloc((void**)&dA_rowchk_r, size);
  cudaMemset(dA_rowchk_r, 0, size);
  //std::cout << "  finish dA." << std::endl;
  
  cudaMalloc((void**)&dB_colchk, size);
  cudaMemset(dB_colchk, 0, size);
  cudaMalloc((void**)&dB_colchk_r, size);
  cudaMemset(dB_colchk_r, 0, size);
  
  cudaMalloc((void**)&dB_rowchk, size);
  cudaMemset(dB_rowchk, 0, size);
  cudaMalloc((void**)&dB_rowchk_r, size);
  cudaMemset(dB_rowchk_r, 0, size);
  //std::cout << "  finish dB." << std::endl;

  size = (2*num_batches) * n * sizeof(T);
  cudaMalloc((void**)&dC_colchk, size);
  cudaMemset(dC_colchk, 0, size);
  cudaMalloc((void**)&dC_colchk_r, size);
  cudaMemset(dC_colchk_r, 0, size);
  
  size = (2*num_batches) * m * sizeof(T);
  cudaMalloc((void**)&dC_rowchk, size);
  cudaMemset(dC_rowchk, 0, size);
  cudaMalloc((void**)&dC_rowchk_r, size);
  cudaMemset(dC_rowchk_r, 0, size);
  //std::cout << "  finish dC." << std::endl;

  //int64_t len = (m > n)? m : n;
  int64_t len = m;
  size = 2 * len * sizeof(T);
  cudaMalloc((void**)&chk_v_a, size);
  // std::cout << "  assign values to chk_v_a." << std::endl;
  T *h_matrix;
  h_matrix = (T *)malloc(size);
  int idx = 0;
  for(int i = 0; i < len; i++){
    idx = i*ld_chk_v;
    h_matrix[idx] = T(1);
    h_matrix[idx+1] = T(i+1);
  }
  cudaMemcpy(chk_v_a, h_matrix, size, cudaMemcpyHostToDevice);
  // outputMatrixChk(chk_v_a, ld_chk_v, 0, 1, 2, len);
  free(h_matrix);

  len = n;
  size = 2 * len * sizeof(T);
  cudaMalloc((void**)&chk_v_b, size);
  // std::cout << "  assign values to chk_v_b." << std::endl;
  h_matrix = (T *)malloc(size);
  idx = 0;
  for(int i = 0; i < len; i++){
    idx = i*ld_chk_v;
    h_matrix[idx] = T(1);
    h_matrix[idx+1] = T(i+1);
  }
  cudaMemcpy(chk_v_b, h_matrix, size, cudaMemcpyHostToDevice);
  // outputMatrixChk(chk_v_b, ld_chk_v, 0, 1, 2, len);
  free(h_matrix);
  //std::cout << "  finish chk_v." << std::endl;

  bool COL_FT = true;
  bool ROW_FT = true;
  bool DEBUG = true;
  bool CHECK_BEFORE = true;
  bool CHECK_AFTER = true;

  // std::cout << "Calling abftgemm-T function." << std::endl;

  // for (int i = 0; i <10; i++) {
  auto start = high_resolution_clock::now();
  // const int64_t M = 72;
  // const int64_t N = 72;
  // const int64_t K = 64;
  abftbgemm<T, 72, 72, 64>(transa, transb, m, n, k,
      alpha, dA_, ldda, stridea,
      dB_, lddb, strideb, beta,
      dC, lddc, stridec,
      dA_colchk, ldda_colchk,
      dA_rowchk, ldda_rowchk,
      dA_colchk_r, ldda_colchk_r,
      dA_rowchk_r, ldda_rowchk_r,
      dB_colchk, lddb_colchk,
      dB_rowchk, lddb_rowchk,
      dB_colchk_r, lddb_colchk_r,
      dB_rowchk_r, lddb_rowchk_r,
      dC_colchk, lddc_colchk,
      dC_rowchk, lddc_rowchk,
      dC_colchk_r, lddc_colchk_r,
      dC_rowchk_r, lddc_rowchk_r,
      chk_v_a, chk_v_b, ld_chk_v,
      num_batches,
      COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER);
  cudaDeviceSynchronize();
  auto stop = high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(stop - start);
  std::cout << "abftbgemm: " << duration.count() / 1000.0 << std::endl;
  // }

  cudaFree(dA_colchk);
  cudaFree(dA_rowchk);
  cudaFree(dA_colchk_r);
  cudaFree(dA_rowchk_r);
  cudaFree(dB_colchk);
  cudaFree(dB_rowchk);
  cudaFree(dB_colchk_r);
  cudaFree(dB_rowchk_r);
  cudaFree(dC_colchk);
  cudaFree(dC_rowchk);
  cudaFree(dC_colchk_r);
  cudaFree(dC_rowchk_r);
  cudaFree(chk_v_a);
  cudaFree(chk_v_b);
}

#if defined(USE_ROCM) && ROCM_VERSION <= 50500
// ROCm 5.6 hipblas matches the const Dtype *A API, but prior hipblas does not.
#define CUDABLAS_TRSM_ARGTYPES(Dtype)                                  \
  hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, \
      hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n,    \
      const Dtype *alpha,       Dtype *A, int lda, Dtype *B, int ldb
#else
#define CUDABLAS_TRSM_ARGTYPES(Dtype)                                  \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, const Dtype *A, int lda, Dtype *B, int ldb
#endif

template <typename Dtype>
inline void trsm(CUDABLAS_TRSM_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(false, "at::cuda::blas::trsm: not implemented for ", typeid(Dtype).name());
}

template <>
TORCH_CUDA_CU_API void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float));
template <>
TORCH_CUDA_CU_API void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double));
template <>
TORCH_CUDA_CU_API void trsm<c10::complex<float>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<float>));
template <>
TORCH_CUDA_CU_API void trsm<c10::complex<double>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<double>));

#define CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)                          \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, Dtype *A[], int lda, Dtype *B[], int ldb,    \
      int batchCount

template <typename Dtype>
inline void trsmBatched(CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::blas::trsmBatched: not implemented for ",
      typeid(Dtype).name());
}

template <>
TORCH_CUDA_CU_API void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float));
template <>
TORCH_CUDA_CU_API void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double));
template <>
TORCH_CUDA_CU_API void trsmBatched<c10::complex<float>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>));
template <>
TORCH_CUDA_CU_API void trsmBatched<c10::complex<double>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>));

/* LEVEL 2 BLAS FUNCTIONS */

#define CUDABLAS_GEMV_ARGTYPES(Dtype)                                         \
  char trans, int64_t m, int64_t n, Dtype alpha, const Dtype *a, int64_t lda, \
      const Dtype *x, int64_t incx, Dtype beta, Dtype *y, int64_t incy

template <typename Dtype>
inline void gemv(CUDABLAS_GEMV_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemv: not implemented for ", typeid(Dtype).name());
}

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double));
template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float));
template <>
void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>));
template <>
void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>));
template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half));
template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16));

/* LEVEL 1 BLAS FUNCTIONS */

#define CUDABLAS_DOT_ARGTYPES(Dtype)                                      \
  cublasHandle_t handle, int n, const Dtype *x, int incx, const Dtype *y, \
      int incy, Dtype *result

template <typename Dtype>
inline void dot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::dot: not implemented for ", typeid(Dtype).name());
}

template <>
void dot<double>(CUDABLAS_DOT_ARGTYPES(double));
template <>
void dot<float>(CUDABLAS_DOT_ARGTYPES(float));
template <>
void dot<at::Half>(CUDABLAS_DOT_ARGTYPES(at::Half));
template <>
void dot<at::BFloat16>(CUDABLAS_DOT_ARGTYPES(at::BFloat16));
template <>
void dot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>));
template <>
void dot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>));

template <typename Dtype>
inline void vdot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::vdot: not implemented for ", typeid(Dtype).name());
}

template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>));
template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>));

#define CUDABLAS_GETRS_ARGTYPES(Dtype)  \
  cublasHandle_t handle, cublasOperation_t trans, \
  int n, int nrhs, Dtype** dA_array, int lda, int* ipiv_array, \
  Dtype** dB_array, int ldb, int* info_array, int batchsize

template<class Dtype>
void getrsBatched(CUDABLAS_GETRS_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(false, "at::cuda::blas::getrsBatched: not implemented for ",
    typeid(Dtype).name());
}
template<>
TORCH_CUDA_CU_API void getrsBatched<float>(CUDABLAS_GETRS_ARGTYPES(float));
template<>
TORCH_CUDA_CU_API void getrsBatched<double>(CUDABLAS_GETRS_ARGTYPES(double));
template<>
TORCH_CUDA_CU_API void getrsBatched<c10::complex<float>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<float>));
template<>
TORCH_CUDA_CU_API void getrsBatched<c10::complex<double>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<double>));

#define CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)                   \
  cublasHandle_t handle, int m, int n, Dtype **A_array, int lda, \
      Dtype **tau_array, int *info, int batchsize

template <class Dtype>
void geqrfBatched(CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::blas::geqrfBatched: not implemented for ",
      typeid(Dtype).name());
}
template <>
TORCH_CUDA_CU_API void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float));
template <>
TORCH_CUDA_CU_API void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double));
template <>
TORCH_CUDA_CU_API void geqrfBatched<c10::complex<double>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<double>));
template <>
TORCH_CUDA_CU_API void geqrfBatched<c10::complex<float>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<float>));

#define CUDABLAS_GETRF_ARGTYPES(Dtype)  \
  int n, Dtype** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize

template<class Dtype>
void getrfBatched(CUDABLAS_GETRF_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::getrfBatched: not implemented for ", typeid(Dtype).name());
}
template<>
TORCH_CUDA_CU_API void getrfBatched<float>(CUDABLAS_GETRF_ARGTYPES(float));
template<>
TORCH_CUDA_CU_API void getrfBatched<double>(CUDABLAS_GETRF_ARGTYPES(double));
template<>
TORCH_CUDA_CU_API void getrfBatched<c10::complex<double>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<double>));
template<>
TORCH_CUDA_CU_API void getrfBatched<c10::complex<float>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<float>));

#define CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)  \
  cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, Dtype** dA_array, int ldda, Dtype** dC_array, int lddc, int* info, int *devInfoArray, int batchSize

template <class Dtype>
void gelsBatched(CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(false, "at::cuda::blas::gelsBatched: not implemented for ", typeid(Dtype).name());
}

template<>
TORCH_CUDA_CU_API void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double));
template<>
TORCH_CUDA_CU_API void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float));
template<>
TORCH_CUDA_CU_API void gelsBatched<c10::complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<double>));
template<>
TORCH_CUDA_CU_API void gelsBatched<c10::complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<float>));

} // namespace at::cuda::blas
