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
// #include <string>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <stdio.h>
// #include <stdlib.h>

// #include <chrono>
// using namespace std::chrono;

#include <ATen/cuda/CUDAContext.h>
#include <ATen/OpMathType.h>
// #include "opt_kernels.cu"

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

template <typename T>
void myGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha,
            const T *a, int64_t lda, const T *b, int64_t ldb, at::opmath_type<T> beta,
            T *c, int64_t ldc);

template <typename T>
void abftGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, 
              at::opmath_type<T> alpha, T *a, int64_t lda, 
              T *b, int64_t ldb, at::opmath_type<T> beta,
              T *c, int64_t ldc,
              /*T *dA_colchk, int64_t ldda_colchk, T *dA_rowchk, int64_t ldda_rowchk,             
              T *dA_colchk_r, int64_t ldda_colchk_r, T *dA_rowchk_r, int64_t ldda_rowchk_r,      
              T *dB_colchk, int64_t lddb_colchk, T *dB_rowchk, int64_t lddb_rowchk,            
              T *dB_colchk_r, int64_t lddb_colchk_r, T *dB_rowchk_r, int64_t lddb_rowchk_r,      
              T *dC_colchk, int64_t lddc_colchk, T *dC_rowchk, int64_t lddc_rowchk,           
              T *dC_colchk_r, int64_t lddc_colchk_r, T *dC_rowchk_r, int64_t lddc_rowchk_r,*/   
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
    at::opmath_type<Dtype> alpha_val, Dtype* mat1_ptr, int64_t mat1_ld,
    Dtype* mat2_ptr, int64_t mat2_ld, Dtype* bias,
    Dtype* result_ptr, int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation,
    /*Dtype *dA_colchk, int64_t ldda_colchk, Dtype *dA_rowchk, int64_t ldda_rowchk,              
    Dtype *dA_colchk_r, int64_t ldda_colchk_r, Dtype *dA_rowchk_r, int64_t ldda_rowchk_r,      
    Dtype *dB_colchk, int64_t lddb_colchk, Dtype *dB_rowchk, int64_t lddb_rowchk,            
    Dtype *dB_colchk_r, int64_t lddb_colchk_r, Dtype *dB_rowchk_r, int64_t lddb_rowchk_r,      
    Dtype *dC_colchk, int64_t lddc_colchk, Dtype *dC_rowchk, int64_t lddc_rowchk,           
    Dtype *dC_colchk_r, int64_t lddc_colchk_r, Dtype *dC_rowchk_r, int64_t lddc_rowchk_r, */   
    Dtype *chk_v_a, Dtype *chk_v_b, int64_t ld_chk_v,   
    Dtype *dBias_colchk, Dtype *dBias_rowchk, Dtype *dBias_colchk_r, Dtype *dBias_rowchk_r,          
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER
);

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
template<typename T>
void mybgemm(CUDABLAS_MYBGEMM_ARGTYPES(T));

// #define CUDABLAS_ABFTGEMM_ARGTYPES(Dtype)                                                       \
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
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, bool ifPassChk

#define CUDABLAS_ABFTGEMM_ARGTYPES(Dtype)                                                       \
    char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,  \
    Dtype *dA, int64_t ldda, int64_t stridea,                                           \
    Dtype *dB, int64_t lddb, int64_t strideb, at::opmath_type<Dtype> beta,             \
    Dtype *dC, int64_t lddc, int64_t stridec,                                                 \
    int64_t num_batches,                                                                    \
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, bool ifPassChk

template <typename T, int64_t M, int64_t N, int64_t K>
void abftbgemm(CUDABLAS_ABFTGEMM_ARGTYPES(T));

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
