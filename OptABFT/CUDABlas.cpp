/*
  Provides the implementations of CUDA BLAS function templates.
 */
#undef max
#undef min
// #include "./abft_checker.h"

#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDADataType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Export.h>
#include <c10/util/irange.h>

// #include <chrono>
// using namespace std::chrono;

// #include "opt_kernels.cu"

#ifdef USE_ROCM
// until hipblas has an API to accept flags, we must use rocblas here
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 + ROCBLAS_VERSION_MINOR)
#define USE_GEMM_FLAGS_FP16_ALT_IMPL (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242)
// needed to work around calling rocblas API instead of hipblas API
static rocblas_operation hipOperationToRocOperation(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return rocblas_operation_none;
    case HIPBLAS_OP_T:
        return rocblas_operation_transpose;
    case HIPBLAS_OP_C:
        return rocblas_operation_conjugate_transpose;
    }
    AT_ERROR("HIPBLAS_STATUS_INVALID_ENUM");
}
static hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    AT_ERROR("HIPBLAS_STATUS_INVALID_ENUM");
}
// hipblas does not have hipblasSetMathMode
#define hipblasSetMathMode(handle, flags) HIPBLAS_STATUS_SUCCESS
// until we use hiblas v2
// hipify correctly maps things like CUDA_R_16F to HIP_R_16F,
// however hipblas v1 is still using its custom type
#ifndef HIPBLAS_V2
#define HIP_R_16F  HIPBLAS_R_16F
#define HIP_R_32F  HIPBLAS_R_32F
#define HIP_R_64F  HIPBLAS_R_64F
#define HIP_C_16F  HIPBLAS_C_16F
#define HIP_C_32F  HIPBLAS_C_32F
#define HIP_C_64F  HIPBLAS_C_64F
#define HIP_R_8I   HIPBLAS_R_8I
#define HIP_R_8U   HIPBLAS_R_8U
#define HIP_R_32I  HIPBLAS_R_32I
#define HIP_R_32U  HIPBLAS_R_32U
#define HIP_C_8I   HIPBLAS_C_8I
#define HIP_C_8U   HIPBLAS_C_8U
#define HIP_C_32I  HIPBLAS_C_32I
#define HIP_C_32U  HIPBLAS_C_32U
#define HIP_R_16BF HIPBLAS_R_16B
#define HIP_C_16BF HIPBLAS_C_16B
#endif
#endif

#define CUDABLAS_POSINT_CHECK(FD, X)         \
  TORCH_CHECK(                               \
      (X > 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X \
      " must be positive and less than ",    \
      INT_MAX,                               \
      " but got ",                           \
      X)

#define CUDABLAS_NONNEGINT_CHECK(FD, X)       \
  TORCH_CHECK(                                \
      (X >= 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X  \
      " must be non-negative and less than ", \
      INT_MAX,                                \
      " but got ",                            \
      X)

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

static void _cublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).

  // Q: Why does Level3 check trans but this doesn't?
  // A: In level 2, the sizes (m, n) specify the size of A
  // (independent of trans value). In level 3. the sizes (m, n, k)
  // specify the sizes of op(A), op(B) where op depend on trans
  // values.
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

static void _cublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}

#ifndef USE_ROCM
uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (; ; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;
    }
  }
}
#endif

static size_t _parseChosenWorkspaceSize() {
  const char * val = getenv("CUBLASLT_WORKSPACE_SIZE");
#ifdef USE_ROCM
  if (!val) {
    // accept either env var
    val = getenv("HIPBLASLT_WORKSPACE_SIZE");
  }
#endif
  size_t workspace_size = 1024; /* default size in KiB according to #73328 */
  if (val) {
    try {
      workspace_size = std::stoi(val);
    } catch(std::invalid_argument const& e) {
      TORCH_WARN("invalid CUBLASLT_WORKSPACE_SIZE,",
                 " using default workspace size of ", workspace_size, " bytes.");
    } catch(std::out_of_range const& e) {
      TORCH_WARN("CUBLASLT_WORKSPACE_SIZE out of range,",
                 " using default workspace size of ", workspace_size, " bytes.");
    }
  }
  return workspace_size * 1024;
}

static size_t _getWorkspaceSize() {
  static size_t workspace_size = _parseChosenWorkspaceSize();
  return workspace_size;
}

} // anonymous namespace

namespace at::cuda::blas {

/* LEVEL 3 BLAS FUNCTIONS */

#define GEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
  } while (0)

#define BGEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldc);  \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, num_batches);  \
  } while (0)

template <>
void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(cublasDgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <>
void bgemm<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, stridea, reinterpret_cast<const cuDoubleComplex*>(b), ldb, strideb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc, stridec, num_batches));
}

template <>
void bgemm<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(cublasCgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, stridea, reinterpret_cast<const cuComplex*>(b), ldb, strideb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc, stridec, num_batches));
}

template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(at::Half);
  float falpha = alpha;
  float fbeta = beta;
#ifdef USE_ROCM
  int flag = 0;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                   hipOperationToRocOperation(opa),
                                   hipOperationToRocOperation(opb), (int)m, (int)n, (int)k,
                                   (void*)&falpha, a, rocblas_datatype_f16_r, (int)lda, stridea,
                                   b, rocblas_datatype_f16_r, (int)ldb, strideb,
                                   (void*)&fbeta, c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                   c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                   (int) num_batches, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                   0, flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5){
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, opa, opb, m, n, k,
      (void*)(&falpha), a, CUDA_R_16F, lda, stridea,
      b, CUDA_R_16F, ldb, strideb, (void*)(&fbeta),
      c, CUDA_R_16F, ldc, stridec,
      num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    for (const auto i : c10::irange(num_batches)) {
      at::cuda::blas::gemm<at::Half>(
        transa, transb,
        m, n, k,
        alpha, (a + i * stridea), lda,
        (b + i * strideb), ldb, beta,
        (c + i * stridec), ldc);
    }
  }
#endif // USE_ROCM
}

template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  BGEMM_CHECK_ARGVALUES(at::BFloat16);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  const float falpha = alpha;
  const float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

#if defined(USE_ROCM) && ROCM_VERSION >= 60000
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                  opa, opb, (int)m, (int)n, (int)k,
                                  (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea,
                                  b, CUDA_R_16BF, (int)ldb, strideb,
                                  (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec,
                                  (int)num_batches,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

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

template <typename Dtype>
void outputMatrix(Dtype *A, int64_t ld, int64_t stride, int64_t num_batches, int64_t row, int64_t col){
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
  if (m == 72 && k == 64){
    abftbgemm<T, 72, 72, 64>(transa, transb, m, n, k,
      alpha, dA, ldda, stridea,
      dB, lddb, strideb, beta,
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
  }
  else if(m == 64 && k == 72){
    abftbgemm<T, 64, 72, 72>(transa, transb, m, n, k,
      alpha, dA, ldda, stridea,
      dB, lddb, strideb, beta,
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
  }
  else if(m == 71 && k == 64){
    abftbgemm<T, 71, 71, 64>(transa, transb, m, n, k,
      alpha, dA, ldda, stridea,
      dB, lddb, strideb, beta,
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
  }
  else if(m == 64 && k == 71){
    abftbgemm<T, 64, 71, 71>(transa, transb, m, n, k,
      alpha, dA, ldda, stridea,
      dB, lddb, strideb, beta,
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
  }
  else{
    // orgAbftbgemm<T>(transa, transb, m, n, k,
    //   alpha, dA, ldda, stridea,
    //   dB, lddb, strideb, beta,
    //   dC, lddc, stridec,
    //   dA_colchk, ldda_colchk,
    //   dA_rowchk, ldda_rowchk,
    //   dA_colchk_r, ldda_colchk_r,
    //   dA_rowchk_r, ldda_rowchk_r,
    //   dB_colchk, lddb_colchk,
    //   dB_rowchk, lddb_rowchk,
    //   dB_colchk_r, lddb_colchk_r,
    //   dB_rowchk_r, lddb_rowchk_r,
    //   dC_colchk, lddc_colchk,
    //   dC_rowchk, lddc_rowchk,
    //   dC_colchk_r, lddc_colchk_r,
    //   dC_rowchk_r, lddc_rowchk_r,
    //   chk_v_a, chk_v_b, ld_chk_v,
    //   num_batches,
    //   COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER);
  }
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

template <typename T, int64_t M, int64_t N, int64_t K>
void abftbgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha,  \
              T *dA, int64_t ldda, int64_t stridea,                                           \
              T *dB, int64_t lddb, int64_t strideb, at::opmath_type<T> beta,             \
              T *dC, int64_t lddc, int64_t stridec,                                                 \
              T *dA_colchk, int64_t ldda_colchk, T *dA_rowchk, int64_t ldda_rowchk,              \
              T *dA_colchk_r, int64_t ldda_colchk_r, T *dA_rowchk_r, int64_t ldda_rowchk_r,      \
              T *dB_colchk, int64_t lddb_colchk, T *dB_rowchk, int64_t lddb_rowchk,            \
              T *dB_colchk_r, int64_t lddb_colchk_r, T *dB_rowchk_r, int64_t lddb_rowchk_r,      \
              T *dC_colchk, int64_t lddc_colchk, T *dC_rowchk, int64_t lddc_rowchk,           \
              T *dC_colchk_r, int64_t lddc_colchk_r, T *dC_rowchk_r, int64_t lddc_rowchk_r,   \
              T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,                                      \
              int64_t num_batches,
              bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER ){
  std::cout << "Using abftbgemm-T function." << std::endl;
  // See Note [Writing Nondeterministic Operations]
  // std::cout << "globalContext. \n";
  globalContext().alertCuBLASConfigNotDeterministic();
  // std::cout << "handle. \n";
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // std::cout << "transA & transB. \n";
  cublasOperation_t transA = _cublasOpFromChar(transa);
  cublasOperation_t transB = _cublasOpFromChar(transb);

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
    encode_col_lancher<T, M, K, 4>(num_batches,
                  dA, ldda, stridea, 
                  dA_colchk, ldda_colchk, (2*k), stream_colchk);
    // encode_col_v5<T, M, K, 4><<<num_batches, dim3(M*4, 1), (M+1)*K*sizeof(T), stream_colchk>>>(num_batches,
    //               dA, ldda, stridea, 
    //               dA_colchk, ldda_colchk, (2*k));
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
    encode_row_lancher<T, k, N>(num_batches,
                  dB, lddb, strideb, 
                  dB_rowchk, lddb_rowchk, (2*k), stream_rowchk);
    // encode_row_v5<T, K, N><<<num_batches, dim3(K*2, 1, 1), 0, stream_rowchk>>>(num_batches,
    //               dB, lddb, strideb, 
    //               dB_rowchk, lddb_rowchk, (2*k));
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
        update_col_lancher<T, K, N, 4>(num_batches,
                    dA_colchk, ldda_colchk, k*2, 
                    dB, lddb, strideb, 
                    dC_colchk, lddc_colchk, n*2, stream_colchk);
        // update_col_v5<T, K, N, 4><<<num_batches, dim3(K*4, 1, 1), ((K+1)*N+2*K) * sizeof(T), stream_colchk>>>(num_batches,
        //             dA_colchk, ldda_colchk, k*2, 
        //             dB, lddb, strideb, 
        //             dC_colchk, lddc_colchk, n*2);
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
        update_row_lancher<T, M, K>(num_batches,
                    dA, ldda, stridea, 
                    dB_rowchk, lddb_rowchk, k*2, 
                    dC_rowchk, lddc_rowchk, m*2, stream_rowchk);
        // update_row_v5<T, M, K><<<num_batches, dim3(M*2, 1, 1), (2*K) * sizeof(T), stream_rowchk>>>(num_batches,
        //             dA, ldda, stridea, 
        //             dB_rowchk, lddb_rowchk, k*2, 
        //             dC_rowchk, lddc_rowchk, m*2);
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
    encode_col_lancher<T, M, N, 4>(num_batches,
                    dC, lddc, stridec, 
                    dC_colchk_r, lddc_colchk_r, (2*n));
    // encode_col_v5<T, M, N, 4><<<num_batches, dim3(M*4, 1), (M+1)*N*sizeof(T), stream_colchk>>>(num_batches,
    //                dC, lddc, stridec, 
    //                 dC_colchk_r, lddc_colchk_r, (2*n));

    T E = 1e-2;
    detect_correct_col_lancher(dC, lddc, E, stridec,
                                            dC_colchk,      lddc_colchk,    (2*n),
                                            dC_colchk_r,    lddc_colchk_r,  (2*n),
                                            num_batches, n, stream_colchk);
    // detect_correct_col<<<dim3(num_batches), dim3(n), 0, stream_colchk>>>(dC, lddc, E, stridec,
    //                                         dC_colchk,      lddc_colchk,    (2*n),
    //                                         dC_colchk_r,    lddc_colchk_r,  (2*n));
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
    encode_row_lancher<T, M, N>(num_batches,
                    dC, lddc, stridec,
                    dC_rowchk_r, lddc_rowchk_r, (2*m), stream_rowchk);
    // encode_row_v5<T, M, N><<<num_batches, dim3(M*2, 1, 1), 0, stream_rowchk>>>(num_batches,
    //                 dC, lddc, stridec,
    //                 dC_rowchk_r, lddc_rowchk_r, (2*m));
    T E = 1e-2;
    detect_correct_row_lancher(dC, lddc, E, stridec,
                                          dC_rowchk, lddc_rowchk,     (2*m),
                                          dC_rowchk_r, lddc_rowchk_r, (2*m), 
                                          num_batches, m, stream_rowchk);
    // detect_correct_row<<<dim3(num_batches), dim3(m), 0, stream_rowchk>>>(dC, lddc, E, stridec,
    //                                       dC_rowchk, lddc_rowchk,     (2*m),
    //                                       dC_rowchk_r, lddc_rowchk_r, (2*m));
  }

  if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*m*2*n*2/t1/1e6, (double)num_batches*(m*n+2*n+2*m)/t1/1e6);
  }
}

template<typename T>
void myGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,
            const Dtype *a, int64_t lda, const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta,
            Dtype *c, int64_t ldc){
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

  T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
  T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
  T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
  T *chk_v_a;
  T *chk_v_b;

  size = (2*num_batches) * k * sizeof(T);
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

  int64_t len = m;
  size = 2 * len * sizeof(T);
  cudaMalloc((void**)&chk_v_a, size);
  T *h_matrix;
  h_matrix = (T *)malloc(size);
  int idx = 0;
  for(int i = 0; i < len; i++){
      idx = i*ld_chk_v;
      h_matrix[idx] = T(1);
      h_matrix[idx+1] = T(i+1);
  }
  cudaMemcpy(chk_v_a, h_matrix, size, cudaMemcpyHostToDevice);
  free(h_matrix);

  len = n;
  size = 2 * len * sizeof(T);
  cudaMalloc((void**)&chk_v_b, size);
  h_matrix = (T *)malloc(size);
  idx = 0;
  for(int i = 0; i < len; i++){
      idx = i*ld_chk_v;
      h_matrix[idx] = T(1);
      h_matrix[idx+1] = T(i+1);
  }
  cudaMemcpy(chk_v_b, h_matrix, size, cudaMemcpyHostToDevice);
  free(h_matrix);
  
  bool COL_FT = true;
  bool ROW_FT = true;
  bool DEBUG = false;
  bool CHECK_BEFORE = true;
  bool CHECK_AFTER = true;

  int64_t stridea = m*k;
  int64_t strideb = n*k;
  int64_t stridec = m*n;
  int64_t ldda = m;
  int64_t lddb = k;
  int64_t lddc = m;

  auto start = high_resolution_clock::now();
  abftGemm<T>(transa, transb, m, n, k,
      alpha, dA, ldda, stridea,
      dB, lddb, strideb, beta,
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
  std::cout << "abftGemm: " << duration.count() / 1000.0 << std::endl;

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

  return 0;
}

template <typename T>
void abftGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, 
      at::opmath_type<T> alpha, const T *a, int64_t lda, 
      T *b, int64_t ldb, at::opmath_type<T> beta,\
      T *c, int64_t ldc,
      T *dA_colchk, int64_t ldda_colchk, T *dA_rowchk, int64_t ldda_rowchk,             
      T *dA_colchk_r, int64_t ldda_colchk_r, T *dA_rowchk_r, int64_t ldda_rowchk_r,      
      T *dB_colchk, int64_t lddb_colchk, T *dB_rowchk, int64_t lddb_rowchk,            
      T *dB_colchk_r, int64_t lddb_colchk_r, T *dB_rowchk_r, int64_t lddb_rowchk_r,      
      T *dC_colchk, int64_t lddc_colchk, T *dC_rowchk, int64_t lddc_rowchk,           
      T *dC_colchk_r, int64_t lddc_colchk_r, T *dC_rowchk_r, int64_t lddc_rowchk_r,   
      T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,                                      
      bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER){
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  
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

  float falpha = at::opmath_type<float>(1);
  float fbeta = at::opmath_type<float>(0);

  if (DEBUG) cudaEventRecord(start, stream_main);
  if(opa == CUBLAS_OP_N){
    if constexpr (std::is_same<T, float>::value) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m, 
                    &falpha, chk_v_a, ld_chk_v, 
                    a, lda, &fbeta, 
                    dA_colchk, ldda_colchk);
    }
    else if constexpr(std::is_same<T, half>::value) {
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m,
                    &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                    a, CUDA_R_16F, lda,
                    &fbeta, dA_colchk, CUDA_R_16F, lda,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } 
  }
  else{
      
  }
  if (DEBUG) {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Achk, start, stop);
  }

  if (DEBUG) cudaEventRecord(start, stream_main);
  if (transB == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n, 
                    &falpha, b, ldb, 
                    chk_v_b, ld_chk_v, &fbeta, 
                    dB_rowchk, lddb_rowchk);
      }
      else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, m,
                      &falpha, b, CUDA_R_16F, ldb, 
                      chk_v_b, CUDA_R_16F, ld_chk_v,
                      &fbeta, dB_rowchk, CUDA_R_16F, lddb_rowchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
  } else{

  }
  if (DEBUG) {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
  }

  int64_t mem_row = 0;
  int64_t mem_col = 0;
  falpha = alpha;
  fbeta = beta;

  if (DEBUG)  cudaEventRecord(start, stream_main);
  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  if constexpr (std::is_same<T, float>::value) {
      cublasSgemm(handle, opa, opb, m, n, k, 
                    &alpha, a, lda, 
                    b, ldb, &beta, 
                    c, ldc);
  } else if constexpr(std::is_same<T, half>::value) {
      cublasGemmEx(handle, opa, opb, m, n, k,
        &falpha, a, CUDA_R_16F, lda, 
        b, CUDA_R_16F, ldb,
        &fbeta, c, CUDA_R_16F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaStreamSynchronize(stream_main);
  // std::cout << "Output dC: " << std::endl;
  // outputMatrix(dC, lddc, stridec, num_batches, m, n);

  if (DEBUG)  {
    cudaEventRecord(stop, stream_main);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);
    printf("  gemm: %f (%f)(%f)\n", t, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
    printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
    printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
  }

  if (DEBUG)  cudaEventRecord(start, stream_colchk);
  if(COL_FT){
    //std::cout << "  COL_FT" << std::endl;
    if (transA == CUBLAS_OP_N) {
      if (DEBUG) std::cout << "dA_colchk * dB = dC_colchk" << std::endl;;
      // K*4 must be greater then 2 * N
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, k,
                    &alpha, dA_colchk, ldda_colchk, 
                    b, ldb, &beta, 
                    dC_colchk, lddc_colchk);
      }
      else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, k,
                      &falpha, dA_colchk, CUDA_R_16F, ldda_colchk, 
                      b, CUDA_R_16F, ldb,
                      &fbeta, dC_colchk, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    else{
      if (DEBUG) std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;

    }
    // std::cout << "Output dC_colchk: " << std::endl;
    // outputMatrixChk(dC_colchk, ldda_colchk, n*2, num_batches, 2, n);
  }
  if (DEBUG)  {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*2*n*k*2/t1/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t1/1e6);
  }

  if (DEBUG)  cudaEventRecord(start, stream_rowchk);
  if (ROW_FT) {
      //std::cout << "  ROW_FT" << std::endl;
      if (transB == CUBLAS_OP_N) {
        if (DEBUG) std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
        //we can further work on this to support trans A
        if constexpr (std::is_same<T, float>::value){
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,  m, 2, k,
                      &alpha, a, lda, 
                      dB_rowchkb, lddb_rowchk, &beta, 
                      dC_rowchk, lddc_rowchk);
        }
        else if constexpr(std::is_same<T, half>::value){
          cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,  m, 2, k,
                        &falpha, a, CUDA_R_16F, lda, 
                        dB_rowchk, CUDA_R_16F, lddb_rowchk,
                        &fbeta, dC_rowchk, CUDA_R_16F, lddc_rowchk,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
      } else{
        if (DEBUG) std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
      }
      // std::cout << "Output dC_rowchk: " << std::endl;
      // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
  }
  if (DEBUG)  {
    cudaEventRecord(stop, stream_rowchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t1, start, stop);
    printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*m*2*k*2/t1/1e6, (double)1*(m*k+k*2+m*2)*sizeof(T)/t1/1e6);
  }

  // --- check check-sum of C---//
  if (DEBUG) std::cout << "------Check check-sum-------" << std::endl;
  if (DEBUG) cudaEventRecord(start, stream_colchk);
  if (COL_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-col\n");
    if constexpr (std::is_same<T, float>::value){
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &alpha, chk_v_a, ld_chk, 
                      c, ldc, &beta, 
                      dC_colchk_r, lddc_colchk_r);
    }
    else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk, 
                      c, CUDA_R_16F, ldc,
                      &fbeta, dC_colchk_r, CUDA_R_16F, lddc_colchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    T E = 1e-2;
    detect_correct_col_lancher(dC, lddc, E, stridec,
                                    dC_colchk,      lddc_colchk,    (2*n),
                                    dC_colchk_r,    lddc_colchk_r,  (2*n), stream_colchk);
  }
  if (DEBUG)  {
    cudaEventRecord(stop, stream_colchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t1, start, stop);
    printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*2*n*m*2/t1/1e6, (double)num_batches*(m*n+2*m+2*n)*sizeof(T)/t1/1e6);
  }

  if (DEBUG)  cudaEventRecord(start, stream_rowchk);
  if (ROW_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-row\n");
    if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &alpha, chk_v_a, ld_chk, 
                      c, ldc, &beta, 
                      dC_colchk_r, lddc_colchk_r);
    }
    else if constexpr(std::is_same<T, half>::value){
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk, 
                      c, CUDA_R_16F, ldc,
                      &fbeta, dC_colchk_r, CUDA_R_16F, lddc_colchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    T E = 1e-2;
    detect_correct_row_lancher(dC, lddc, E, stridec,
                                    dC_rowchk, lddc_rowchk,     (2*m),
                                    dC_rowchk_r, lddc_rowchk_r, (2*m), stream_rowchk);
  }
  if (DEBUG)  {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(1)*m*2*n*2/t1/1e6, (double)1*(m*n+2*n+2*m)*sizeof(T)/t1/1e6);
  }
}

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(cublasDgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(cublasZgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc));
}

template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(cublasCgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc));
}

template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::Half);
#ifdef USE_ROCM
  int flag = 0;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_ex(
      (rocblas_handle)handle,
      hipOperationToRocOperation(opa),
      hipOperationToRocOperation(opb),
      m,
      n,
      k,
      &falpha,
      a,
      rocblas_datatype_f16_r,
      lda,
      b,
      rocblas_datatype_f16_r,
      ldb,
      &fbeta,
      c,
      rocblas_datatype_f16_r,
      ldc,
      c,
      rocblas_datatype_f16_r,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5) {
#ifndef USE_ROCM
    cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
    if (!at::globalContext().allowFP16ReductionCuBLAS()) {
      cublas_flags = static_cast<cublasMath_t>(cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
#endif
    // Disallow fp16 reductions that could lead to unexpected overflow issues.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc));
  }
#endif
}

template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::BFloat16);
#ifndef USE_ROCM
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  if (!at::globalContext().allowBF16ReductionCuBLAS()) {
    cublas_flags = static_cast<cublasMath_t>(cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }
#endif
#if defined(USE_ROCM) && ROCM_VERSION >= 60000
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &falpha,
      a,
      CUDA_R_16BF,
      lda,
      b,
      CUDA_R_16BF,
      ldb,
      &fbeta,
      c,
      CUDA_R_16BF,
      ldc,
      compute_type,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

#if (!defined(USE_ROCM) && !defined(_MSC_VER)) || (defined(USE_ROCM) && ROCM_VERSION >= 50700)

#if defined(USE_ROCM) && ROCM_VERSION >= 50700 && ROCM_VERSION < 60000
// only for rocm 5.7 where we first supported hipblaslt, it was difficult
// to hipify correctly without this change.
#define hipDataType hipblasDatatype_t
#endif

// hipblaslt custom types were a temporary work-around
#if defined(USE_ROCM) && ROCM_VERSION >= 60000 && HIPBLASLT_CUSTOM_DATA_TYPE
hipblasltDatatype_t hipToLt(hipDataType type) {
    switch (type) {
        case HIP_R_32F: return HIPBLASLT_R_32F;
        case HIP_R_64F: return HIPBLASLT_R_64F;
        case HIP_R_16F: return HIPBLASLT_R_16F;
        case HIP_R_8I: return HIPBLASLT_R_8I;
        case HIP_C_32F: return HIPBLASLT_C_32F;
        case HIP_C_64F: return HIPBLASLT_C_64F;
        case HIP_C_16F: return HIPBLASLT_C_16F;
        case HIP_C_8I: return HIPBLASLT_C_8I;
        case HIP_R_8U: return HIPBLASLT_R_8U;
        case HIP_C_8U: return HIPBLASLT_C_8U;
        case HIP_R_32I: return HIPBLASLT_R_32I;
        case HIP_C_32I: return HIPBLASLT_C_32I;
        case HIP_R_32U: return HIPBLASLT_R_32U;
        case HIP_C_32U: return HIPBLASLT_C_32U;
        case HIP_R_16BF: return HIPBLASLT_R_16B;
        case HIP_C_16BF: return HIPBLASLT_C_16B;
        default: TORCH_CHECK(false);
    }
}
#define HIPTOLT(type) hipToLt(type)
#else
#define HIPTOLT(type) type
#endif

#if defined(USE_ROCM) && ROCM_VERSION >= 60000 && HIPBLASLT_CUSTOM_COMPUTE_TYPE
hipblasLtComputeType_t hipblasToLt(hipblasComputeType_t type) {
    switch (type) {
        case HIPBLAS_COMPUTE_32F: return HIPBLASLT_COMPUTE_F32;
        case HIPBLAS_COMPUTE_32F_FAST_16F: return HIPBLASLT_COMPUTE_F32_FAST_F16;
        case HIPBLAS_COMPUTE_32F_FAST_TF32: return HIPBLASLT_COMPUTE_F32_FAST_XF32;
        case HIPBLAS_COMPUTE_64F: return HIPBLASLT_COMPUTE_F64;
        case HIPBLAS_COMPUTE_32I: return HIPBLASLT_COMPUTE_I32;
        default: TORCH_CHECK(false);
    }
}
#define HIPCOMPTOLT(type) hipblasToLt(type)
#else
#define HIPCOMPTOLT(type) type
#endif

namespace {
// Following the pattern of CuSparseDescriptor
// Defined here for now because this is the only place cublas_lt interface is
// used but can be moved to a header once cublas_lt interface is used in
// multiple places.
template <typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, cublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<
                                     cublasLtMatmulDescOpaque_t,
                                     &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(
      cublasComputeType_t compute_type,
      cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescCreate(&raw_descriptor, HIPCOMPTOLT(compute_type), HIPTOLT(scale_type)));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatrixLayout : public CuBlasLtDescriptor<
                                 cublasLtMatrixLayoutOpaque_t,
                                 &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(
      cudaDataType_t type,
      uint64_t rows,
      uint64_t cols,
      int64_t ld,
      bool t = false) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatrixLayoutCreate(&raw_descriptor, HIPTOLT(type), t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
};

class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<
                                     cublasLtMatmulPreferenceOpaque_t,
                                     &cublasLtMatmulPreferenceDestroy> {
 public:
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatmulPreferenceAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};
} // namespace

template <typename T>
void myGemmBias (
  bool transpose_mat1, bool transpose_mat2,
  int64_t m, int64_t n, int64_t k,
  at::opmath_type<T> alpha_val,
  const T* mat1_ptr,
  int64_t mat1_ld,
  const T* mat2_ptr,
  int64_t mat2_ld,
  const T* bias,
  T* result_ptr,
  int64_t result_ld,
  GEMMAndBiasActivationEpilogue activation){
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

    T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
    T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
    T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
    T *chk_v_a;
    T *chk_v_b;

    size = 2 * k * sizeof(T);
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

    size = 2 * n * sizeof(T);
    cudaMalloc((void**)&dC_colchk, size);
    cudaMemset(dC_colchk, 0, size);
    cudaMalloc((void**)&dC_colchk_r, size);
    cudaMemset(dC_colchk_r, 0, size);
    
    size = 2 * m * sizeof(T);
    cudaMalloc((void**)&dC_rowchk, size);
    cudaMemset(dC_rowchk, 0, size);
    cudaMalloc((void**)&dC_rowchk_r, size);
    cudaMemset(dC_rowchk_r, 0, size);

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
    // std::cout << "chk_v_a: " << std::endl;
    // outputChk(chk_v_a, 1, ld_chk_v, 0, 2, m);
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
    // std::cout << "chk_v_b: " << std::endl;
    // outputChk(chk_v_a, 1, ld_chk_v, 0, 2, len);
    free(h_matrix);
    //std::cout << "  finish chk_v." << std::endl;

    bool COL_FT = true;
    bool ROW_FT = true;
    bool DEBUG = true;
    bool CHECK_BEFORE = true;
    bool CHECK_AFTER = true;

    T alpha = 1;
    T beta = 0;
    int64_t stridea = m*k;
    int64_t strideb = n*k;
    int64_t stridec = m*n;
    int64_t ldda = m;
    int64_t lddb = k;
    int64_t lddc = m;

    auto start = high_resolution_clock::now();
    abftGemmBias<T>(transpose_mat1, transpose_mat2, m, n, k,
        alpha_val, mat1_ptr, mat1_ld,
        mat2_ptr, mat2_ld, bias, 
        result_ptr, result_ld,
        activation,
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
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<microseconds>(stop - start);
    std::cout << "abftGemmBias: " << duration.count() / 1000.0 << std::endl;
}

template <typename T, int64_t M, int64_t N, int64_t K>
void abftGemmBias(
    bool transpose_mat1, bool transpose_mat2,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<T> alpha_val, const T* mat1_ptr, int64_t mat1_ld,
    const T* mat2_ptr, int64_t mat2_ld, const T* bias,
    T* result_ptr, int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation,
    T *dA_colchk, int64_t ldda_colchk, T *dA_rowchk, int64_t ldda_rowchk,              
    T *dA_colchk_r, int64_t ldda_colchk_r, T *dA_rowchk_r, int64_t ldda_rowchk_r,      
    T *dB_colchk, int64_t lddb_colchk, T *dB_rowchk, int64_t lddb_rowchk,            
    T *dB_colchk_r, int64_t lddb_colchk_r, T *dB_rowchk_r, int64_t lddb_rowchk_r,     
    T *dC_colchk, int64_t lddc_colchk, T *dC_rowchk, int64_t lddc_rowchk,           
    T *dC_colchk_r, int64_t lddc_colchk_r, T *dC_rowchk_r, int64_t lddc_rowchk_r,   
    T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,                                      
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER) {
  
  std::cout << "Using gemm_and_bias." << std::endl;
  using opmath_t = at::opmath_type<T>;
  opmath_t beta_val = 0; // bias is added in epilogue

  cudaDataType_t abcType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  if constexpr (std::is_same_v<T, double>) {
#if !defined(USE_ROCM) || (defined(USE_ROCM) && ROCM_VERSION >= 60000)
    abcType = CUDA_R_64F;
    computeType = CUBLAS_COMPUTE_64F;
    scaleType = CUDA_R_64F;
#else
    TORCH_CHECK(false, "gemm_and_bias is only supported for double type on ROCm 6.0 and above");
#endif
  } else if constexpr (std::is_same_v<T, float>) {
#ifndef USE_ROCM
    if (at::globalContext().allowTF32CuBLAS()) {
      computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
#endif
    abcType = CUDA_R_32F;
  } else if constexpr (std::is_same_v<T, at::Half>) {
    abcType = CUDA_R_16F;
  } else if constexpr (std::is_same_v<T, at::BFloat16>) {
    abcType = CUDA_R_16BF;
  }

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  if (activation == GEMMAndBiasActivationEpilogue::RELU) {
    epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
  } else if (activation == GEMMAndBiasActivationEpilogue::GELU) {
#if CUDA_VERSION >= 11040 || defined(USE_ROCM)
    epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
#endif
  }

  if (bias != nullptr) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
  }

  CuBlasLtMatrixLayout Adesc(abcType, m, k, mat1_ld, transpose_mat1);
  CuBlasLtMatrixLayout Bdesc(abcType, k, n, mat2_ld, transpose_mat2);
  CuBlasLtMatrixLayout Cdesc(abcType, m, n, result_ld);

  CuBlasLtMatmulPreference preference;
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
  // setting this to 1M.
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

#ifndef USE_ROCM
  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat1_ptr));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat2_ptr));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(result_ptr));
  uint32_t d_alignment = _getAlignment(reinterpret_cast<uintptr_t>(bias));
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, d_alignment);
#endif

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspace = allocator.allocate(workspaceSize);

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  std::cout << "cublasLtMatmulAlgoGetHeuristic()" << std::endl;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  //
  cudaStream_t stream_main, stream_colchk, stream_rowchk;
  cudaStreamCreate(&stream_main);
  cudaStreamCreate(&stream_colchk);
  cudaStreamCreate(&stream_rowchk);
  cublasSetStream(ltHandle, stream_main);

  cudaEvent_t main_compute_done;
  cudaEventCreate(&main_compute_done);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t, t1, t_Achk, t_Bchk, t_Crowchk, t_Ccolchk;

  float falpha = at::opmath_type<T>(1);
  float fbeta = at::opmath_type<T>(0);

  if (DEBUG) cudaEventRecord(start, stream_colchk);
  if(transa == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m, 
                      &falpha, chk_v_a, ld_chk_v, 
                      mat1_ptr, mat1_ld, &fbeta, 
                      dA_colchk, ldda_colchk);
      }
      else if constexpr(std::is_same<T, half>::value) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      mat1_ptr, CUDA_R_16F, mat1_ld,
                      &fbeta, dA_colchk, CUDA_R_16F, lda,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } 
  }
  if (DEBUG) {
        cudaEventRecord(stop, stream_colchk);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Achk, start, stop);
  }

  if (DEBUG) cudaEventRecord(start, stream_rowchk);
  if (transb == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n, 
                    &falpha, mat2_ptr, mat2_ptr, 
                    chk_v_b, ld_chk_v, &fbeta, 
                    dB_rowchk, lddb_rowchk);
      }
      else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, m,
                      &falpha, mat2_ptr, CUDA_R_16F, mat2_ld, 
                      chk_v_b, CUDA_R_16F, ld_chk_v,
                      &fbeta, dB_rowchk, CUDA_R_16F, lddb_rowchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
  } 
  if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
  }

  if (DEBUG) cudaEventRecord(start, stream_colchk);
  if(transa == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m, 
                      &falpha, chk_v_a, ld_chk_v, 
                      mat1_ptr, mat1_ld, &fbeta, 
                      dA_colchk, ldda_colchk);
      }
      else if constexpr(std::is_same<T, half>::value) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      mat1_ptr, CUDA_R_16F, mat1_ld,
                      &fbeta, dA_colchk, CUDA_R_16F, lda,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } 
  }
  if (DEBUG) {
        cudaEventRecord(stop, stream_colchk);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Achk, start, stop);
  }

  if (DEBUG) cudaEventRecord(start, stream_rowchk);
  if (transb == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n, 
                    &falpha, mat2_ptr, mat2_ptr, 
                    chk_v_b, ld_chk_v, &fbeta, 
                    dB_rowchk, lddb_rowchk);
      }
      else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, m,
                      &falpha, mat2_ptr, CUDA_R_16F, mat2_ld, 
                      chk_v_b, CUDA_R_16F, ld_chk_v,
                      &fbeta, dB_rowchk, CUDA_R_16F, lddb_rowchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
  } 
  if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
  }

  

  int64_t mem_row = 0;
  int64_t mem_col = 0;

  falpha = alpha_val;
  fbeta = beta_val;

  if (DEBUG)  cudaEventRecord(start, stream_main);
  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  std::cout << "cublasLtMatmul" << std::endl;
  cublasLtMatmul( ltHandle, computeDesc.descriptor(), &alpha_val,
      mat1_ptr, Adesc.descriptor(),
      mat2_ptr, Bdesc.descriptor(), &beta_val,
      result_ptr, Cdesc.descriptor(), 
      result_ptr, Cdesc.descriptor(),
      &heuristicResult.algo, workspace.mutable_get(), workspaceSize, stream_main);
  cudaStreamSynchronize(stream_main);

  if (DEBUG) {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t, start, stop);
      printf("  gemm: %f (%f)(%f)\n", t, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
      printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
      printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
  }

  if (DEBUG)  cudaEventRecord(start, stream_colchk);
  if(COL_FT){
    //std::cout << "  COL_FT" << std::endl;
    if (transa == CUBLAS_OP_N) {
      if (DEBUG) std::cout << "dA_colchk * dB = dC_colchk" << std::endl;;
      // K*4 must be greater then 2 * N
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, k,
                    &falpha, dA_colchk, ldda_colchk, 
                    mat2_ptr, mat2_ld, &fbeta, 
                    dC_colchk, lddc_colchk);
      }
      else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, k,
                      &falpha, dA_colchk, CUDA_R_16F, ldda_colchk, 
                      mat2_ptr, CUDA_R_16F, mat2_ptr,
                      &fbeta, dC_colchk, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    else{
      if (DEBUG) std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;
    }
      // std::cout << "Output dC_colchk: " << std::endl;
      // outputMatrixChk(dC_colchk, ldda_colchk, n*2, num_batches, 2, n);
  }
  if (DEBUG)  {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*2*n*k*2/t1/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t1/1e6);
  }

  if (DEBUG)  cudaEventRecord(start, stream_rowchk);
  if (ROW_FT) {
      //std::cout << "  ROW_FT" << std::endl;
      if (transB == CUBLAS_OP_N) {
        if (DEBUG) std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
        //we can further work on this to support trans A
        if constexpr (std::is_same<T, float>::value){
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,  m, 2, k,
                      &falpha, mat1_ptr, mat1_ptr, 
                      dB_rowchkb, lddb_rowchk, &fbeta, 
                      dC_rowchk, lddc_rowchk);
        }
        else if constexpr(std::is_same<T, half>::value){
          cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,  m, 2, k,
                        &falpha, mat1_ptr, CUDA_R_16F, mat1_ld, 
                        dB_rowchk, CUDA_R_16F, lddb_rowchk,
                        &fbeta, dC_rowchk, CUDA_R_16F, lddc_rowchk,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
      } else{
        if (DEBUG) std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
      }
      // std::cout << "Output dC_rowchk: " << std::endl;
      // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
  }
  if (DEBUG)  {
    cudaEventRecord(stop, stream_rowchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t1, start, stop);
    printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*m*2*k*2/t1/1e6, (double)1*(m*k+k*2+m*2)*sizeof(T)/t1/1e6);
  }

  // --- check check-sum of C---//
  if (DEBUG) std::cout << "------Check check-sum-------" << std::endl;
  if (DEBUG) cudaEventRecord(start, stream_colchk);
  if (COL_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-col\n");
    if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &falpha, chk_v_a, ld_chk_v, 
                      result_ptr, result_ld, &fbeta, 
                      dC_colchk_r, lddc_colchk_r);
    }
    else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      result_ptr, CUDA_R_16F, result_ld,
                      &fbeta, dC_colchk_r, CUDA_R_16F, lddc_colchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    T E = 1e-2;
    detect_correct_col_lancher(dC, lddc, E, stridec,
                              dC_colchk,      lddc_colchk,    (2*n),
                              dC_colchk_r,    lddc_colchk_r,  (2*n), stream_colchk);
    
    // detect_correct_col<<<dim3(num_batches), dim3(n), 0, stream_colchk>>>(dC, lddc, E, stridec,
    //                                       dC_colchk,      lddc_colchk,    (2*n),
    //                                       dC_colchk_r,    lddc_colchk_r,  (2*n));
  }
  if (DEBUG)  {
    cudaEventRecord(stop, stream_colchk);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t1, start, stop);
    printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(1)*2*n*m*2/t1/1e6, (double)1*(m*n+2*m+2*n)*sizeof(T)/t1/1e6);
  }

  if (DEBUG)  cudaEventRecord(start, stream_rowchk);
  if (ROW_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-row\n");
    if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &alpha, result_ptr, result_ld, 
                      chk_v_b, ld_chk_v, &beta, 
                      dC_rowchk_r, lddc_rowchk_r);
    }
    else if constexpr(std::is_same<T, half>::value){
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &falpha, result_ptr, CUDA_R_16F, result_ld, 
                      chk_v_b, CUDA_R_16F, ld_chk_v,
                      &fbeta, dC_rowchk_r, CUDA_R_16F, lddc_rowchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    T E = 1e-2;
    detect_correct_row_lancher(dC, lddc, E, stridec,
                                dC_rowchk, lddc_rowchk,     (2*m),
                                dC_rowchk_r, lddc_rowchk_r, (2*m), stream_rowchk);
    // detect_correct_row<<<dim3(num_batches), dim3(m), 0, stream_rowchk>>>(dC, lddc, E, stridec,
    //                                       dC_rowchk, lddc_rowchk,     (2*m),
    //                                       dC_rowchk_r, lddc_rowchk_r, (2*m));

  }

  if (DEBUG)  {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(1)*m*2*n*2/t1/1e6, (double)1*(m*n+2*n+2*m)*sizeof(T)/t1/1e6);
  }
}


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
    GEMMAndBiasActivationEpilogue activation) {
  
  std::cout << "Using gemm_and_bias." << std::endl;
  using opmath_t = at::opmath_type<Dtype>;
  opmath_t beta_val = 0; // bias is added in epilogue

  cudaDataType_t abcType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  if constexpr (std::is_same_v<Dtype, double>) {
#if !defined(USE_ROCM) || (defined(USE_ROCM) && ROCM_VERSION >= 60000)
    abcType = CUDA_R_64F;
    computeType = CUBLAS_COMPUTE_64F;
    scaleType = CUDA_R_64F;
#else
    TORCH_CHECK(false, "gemm_and_bias is only supported for double type on ROCm 6.0 and above");
#endif
  } else if constexpr (std::is_same_v<Dtype, float>) {
#ifndef USE_ROCM
    if (at::globalContext().allowTF32CuBLAS()) {
      computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
#endif
    abcType = CUDA_R_32F;
  } else if constexpr (std::is_same_v<Dtype, at::Half>) {
    abcType = CUDA_R_16F;
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    abcType = CUDA_R_16BF;
  }

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  if (activation == GEMMAndBiasActivationEpilogue::RELU) {
    epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
  } else if (activation == GEMMAndBiasActivationEpilogue::GELU) {
#if CUDA_VERSION >= 11040 || defined(USE_ROCM)
    epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
#endif
  }

  if (bias != nullptr) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
  }

  CuBlasLtMatrixLayout Adesc(abcType, m, k, mat1_ld, transpose_mat1);
  CuBlasLtMatrixLayout Bdesc(abcType, k, n, mat2_ld, transpose_mat2);
  CuBlasLtMatrixLayout Cdesc(abcType, m, n, result_ld);

  CuBlasLtMatmulPreference preference;
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
  // setting this to 1M.
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

#ifndef USE_ROCM
  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat1_ptr));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat2_ptr));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(result_ptr));
  uint32_t d_alignment = _getAlignment(reinterpret_cast<uintptr_t>(bias));
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, d_alignment);
#endif

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspace = allocator.allocate(workspaceSize);

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  std::cout << "cublasLtMatmulAlgoGetHeuristic()" << std::endl;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  std::cout << "cublasLtMatmul" << std::endl;
  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      workspace.mutable_get(),
      workspaceSize,
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abcType ",
      abcType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<double> alpha_val,
    const double* mat1_ptr,
    int64_t mat1_ld,
    const double* mat2_ptr,
    int64_t mat2_ld,
    const double* bias,
    double* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<float> alpha_val,
    const float* mat1_ptr,
    int64_t mat1_ld,
    const float* mat2_ptr,
    int64_t mat2_ld,
    const float* bias,
    float* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::Half> alpha_val,
    const at::Half* mat1_ptr,
    int64_t mat1_ld,
    const at::Half* mat2_ptr,
    int64_t mat2_ld,
    const at::Half* bias,
    at::Half* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::BFloat16> alpha_val,
    const at::BFloat16* mat1_ptr,
    int64_t mat1_ld,
    const at::BFloat16* mat2_ptr,
    int64_t mat2_ld,
    const at::BFloat16* bias,
    at::BFloat16* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

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
    const void *result_scale_ptr,
    int64_t result_ld,
    ScalarType result_dtype,
    void* amax_ptr,
    bool use_fast_accum) {
  #if CUDA_VERSION >= 11080
  const auto computeType = CUBLAS_COMPUTE_32F;
  const auto scaleType = CUDA_R_32F;
  const int8_t fastAccuMode = use_fast_accum ? 1 : 0;
  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, _cublasOpFromChar(transa));
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, _cublasOpFromChar(transb));
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, mat1_scale_ptr);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, mat2_scale_ptr);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, result_scale_ptr);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, amax_ptr);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fastAccuMode);
  CuBlasLtMatrixLayout Adesc(ScalarTypeToCudaDataType(mat1_dtype), m, k, mat1_ld, transa == 't');
  CuBlasLtMatrixLayout Bdesc(ScalarTypeToCudaDataType(mat2_dtype), k, n, mat2_ld, transb == 't');
  CuBlasLtMatrixLayout Cdesc(ScalarTypeToCudaDataType(bias_dtype), m, n, result_ld);
  CuBlasLtMatrixLayout Ddesc(ScalarTypeToCudaDataType(result_dtype), m, n, result_ld);
  if (bias_ptr) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, ScalarTypeToCudaDataType(bias_dtype));
  }
  size_t workspaceSize = _getWorkspaceSize();
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspace = allocator.allocate(workspaceSize);

  CuBlasLtMatmulPreference preference;
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Ddesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  float alpha_val = 1.0;
  float beta_val = 0.0;
  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      nullptr,
      Cdesc.descriptor(),
      result_ptr,
      Ddesc.descriptor(),
      &heuristicResult.algo,
      workspace.mutable_get(),
      workspaceSize,
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transa,
      " transpose_mat2 ",
      transb,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
  return;
  #endif // CUDA_VERSION >= 11080
  TORCH_CHECK(false, "scaled_gemm is only supported for CUDA 11.8 and above");
}

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
    int64_t result_ld) {
#if !defined(USE_ROCM) || (defined(USE_ROCM) && ROCM_VERSION >= 60000)

  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
  cudaDataType_t scaleType = CUDA_R_32I;

  cudaDataType_t abType = CUDA_R_8I;
  cudaDataType_t cType = CUDA_R_32I;

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);


  CuBlasLtMatrixLayout Adesc(abType, m, k, mat1_ld, transpose_mat1);
  CuBlasLtMatrixLayout Bdesc(abType, k, n, mat2_ld, transpose_mat2);
  CuBlasLtMatrixLayout Cdesc(cType, m, n, result_ld);

  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  // cublas team: alpha and beta need to be the same dtype as of scaleType
  at::opmath_type<int32_t> alpha_val = 1;
  int32_t beta_val = 0;

  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
      nullptr, // Heuristics don't seem to work for int8
      nullptr, // Non-zero workspace doesn't seem to work.
      0,
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
#else
  TORCH_CHECK(false, "int8_gemm is only supported for ROCm 6.0 and above");
#endif // !defined(USE_ROCM) || (defined(USE_ROCM) && ROCM_VERSION >= 60000)
}
#endif // (!defined(USE_ROCM) && !defined(_MSC_VER)) || (defined(USE_ROCM) && ROCM_VERSION >= 50700)

// ROCm 5.6 hipblas matches the const Dtype *A API, but prior hipblas does not.
#if defined(USE_ROCM) && ROCM_VERSION < 50600
#define ROCM_CONST_BUG_CAST(Type, Input) const_cast<Type>(reinterpret_cast<const Type>(Input))
#else
#define ROCM_CONST_BUG_CAST(Type, Input) reinterpret_cast<const Type>(Input)
#endif

template <>
void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasStrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDtrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<c10::complex<float>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      ROCM_CONST_BUG_CAST(cuComplex*, A),
      lda,
      reinterpret_cast<cuComplex*>(B),
      ldb));
}

template <>
void trsm<c10::complex<double>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      ROCM_CONST_BUG_CAST(cuDoubleComplex*, A),
      lda,
      reinterpret_cast<cuDoubleComplex*>(B),
      ldb));
}

template <>
void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasStrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<float>>(
    CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      reinterpret_cast<cuComplex**>(A),
      lda,
      reinterpret_cast<cuComplex**>(B),
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<double>>(
    CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      reinterpret_cast<cuDoubleComplex**>(A),
      lda,
      reinterpret_cast<cuDoubleComplex**>(B),
      ldb,
      batchCount));
}

/* LEVEL 2 BLAS FUNCTIONS */

#define GEMV_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, n); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incx); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incy); \
  } while (0)

template <>
void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(
      cublasZgemv(handle, op, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(y), incy));
}

template <>
void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>)) {
  // gemv is bw bound, and does not benefit from TF32. But the precision
  // loss still happens on TF32. So we disable it here.
  NoTF32Guard disable_tf32;
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(
      cublasCgemv(handle, op, m, n, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(y), incy));
}

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(
      cublasDgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float)) {
  // gemv is bw bound, and does not benefit from TF32. But the precision
  // loss still happens on TF32. So we disable it here.
  NoTF32Guard disable_tf32;
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(
      cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half)) {
  // In general, cublas regards matrices as column-major.
  // The cublasS/Dgemv usages in cuda::blas::gemv<float>/<double> above
  // require that external blas::gemv callers obey the following convention:
  //
  // If "a" is row-major with shape (output, summed) in blas::gemv's caller,
  // caller interprets it as column-major with shape (summed, output), passes
  // summed and output respectively to our local vars m, n, and requests that cublas
  // internally transpose ("trans") the column-major interpretation of a.
  //
  // There's no such thing as "cublasHalfgemv", so here we hack gemv with a gemm.
  // However, we must allow the same calling convention, because the caller shouldn't
  // have to swap args based on whether it's calling blas::gemv<at::Half> or <float>.

  bool trans_bool = (_cublasOpFromChar(trans) != CUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  // After swap, local vars m, n contain the output and summed sizes respectively,
  // regardless of whether "a" was row-major or column-major in gemv<>'s caller.

  // To handle the possibility incy > 1, interprets vector y as column-major matrix with one row
  // (shape (1, output)) and leading dim incy.
  // trans(a)*x would compute a matrix with one column (shape (output, 1)) which wouldn't match y.
  // So instead, we interpret x similarly to y, as a column-major matrix with one row
  // (shape (1, summed)) and leading dim incx.  The gemm then carries out x*transpose(trans(a)) to
  // produce a matrix with one row (shape (1, output)), matching y.
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::Half>(
      'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}

template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16)) {
  bool trans_bool = (_cublasOpFromChar(trans) != CUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::BFloat16>(
      'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}

/* LEVEL 1 BLAS FUNCTIONS */

template <>
void dot<double>(CUDABLAS_DOT_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
}

template <>
void dot<float>(CUDABLAS_DOT_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
}

template <>
void dot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZdotu(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

template <>
void dot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCdotu(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

template <>
void dot<at::Half>(CUDABLAS_DOT_ARGTYPES(at::Half)) {
  TORCH_CUDABLAS_CHECK(cublasDotEx(
      handle,
      n,
      x,
      CUDA_R_16F,
      incx,
      y,
      CUDA_R_16F,
      incy,
      result,
      CUDA_R_16F,
      CUDA_R_32F));
}

template <>
void dot<at::BFloat16>(CUDABLAS_DOT_ARGTYPES(at::BFloat16)) {
  TORCH_CUDABLAS_CHECK(cublasDotEx(
      handle,
      n,
      x,
      CUDA_R_16BF,
      incx,
      y,
      CUDA_R_16BF,
      incy,
      result,
      CUDA_R_16BF,
      CUDA_R_32F));
}

template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

template <>
void getrsBatched<float>(CUDABLAS_GETRS_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      dA_array,
      lda,
      ipiv_array,
      dB_array,
      ldb,
      info_array,
      batchsize));
}

template <>
void getrsBatched<double>(CUDABLAS_GETRS_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      dA_array,
      lda,
      ipiv_array,
      dB_array,
      ldb,
      info_array,
      batchsize));
}

template <>
void getrsBatched<c10::complex<float>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<cuComplex**>(dA_array),
      lda,
      ipiv_array,
      reinterpret_cast<cuComplex**>(dB_array),
      ldb,
      info_array,
      batchsize));
}

template <>
void getrsBatched<c10::complex<double>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<cuDoubleComplex**>(dA_array),
      lda,
      ipiv_array,
      reinterpret_cast<cuDoubleComplex**>(dB_array),
      ldb,
      info_array,
      batchsize));
}

template <>
void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSgeqrfBatched(
      handle, m, n, A_array, lda, tau_array, info, batchsize));
}

template <>
void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDgeqrfBatched(
      handle, m, n, A_array, lda, tau_array, info, batchsize));
}

template <>
void geqrfBatched<c10::complex<float>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCgeqrfBatched(
      handle,
      m,
      n,
      reinterpret_cast<cuComplex**>(A_array),
      lda,
      reinterpret_cast<cuComplex**>(tau_array),
      info,
      batchsize));
}

template <>
void geqrfBatched<c10::complex<double>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZgeqrfBatched(
      handle,
      m,
      n,
      reinterpret_cast<cuDoubleComplex**>(A_array),
      lda,
      reinterpret_cast<cuDoubleComplex**>(tau_array),
      info,
      batchsize));
}

template <>
void getrfBatched<double>(
    int n, double** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasDgetrfBatched(
      handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
}

template <>
void getrfBatched<float>(
    int n, float** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSgetrfBatched(
      handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
}

template <>
void getrfBatched<c10::complex<double>>(
    int n,
    c10::complex<double>** dA_array,
    int ldda,
    int* ipiv_array,
    int* info_array,
    int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasZgetrfBatched(
      handle,
      n,
      reinterpret_cast<cuDoubleComplex**>(dA_array),
      ldda,
      ipiv_array,
      info_array,
      batchsize));
}

template <>
void getrfBatched<c10::complex<float>>(
    int n,
    c10::complex<float>** dA_array,
    int ldda,
    int* ipiv_array,
    int* info_array,
    int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasCgetrfBatched(
      handle,
      n,
      reinterpret_cast<cuComplex**>(dA_array),
      ldda,
      ipiv_array,
      info_array,
      batchsize));
}


template <>
void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDgelsBatched(
      handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
}

template <>
void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSgelsBatched(
      handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
}

template <>
void gelsBatched<c10::complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZgelsBatched(
      handle, trans,
      m, n, nrhs,
      reinterpret_cast<cuDoubleComplex**>(dA_array),
      ldda,
      reinterpret_cast<cuDoubleComplex**>(dC_array),
      lddc,
      info,
      devInfoArray,
      batchSize));
}

template <>
void gelsBatched<c10::complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCgelsBatched(
      handle, trans,
      m, n, nrhs,
      reinterpret_cast<cuComplex**>(dA_array),
      ldda,
      reinterpret_cast<cuComplex**>(dC_array),
      lddc,
      info,
      devInfoArray,
      batchSize));
}

} // namespace at::cuda::blas
