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
#include <fstream>
#include <cmath>
#include <vector>
#include <cmath>
#include <cstdlib>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDADataType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Export.h>
#include <c10/util/irange.h>

#include <chrono>
using namespace std::chrono;

#include <filesystem>
namespace fs = std::filesystem;

#include "opt_kernels.cu"
// #include "opt_corrector.cu"

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

  // float *dA = const_cast<float*>(a);
  // float *dB = const_cast<float*>(b);

  // printf("a:\n");
  // outputChk(dA, num_batches+1, lda, m*k, m, k);
  // printf("b:\n");
  // outputChk(dB, num_batches+1, ldb, k*n, k, n);
  
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
  
  // printf("additional values for additional allocated\n");
  // memSet<<<1, stridec>>>(c, m*n*num_batches);
  // printf("c:\n");
  // outputChk(c, num_batches+1, ldc, m*n, m, n);
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
void outputChk(T *A, int64_t nb, int64_t ld, int64_t stride, int64_t row, int64_t col){
  size_t size = nb * (row * col) * sizeof(T);
  T *tensor;
  tensor = (T *)malloc(size);
  cudaMemcpy(tensor, A, size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < nb; i++){
    printf("[ \n");
    for(int r = 0; r < row; r++){
      printf("|");
      for(int c = 0; c < col; c++){
        printf("%.6f", T(tensor[i*stride + c*ld + r]));
        printf(", ");
      }
      printf("!\n");
    }
    printf("]\n");
  }
  free(tensor);
}

int64_t CommFactor3(int64_t m,int64_t n){
  int64_t res = n;
  while (m % n != 0)
  {
    res = m % n;
    m = n;
    n = res;
  }
  return res;
}

int64_t closestFactors(int64_t number) {
  int64_t bestFactor1 = 1;
  int64_t bestFactor2 =  number;
  int64_t minDifference = std::abs(bestFactor1 - bestFactor2);
  
  // Iterate up to the square root of the number
  for (int64_t factor = 2; factor <= std::sqrt(number); ++factor) {
      if (number % factor == 0) {
          // Calculate the other factor
          int64_t otherFactor = number / factor;
          // Check if the absolute difference is smaller
          if (std::abs(factor - otherFactor) < minDifference) {
              bestFactor1 = factor;
              bestFactor2 = otherFactor;
              minDifference = std::abs(factor - otherFactor);
          }
      }
  }
  return bestFactor1;
}

void recordEffeciency(string FP, float time, float overhead, float gflop, float memory){
  std::ofstream outFile(FP, std::ios::app);
  if(!outFile){
    std::cerr << "Failed to open the file for appending." << std::endl;
    return;
  }
  outFile << time << " " << overhead << " " << gflop << " " << memory << std::endl;
  // printf("Data appended to the file successfully.\n");
}

void recordTime(string FP, float time, bool DEBUG){
  std::ofstream outFile(FP, std::ios::app);
  if(!outFile){
    std::cerr << "Failed to open the file for appending." << std::endl;
    return;
  }
  outFile << time << std::endl;
  if(DEBUG) printf("Data appended to the file successfully.\n");
}

int64_t getInjPos() {
  const char* homeDir = nullptr;
  homeDir = getenv("HOME");
  // if (homeDir != nullptr) {
  fs::path homePath(homeDir);
  fs::path destinationFile = "abftbgemm/control/pos.txt";
  fs::path fullPath = homePath / destinationFile;
  
  int lineNumber = 0;

  std::ifstream inFile(fullPath);
  std::vector<string> lines;
  string line;

  // Read lines from the file into a vector
  while (getline(inFile, line)) {
      lines.push_back(line);
  }

  inFile.close();

  // Check if lineNumber is valid
  if (lineNumber < 0 || lineNumber >= lines.size()) {
      std::cout << "Invalid line number!" << std::endl;
      return -1;
  }

  // Erase the line at lineNumber
  string deletedLine = lines[lineNumber];
  // cout << "Deleted line: " << deletedLine << endl;

  lines.erase(lines.begin() + lineNumber);

  // Write the modified content back to the file
  std::ofstream outFile(fullPath);
  for (const auto& l : lines) {
      outFile << l << std::endl;
  }
  outFile.close();

  int64_t res = std::stoll(deletedLine);
  return res;
  // cout << "Line at position " << lineNumber << " deleted successfully!" << endl;
}


template<typename T> __device__ T* dA_colchk;
template<typename T> __device__ T* dA_rowchk;
template<typename T> __device__ T* dA_colchk_r;
template<typename T> __device__ T* dA_rowchk_r;

template<typename T> __device__ T* dB_colchk;
template<typename T> __device__ T* dB_rowchk;
template<typename T> __device__ T* dB_colchk_r;
template<typename T> __device__ T* dB_rowchk_r;

template<typename T> __device__ T* dC_colchk;
template<typename T> __device__ T* dC_rowchk;
template<typename T> __device__ T* dC_colchk_r;
template<typename T> __device__ T* dC_rowchk_r;

template<typename T> __device__ T* Q_rowchk;
template<typename T> __device__ T* K_colchk;
template<typename T> __device__ T* K_rowchk;
template<typename T> __device__ T* V_colchk;
template<typename T> __device__ T* tmp_chk;
template<typename T> __device__ T* CL_rowchk;
template<typename T> __device__ T* tmp_colchk;
template<typename T> __device__ T* tmp_rowchk;


__device__ int64_t ldda_colchk;
__device__ int64_t ldda_colchk_r;
__device__ int64_t ldda_rowchk;
__device__ int64_t ldda_rowchk_r;

__device__ int64_t lddb_colchk;
__device__ int64_t lddb_colchk_r;
__device__ int64_t lddb_rowchk;
__device__ int64_t lddb_rowchk_r;

__device__ int64_t lddc_colchk;
__device__ int64_t lddc_colchk_r;
__device__ int64_t lddc_rowchk;
__device__ int64_t lddc_rowchk_r;

// __device__ int64_t ld_chk_v;



template <typename T, int64_t M, int64_t N, int64_t K>
void abftbgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha,  
              T *dA, int64_t ldda, int64_t stridea,                                           
              T *dB, int64_t lddb, int64_t strideb, at::opmath_type<T> beta,             
              T *dC, int64_t lddc, int64_t stridec,                                                 
              T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,                                     
              int64_t num_batches,
              bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, bool ifPassChk, 
              char QKV, int64_t num_head, bool INJECTION, fs::path homePath){
  // std::cout << "Using abftbgemm-T function." << std::endl;
  // See Note [Writing Nondeterministic Operations]
  // std::cout << "globalContext. \n";
  globalContext().alertCuBLASConfigNotDeterministic();
  // std::cout << "handle. \n";
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasHandle_t handle_colchk;
  cublasHandle_t handle_rowchk;
  cublasCreate(&handle_colchk);
  cublasCreate(&handle_rowchk);

  // std::cout << "transA & transB. \n";
  cublasOperation_t transA = _cublasOpFromChar(transa);
  cublasOperation_t transB = _cublasOpFromChar(transb);

  // std::cout << "alpha & beta. \n";
  float falpha = at::opmath_type<T>(1);
  float fbeta = at::opmath_type<T>(0);
  // std::cout << "alpha: " << falpha << "; beta: " << fbeta << std::endl;
  // std::cout << "transa: " << transa << "; transb: " << transb << std::endl;
  // get the col_chk and row_chk of A and B
  // std::cout << "  Get dA_chk: " << std::endl;

  // CUDA Stream
  cudaStream_t stream_main, stream_colchk, stream_rowchk;
  cudaStreamCreate(&stream_main);
  cudaStreamCreate(&stream_colchk);
  cudaStreamCreate(&stream_rowchk);
  cublasSetStream(handle, stream_main);
  cublasSetStream(handle_colchk, stream_colchk);
  cublasSetStream(handle_rowchk, stream_rowchk);

  cudaEvent_t main_compute_done;
  cudaEventCreate(&main_compute_done);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t, t1, t_Achk, t_Bchk;

  fs::path destinationFile, fullPath;

  // clock_t tc_cpu, tc1_cpu, tr_cpu, tr1_cpu; 

  // printf("A: \n");
  // outputChk(dA, num_batches, ldda, m*k, m, k);

  // printf("B: \n");
  // outputChk(dB, num_batches, lddb, k*n, k, n);
  
  // printf("V_colchk: \n");
  // outputChk(V_colchk<T>, num_batches, ldda_colchk, 2*k, 2, k);

  T *A_copy, *B_copy, *C_copy;
  // T *A_copy1, *B_copy1;
  int64_t m_copy = m;
  int64_t n_copy = n;
  int64_t stridea_copy = stridea;
  int64_t strideb_copy = strideb;
  int64_t stridec_copy = stridec;

  if (COL_FT){
    if (DEBUG) std::cout << "dA Col Chk." << std::endl;
    if (DEBUG) {
      cudaEventRecord(start, stream_colchk);
    }
    if (!ifPassChk){
      if (DEBUG) std::cout << "   Navie Calulatation." << std::endl;
      if(transA == CUBLAS_OP_N){
        encode_col_v5<T, M, K, 4><<<num_batches, dim3(M*4, 1), (M+1)*K*sizeof(T), stream_colchk>>>(num_batches,
                      dA, ldda, stridea, 
                      dA_colchk<T>, ldda_colchk, (2*k));
      }
      else{}
    }
    else{
      if (DEBUG) std::cout << "   Pass Chk." << std::endl;
      // tc_cpu = clock();
      if(QKV == 's'){
        dA_colchk<T> = K_colchk<T>;
        // cudaMemcpy(dA_colchk<T>, K_colchk<T>, num_batches*2*k, cudaMemcpyDeviceToDevice);
      }
      else if(QKV == 'v'){
        dA_colchk<T> = V_colchk<T>;
      }
      // tc1_cpu = clock();
      // printf("CPU: %f\n", (tc1_cpu - tc_cpu));
    }
    if (DEBUG) {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Achk, start, stop);
      t_Achk /= 1.0;
      // printf("dA_chk_gemm: %f (%f)(%f)\n", t, (double)num_batches*m*2*k*2/t/1e6, (double)num_batches*(2*k+2*m+k*m)/t/1e6);
    }
    // printf("dA_colchk: \n");
    // outputChk(dA_colchk<T>, num_batches, ldda_colchk, 2*k, 2, k);

    size_t size = (m+2)*k*num_batches*sizeof(T);
    cudaMalloc((void**)&A_copy, size);
    // cudaMalloc((void**)&A_copy1, size);
    stridea_copy = (m+2)*k;
    m_copy += 2;

    // for(int i = 0; i < num_batches; i++){
    //   cudaMemcpy(A_copy+i*stridea_copy, dA+i*stridea, num_batches*stridea_copy*sizeof(T), cudaMemcpyDeviceToDevice);
    //   cudaMemcpy(A_copy+stridea+i*stridea_copy, dA_colchk<T>+i*(2*k), 2*k*sizeof(T), cudaMemcpyDeviceToDevice);
    // }

    for(int i = 0; i < num_batches; i++){
      cublasSetMatrixAsync(m, k, sizeof(T), dA+i*stridea, ldda, A_copy+i*stridea_copy, m_copy, stream_colchk);
      cublasSetMatrixAsync(2, k, sizeof(T), dA_colchk<T>+i*(2*k), ldda_colchk, A_copy+i*stridea_copy+m, m_copy, stream_colchk);
    }

    // BGemmMatrxiChkMerge<<<num_batches, 2>>>(A_copy, dA, dA_colchk<T>, m, k, 2, k);

    // printf("A_copy: \n");
    // outputChk(A_copy, num_batches, ldda+2, stridea_copy, m+2, k);

    // printf("A_copy1: \n");
    // outputChk(A_copy1, num_batches, ldda+2, stridea_copy, m+2, k);
  }

  //std::cout << "  Get dB_chk: " << std::endl;
  if (ROW_FT){
    if (DEBUG) std::cout << "dB Row Chk." << std::endl;
    if (DEBUG) {
      cudaEventRecord(start, stream_rowchk);
    }
    if(!ifPassChk){
      if (DEBUG) std::cout << "   Navie Calulatation." << std::endl;
      if (transB == CUBLAS_OP_N){
        encode_row_v5<T, K, N><<<num_batches, dim3(K*2, 1, 1), 0, stream_rowchk>>>(num_batches,
                      dB, lddb, strideb, 
                      dB_rowchk<T>, lddb_rowchk, (2*k));
      }
      else{}
    }
    else{
      if (DEBUG) std::cout << "   Pass Chk." << std::endl;
      // tr_cpu = clock();
      if(QKV == 's'){
        dB_rowchk<T> = Q_rowchk<T>;
      }
      else if(QKV == 'v'){
        if (DEBUG) printf("      navie\n");
        encode_row_v5<T, K, N><<<num_batches, dim3(K*2, 1, 1), 0, stream_rowchk>>>(num_batches,
                      dB, lddb, strideb, 
                      dB_rowchk<T>, lddb_rowchk, (2*k));
      }
      // tr1_cpu = clock();
      // printf("CPU: %.8f\n", (tr1_cpu - tr_cpu));
    }
    
    if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
      // printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*2*n*k*2/t1/1e6, (double)num_batches*(2*k+k*n+2*n)/t1/1e6);
    }
    // printf("dB_rowchk: \n");
    // outputChk(dB_rowchk<T>, num_batches, lddb_rowchk, 2*k, k, 2);

    size_t size = (n+2)*k*num_batches*sizeof(T);
    cudaMalloc((void**)&B_copy, size);
    strideb_copy = (n+2)*k;
    n_copy += 2;

    for(int i = 0; i < num_batches; i++){
      cudaMemcpy(B_copy+i*strideb_copy, dB+i*strideb, strideb_copy*sizeof(T), cudaMemcpyDeviceToDevice);
      cudaMemcpy(B_copy+strideb+i*strideb_copy, dB_rowchk<T>+i*(2*k), 2*k*sizeof(T), cudaMemcpyDeviceToDevice);
    }
    // printf("B_copy: \n");
    // outputChk(B_copy, num_batches, lddb, strideb_copy, k, n+2);
  }
  stridec_copy = m_copy * n_copy;

  falpha = alpha;
  fbeta = beta;

  // number of row and col of B stored in memory(no trans operation)
  int64_t mem_row = 0;
	int64_t mem_col = 0;

  // --begin-- //
  // C_copy
  size_t size = stridec_copy * num_batches * sizeof(T);
  cudaMalloc((void**)&C_copy, size);
  // calculate check-sum
  std::cout << "-----Begin.------" << std::endl;
  if (DEBUG) cudaEventRecord(start, stream_main);
  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  if constexpr (std::is_same<T, float>::value) {
    // C_copy
    cublasSgemmStridedBatched(
        handle, transA, transB, m_copy, n_copy, k,
        &falpha, A_copy, m_copy, stridea_copy,
        B_copy, lddb, strideb_copy, &fbeta,
        C_copy, m_copy, stridec_copy,
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

  // BGemmResCopyBack<<<1, num_batches, 0, stream_main>>>(dC, C_copy, m, n);

  for(int i = 0; i < num_batches; i++){
      cublasSetMatrixAsync(m, n, sizeof(T), C_copy+i*stridec_copy, m_copy, dC+i*stridec, lddc, stream_main);
  }

  // std::cout << "Output dC: " << std::endl;
  // outputChk(tmpC, num_batches, lddc, m*n, m, n);

  // std::cout << "Output C_copy: " << std::endl;
  // outputChk(C_copy, num_batches, m_copy, stridec_copy, m_copy, n_copy);

  // printf("Before injection C: \n");
  // outputChk(dC, 1, lddc, 0, m, n);

  if(INJECTION){
    if(DEBUG) printf("Injection.\n");
    int64_t pos = getInjPos();
    bitflip<<<1, 1, 0, stream_main>>>(dC, pos);
  }
  
  // printf("After injection C: \n");
  // outputChk(dC, 1, lddc, 0, m, n);
  
  if (DEBUG) {
    cudaEventRecord(stop, stream_main);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);

    destinationFile = "abftbgemm/records/effeciency/abftbgemm.txt";
    fullPath = homePath / destinationFile;
    printf("  gemm: %f (%f)(%f)\n", t, (double)num_batches*m*n*k*2/t/1e6, (double)num_batches*(m*k+k*n+m*n)/t/1e6);
    recordEffeciency(fullPath, t, 1, (double)num_batches*m*n*k*2/t/1e6, (double)num_batches*(m*k+k*n+m*n)/t/1e6);
    
    if(COL_FT){
      printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)num_batches*m*2*k*2/t_Achk/1e6, (double)num_batches*(2*k+2*m+k*m)/t_Achk/1e6);
      recordEffeciency(fullPath, t_Achk, t_Achk/t, 
                                    (double)num_batches*m*2*k*2/t_Achk/1e6, (double)num_batches*(2*k+2*m+k*m)/t_Achk/1e6);
    }
    if(ROW_FT && QKV=='v'){
      printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t_Bchk, t_Bchk/t, (double)num_batches*2*n*k*2/t_Bchk/1e6, (double)num_batches*(2*k+k*n+2*n)/t_Bchk/1e6);
      recordEffeciency(fullPath, t_Bchk, t_Bchk/t, 
                                    (double)num_batches*2*n*k*2/t_Bchk/1e6, (double)num_batches*(2*k+k*n+2*n)/t_Bchk/1e6);
    }
  }

  // BGemmChkCopyBack<<<1, num_batches, 0, stream_colchk>>>(dC_colchk<T>, C_copy, m, n, stridec_copy, true);
  // BGemmChkCopyBack<<<1, num_batches, 0, stream_rowchk>>>(dC_rowchk<T>, C_copy, m, n, stridec_copy, false);
  for(int i = 0; i < num_batches; i++){
    cublasSetMatrixAsync(2, n, sizeof(T), C_copy+i*stridec_copy+m, m_copy, dC_colchk<T>+i*(2*n), lddc_colchk, stream_colchk);
    cublasSetMatrixAsync(m, 2, sizeof(T), C_copy+i*stridec_copy+n*m_copy, m_copy, dC_rowchk<T>+i*(2*m), lddc_rowchk, stream_rowchk);
  }


  if (DEBUG) std::cout << "------Check check-sum-------" << std::endl;
  if (COL_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-col\n");
    if (DEBUG) cudaEventRecord(start, stream_colchk);
    encode_col_v5<T, M, N, 4><<<num_batches, dim3(M*4, 1), (M+1)*N*sizeof(T), stream_colchk>>>(num_batches,
                   dC, lddc, stridec, 
                    dC_colchk_r<T>, lddc_colchk_r, (2*n));

    T E = 1;
    detect_correct_col<T><<<dim3(num_batches), dim3(n), 0, stream_colchk>>>(dC, lddc, E, stridec,
                                            dC_colchk<T>,      lddc_colchk,    (2*n),
                                            dC_colchk_r<T>,    lddc_colchk_r,  (2*n));

    if (DEBUG) {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      destinationFile = "abftbgemm/records/effeciency/abftbgemm.txt";
      fullPath = homePath / destinationFile;

      printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*2*n*m*2/t1/1e6, (double)num_batches*(m*n+2*m+2*n)/t1/1e6);
      recordEffeciency(fullPath,t1, t1/t, (double)(num_batches)*2*n*m*2/t1/1e6, (double)num_batches*(m*n+2*m+2*n)/t1/1e6);
    }
  }

  if (ROW_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-row\n");
    if (DEBUG) cudaEventRecord(start, stream_rowchk);
    encode_row_v5<T, M, N><<<num_batches, dim3(M*2, 1, 1), 0, stream_rowchk>>>(num_batches,
                    dC, lddc, stridec,
                    dC_rowchk_r<T>, lddc_rowchk_r, (2*m));
    T E = 1;
    detect_correct_row<T><<<dim3(num_batches), dim3(m), 0, stream_rowchk>>>(dC, lddc, E, stridec, n,
                                          dC_rowchk<T>, lddc_rowchk,     (2*m),
                                          dC_rowchk_r<T>, lddc_rowchk_r, (2*m));

    if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      
      destinationFile = "abftbgemm/records/effeciency/abftbgemm.txt";
      fullPath = homePath / destinationFile;
      printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*m*2*n*2/t1/1e6, (double)num_batches*(m*n+2*n+2*m)/t1/1e6);
      recordEffeciency(fullPath, t1, t1/t, (double)(num_batches)*m*2*n*2/t1/1e6, (double)num_batches*(m*n+2*n+2*m)/t1/1e6);
    }
  }

  // printf("Final C: \n");
  // outputChk(dC, 1, lddc, 0, m, n);

  // printf("dC_colchk:\n");
  // outputChk(dC_colchk<T>, num_batches, lddc_colchk_r, 2*n, 2, n);
  // printf("dC_rowchk:\n");
  // outputChk(dC_rowchk<T>, num_batches, lddc_rowchk, 2*m, m, 2);

  if(ifPassChk && QKV == 'v'){    
    int64_t nb = int(num_batches/num_head);
    // scale check sum
    ChkSumScale<<<1, num_batches, 0, stream_rowchk>>>(dC_rowchk<T>, m, n, 2*m, num_head);
    // cudaStreamSynchronize(stream_rowchk);
    // printf("After scale dC_rowchk:\n");
    // outputChk(dC_rowchk<T>, num_batches, lddc_rowchk, 2*m, m, 2);
    
    // merage chechk sum
    // MatrixMerge<<<1, dim3(nb, num_head), 0, stream_rowchk>>>(dC_rowchk<T>, tmp_chk<T>, m, 2, m*num_head, 2*nb, num_head);
    // T *t;
    // cudaMalloc((void**)&t, 2*lddc_rowchk*num_batches*sizeof(T));
    for(int c = 0; c < nb; c++){
      for(int r = 0; r < num_head; r++){
        cublasSetMatrixAsync(m, 2, sizeof(T), 
                        dC_rowchk<T>+(r+c*num_head)*(2*m), lddc_rowchk, 
                        tmp_chk<T>+(r*m + c*(m*num_head)*2), lddc_rowchk*num_head, stream_rowchk);
      }
    }
    // printf("tmp chk:\n");
    // outputChk(tmp_chk<T>, 1, lddc_rowchk*num_head, 0, m*num_head, 2*(num_batches/num_head));
    // printf("tmp:\n");
    // outputChk(t, 1, lddc_rowchk*num_head, 0, m*num_head, 2*(num_batches/num_head));
    // sum matrix
    if constexpr (std::is_same<T, float>::value){
      for(int i = 0; i < 2*nb; i+=2){
        cublasSgeam(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_N, m*num_head, 2,
                    &falpha, tmp_chk<T>+i*(lddc_rowchk*num_head), (lddc_rowchk*num_head), &falpha,
                    CL_rowchk<T>, lddc_rowchk*num_head,
                    CL_rowchk<T>, lddc_rowchk*num_head);
      }
    }
    else if constexpr(std::is_same<T, at::Half>::value){
      dim3 blockSize(16, 16);
      dim3 gridSize((2 + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
      for(int i = 0; i < 2*nb; i+=nb){
        addVector<T><<<blockSize, gridSize, (num_head*m*2)*sizeof(T), stream_rowchk>>>(CL_rowchk<T>, tmp_chk<T>+i*(lddc_rowchk*num_head), num_head*m, 2);
      }
    }
    // printf("CL_rowchk:\n");
    // outputChk(CL_rowchk<T>, 1, lddc_rowchk*num_head, 0, m*num_head, 2);
  }

  cudaFree(A_copy);
  cudaFree(B_copy);
  cudaFree(C_copy);
}

template <typename T>
void mybgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha, 
              const T *dA, int64_t ldda, int64_t stridea,                                          
              const T *dB, int64_t lddb, int64_t strideb, at::opmath_type<T> beta,             
              T *dC, int64_t lddc, int64_t stridec,                                                
              int64_t num_batches) {
  // std::cout << "Using mybgemm-T function." << std::endl;

  T *dA_ = const_cast<T*>(dA);
  T *dB_ = const_cast<T*>(dB);
  // std::cout << "C: " << std::endl;
  // outputMatrix(dC, lddc, stridec, num_batches, m, n);
  if(beta == at::opmath_type<T>(0)){
    cudaMemset(dC, 0, (num_batches * m * n * sizeof(T)));
  }

  const char* homeDir = nullptr;
  homeDir = getenv("HOME");
  fs::path homePath(homeDir);
  
  char QKV;
  fs::path destinationFile = "abftbgemm/control/QKV.txt";
  fs::path fullPath = homePath / destinationFile;
  std::ifstream qkvFile(fullPath);

  if(qkvFile.is_open()){
    qkvFile.get(QKV);
  }
  else{
    printf("QKV: Cannot open file, using default setting.\n");
  }
  qkvFile.close();

  std::vector<string> tokens;
  string line, token;
  
  destinationFile = "abftbgemm/control/Batch.txt";
  fullPath = homePath / destinationFile;
  std::ifstream batchFile(fullPath);
  while (std::getline(batchFile, line)) {
      std::istringstream iss(line);

      // Tokenize the line by space
      while (iss >> token) {
          tokens.push_back(token);
      }
  }
  batchFile.close();
  int64_t num_head = std::stoll(tokens[1]);
  
  // std::cout << "_A:" << std::endl;
  // outputMatrix(dA_, ldda, stridea, num_batches, k, n);
  // std::cout << "stridea: " << stridea  << "; lda: " << ldda << std::endl;
  // std::cout << "_B: " << std::endl;
  // outputMatrix(dB_, lddb, strideb, num_batches, m, k);
  // std::cout << "strideb: " << strideb << "; ldb: " << lddb << std::endl;
  std::cout << "num batches: " << num_batches/num_head << "; num heads: " << num_head << std::endl;
  std::cout << "m: " << m << "; n: " << n << "; k: " << k << std::endl;
  // std::cout << "num_batches: " << num_batches << std::endl;

  // std::cout << "leading dimension." << std::endl;
  

  ldda_colchk = 2;
  ldda_colchk_r = 2;
  ldda_rowchk = k;
  ldda_rowchk_r = k;

  lddb_rowchk = k;
  lddb_rowchk_r = k;
  lddb_colchk = 2;
  lddb_colchk_r = 2;

  lddc_colchk = 2;
  lddc_colchk_r = 2;
  lddc_rowchk = m;
  lddc_rowchk_r = m;
  int64_t ld_chk_v = 2;

  //std::cout << "alloc chk vectors" << std::endl;

  // T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
  // T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
  // T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
  T *chk_v_a;
  T *chk_v_b;

  size_t size = (2*num_batches) * k * sizeof(T);
  cudaMalloc((void**)&dA_colchk<T>, size);
  cudaMemset(dA_colchk<T>, 0, size);
  cudaMalloc((void**)&dA_colchk_r<T>, size);
  cudaMemset(dA_colchk_r<T>, 0, size);

  cudaMalloc((void**)&dA_rowchk<T>, size);
  cudaMemset(dA_rowchk<T>, 0, size);
  cudaMalloc((void**)&dA_rowchk_r<T>, size);
  cudaMemset(dA_rowchk_r<T>, 0, size);
  //std::cout << "  finish dA." << std::endl;
  
  cudaMalloc((void**)&dB_colchk<T>, size);
  cudaMemset(dB_colchk<T>, 0, size);
  cudaMalloc((void**)&dB_colchk_r<T>, size);
  cudaMemset(dB_colchk_r<T>, 0, size);
  
  cudaMalloc((void**)&dB_rowchk<T>, size);
  cudaMemset(dB_rowchk<T>, 0, size);
  cudaMalloc((void**)&dB_rowchk_r<T>, size);
  cudaMemset(dB_rowchk_r<T>, 0, size);
  //std::cout << "  finish dB." << std::endl;

  size = (2*num_batches) * n * sizeof(T);
  cudaMalloc((void**)&dC_colchk<T>, size);
  cudaMemset(dC_colchk<T>, 0, size);
  cudaMalloc((void**)&dC_colchk_r<T>, size);
  cudaMemset(dC_colchk_r<T>, 0, size);
  
  size = (2*num_batches) * m * sizeof(T);
  cudaMalloc((void**)&dC_rowchk<T>, size);
  cudaMemset(dC_rowchk<T>, 0, size);
  cudaMalloc((void**)&dC_rowchk_r<T>, size);
  cudaMemset(dC_rowchk_r<T>, 0, size);
  //std::cout << "  finish dC." << std::endl;
  if(QKV == 'v'){
    cudaMalloc((void**)&tmp_chk<T>, size);
    cudaMemset(tmp_chk<T>, 0, size);

    size = num_head * 2 * m * sizeof(T);
    cudaMalloc((void**)&CL_rowchk<T>, size);
    cudaMemset(CL_rowchk<T>, 0, size);
  }

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
  // outputChk(chk_v_a, 1, ld_chk_v, 2*len, 2, len);
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
  // outputChk(chk_v_b, 1, ld_chk_v, 2*len, 2, len);
  free(h_matrix);
  //std::cout << "  finish chk_v." << std::endl;

  bool COL_FT = true;
  bool ROW_FT = true;
  bool DEBUG = true;
  bool CHECK_BEFORE = true;
  bool CHECK_AFTER = true;
  bool ifPassChk = false;
  bool INJECTION = false;

  char flag;
  destinationFile = "abftbgemm/control/abftCOL_FT.txt";
  fullPath = homePath / destinationFile;
  std::ifstream colFile(fullPath);
  if (colFile.is_open()){
    colFile.get(flag);
    if(flag == 'f'){
      COL_FT = false;
    }
    // printf("%c", flag);
  }
  else{
    printf("COL_FT: Cannot open file, using default setting.\n");
  }
  colFile.close();
  
  destinationFile = "abftbgemm/control/abftROW_FT.txt";
  fullPath = homePath / destinationFile;
  std::ifstream rowFile(fullPath);
  if (rowFile.is_open()){
    rowFile.get(flag);
    if(flag == 'f'){
      ROW_FT = false;
    }
    // printf("%c", flag);
  }
  else{
    printf("ROW_FT: Cannot open file, using default setting.\n");
  }
  rowFile.close();

  destinationFile = "abftbgemm/control/IFPassChk.txt";
  fullPath = homePath / destinationFile;
  std::ifstream PassFile(fullPath);
  if(PassFile.is_open()){
    PassFile.get(flag);
    if(flag == 't'){
      ifPassChk = true;
    }
  }
  else{
    printf("PassChksum: Cannot open file, using default setting.\n");
  }
  PassFile.close();

  destinationFile = "abftbgemm/control/Injection.txt";
  fullPath = homePath / destinationFile;
  std::ifstream injFile(fullPath);
  if (injFile.is_open()){
    injFile.get(flag);
    if(flag == 't'){
      INJECTION = true;
    }
    // printf("%c", flag);
  }
  else{
    printf("Injection: Cannot open file, using default setting.\n");
  }
  injFile.close();

  // std::cout << "Calling abftgemm-T function." << std::endl;

  auto start = high_resolution_clock::now();
  if constexpr (std::is_same<T, float>::value) {
    if (m == 72 && k == 64){
      abftbgemm<float, 72, 72, 64>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 64 && k == 72){
      abftbgemm<float, 64, 72, 72>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 71 && k == 64){
      abftbgemm<float, 71, 71, 64>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 64 && k == 71){
      abftbgemm<float, 64, 71, 71>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 70 && k == 64){
      abftbgemm<float, 70, 70, 64>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 64 && k == 70){
      abftbgemm<float, 64, 70, 70>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 75 && k == 64){
      abftbgemm<float, 75, 75, 64>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 64 && k == 75){
      abftbgemm<float, 64, 75, 75>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 5 && k == 4){
      abftbgemm<float, 5, 5, 4>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 4 && k == 5){
      abftbgemm<float, 4, 5, 5>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
  } 
  else if constexpr(std::is_same<T, at::Half>::value) {
      if (m == 72 && k == 64){
        abftbgemm<at::Half, 72, 72, 64>(transa, transb, m, n, k,
          alpha, dA_, ldda, stridea,
          dB_, lddb, strideb, beta,
          dC, lddc, stridec,
          chk_v_a, chk_v_b, ld_chk_v,
          num_batches,
          COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
      }
      else if(m == 64 && k == 72){
        abftbgemm<at::Half, 64, 72, 72>(transa, transb, m, n, k,
          alpha, dA_, ldda, stridea,
          dB_, lddb, strideb, beta,
          dC, lddc, stridec,
          chk_v_a, chk_v_b, ld_chk_v,
          num_batches,
          COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
      }
      else if(m == 70 && k == 64){
      abftbgemm<at::Half, 70, 70, 64>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 64 && k == 70){
      abftbgemm<at::Half, 64, 70, 70>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
      else if(m == 71 && k == 64){
        abftbgemm<at::Half, 71, 71, 64>(transa, transb, m, n, k,
          alpha, dA_, ldda, stridea,
          dB_, lddb, strideb, beta,
          dC, lddc, stridec,
          chk_v_a, chk_v_b, ld_chk_v,
          num_batches,
          COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
      }
      else if(m == 64 && k == 71){
        abftbgemm<at::Half, 64, 71, 71>(transa, transb, m, n, k,
          alpha, dA_, ldda, stridea,
          dB_, lddb, strideb, beta,
          dC, lddc, stridec,
          chk_v_a, chk_v_b, ld_chk_v,
          num_batches,
          COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
      }
      else if(m == 74 && k == 64){
      abftbgemm<at::Half, 74, 74, 64>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
    else if(m == 64 && k == 74){
      abftbgemm<at::Half, 64, 74, 74>(transa, transb, m, n, k,
        alpha, dA_, ldda, stridea,
        dB_, lddb, strideb, beta,
        dC, lddc, stridec,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, ifPassChk, QKV, num_head, INJECTION, homePath);
    }
  }
  cudaDeviceSynchronize();
  auto stop = high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(stop - start);
  std::cout << "abftbgemm: " << duration.count() / 1000.0 << std::endl;
  destinationFile = "abftbgemm/records/time/abftbgemm.txt";
  fullPath = homePath / destinationFile;
  recordTime(fullPath, (duration.count() / 1000.0), DEBUG);

  cudaFree(dA_colchk<T>);
  cudaFree(dA_rowchk<T>);
  cudaFree(dA_colchk_r<T>);
  cudaFree(dA_rowchk_r<T>);
  cudaFree(dB_colchk<T>);
  cudaFree(dB_rowchk<T>);
  cudaFree(dB_colchk_r<T>);
  cudaFree(dB_rowchk_r<T>);
  cudaFree(dC_colchk<T>);
  cudaFree(dC_rowchk<T>);
  cudaFree(dC_colchk_r<T>);
  cudaFree(dC_rowchk_r<T>);
  cudaFree(chk_v_a);
  cudaFree(chk_v_b);

  // if(QKV == 's'){
  //   cudaFree(Q_rowchk<T>);
  //   cudaFree(K_colchk<T>);
  //   cudaFree(K_rowchk<T>);
  // }
  // else if(QKV == 'v'){
  //   cudaFree(V_colchk<T>);
  //   cudaFree(tmp_chk<T>);
  // }
}

template void mybgemm<float>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<float> alpha, 
              const float *dA, int64_t ldda, int64_t stridea,                                          
              const float *dB, int64_t lddb, int64_t strideb, at::opmath_type<float> beta,             
              float *dC, int64_t lddc, int64_t stridec,                                                
              int64_t num_batches);

template void mybgemm<at::Half>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<at::Half> alpha, 
              const at::Half *dA, int64_t ldda, int64_t stridea,                                          
              const at::Half *dB, int64_t lddb, int64_t strideb, at::opmath_type<at::Half> beta,             
              at::Half *dC, int64_t lddc, int64_t stridec,                                                
              int64_t num_batches);

template void mybgemm<double>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<double> alpha, 
              const double *dA, int64_t ldda, int64_t stridea,                                          
              const double*dB, int64_t lddb, int64_t strideb, at::opmath_type<double> beta,             
              double *dC, int64_t lddc, int64_t stridec,                                                
              int64_t num_batches);

template void mybgemm<c10::complex<float>>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<c10::complex<float>> alpha, 
              const c10::complex<float> *dA, int64_t ldda, int64_t stridea,                                          
              const c10::complex<float> *dB, int64_t lddb, int64_t strideb, at::opmath_type<c10::complex<float>> beta,             
              c10::complex<float> *dC, int64_t lddc, int64_t stridec,                                                
              int64_t num_batches);

template void mybgemm<c10::complex<double>>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<c10::complex<double>> alpha, 
              const c10::complex<double> *dA, int64_t ldda, int64_t stridea,                                          
              const c10::complex<double> *dB, int64_t lddb, int64_t strideb, at::opmath_type<c10::complex<double>> beta,             
              c10::complex<double> *dC, int64_t lddc, int64_t stridec,                                                
              int64_t num_batches);

template void mybgemm<at::BFloat16>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<at::BFloat16> alpha, 
              const at::BFloat16 *dA, int64_t ldda, int64_t stridea,                                          
              const at::BFloat16 *dB, int64_t lddb, int64_t strideb, at::opmath_type<at::BFloat16> beta,             
              at::BFloat16 *dC, int64_t lddc, int64_t stridec,                                                
              int64_t num_batches);

template<typename T>
void myGemmPassChk(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha,
            const T *a, int64_t lda, const T *b, int64_t ldb, at::opmath_type<T> beta,
            T *c, int64_t ldc){

  printf("m:%d, n:%d, k:%d\n", m, n, k);
  // printf("%d, %d, %d\n", lda, ldb, ldc);
  const char* homeDir = nullptr;
  homeDir = getenv("HOME");
  fs::path homePath(homeDir);
  fs::path destinationFile = "abftbgemm/control/QKV.txt";
  fs::path fullPath = homePath / destinationFile;

  char QKV;
  std::ifstream qkvFile(fullPath);
  if (qkvFile.is_open()){
    qkvFile.get(QKV);
  }
  else{
    printf("QKV: Cannot open file, using default setting.\n");
  }
  qkvFile.close();

  std::vector<string> tokens;
  string line, token;
  destinationFile = "abftbgemm/control/Batch.txt";
  fullPath = homePath / destinationFile;
  std::ifstream batchFile(fullPath);
  while (std::getline(batchFile, line)) {
      std::istringstream iss(line);

      // Tokenize the line by space
      while (iss >> token) {
          tokens.push_back(token);
      }
  }
  batchFile.close();
  
  int64_t num_batches = std::stoll(tokens[0]);
  int64_t num_head = std::stoll(tokens[1]);

  printf("num_batch: %d, num_head: %d\n", num_batches, num_head);

  T *dA_ = const_cast<T*>(a);
  T *dB_ = const_cast<T*>(b);

  ldda_colchk = 2*num_head;
  ldda_colchk_r = 2*num_head;
  ldda_rowchk = k;
  ldda_rowchk_r = k;

  lddb_rowchk = k;
  lddb_rowchk_r = k;
  lddb_colchk = 2;
  lddb_colchk_r = 2;

  lddc_colchk = 2*num_head;
  lddc_colchk_r = 2*num_head;
  lddc_rowchk = m;
  lddc_rowchk_r = m;
  int64_t ld_chk_v = 2;

  // T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
  // T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
  // T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
  T *chk_v_a;
  T *chk_v_b;

  size_t size = num_head * 2 * k * sizeof(T);
  cudaMalloc((void**)&dA_colchk<T>, size);
  cudaMemset(dA_colchk<T>, 0, size);
  cudaMalloc((void**)&dA_colchk_r<T>, size);
  cudaMemset(dA_colchk_r<T>, 0, size);

  cudaMalloc((void**)&dA_rowchk<T>, size);
  cudaMemset(dA_rowchk<T>, 0, size);
  cudaMalloc((void**)&dA_rowchk_r<T>, size);
  cudaMemset(dA_rowchk_r<T>, 0, size);
  //std::cout << "  finish dA." << std::endl;
  
  size = num_batches * 2 * k * sizeof(T);
  cudaMalloc((void**)&dB_colchk<T>, size);
  cudaMemset(dB_colchk<T>, 0, size);
  cudaMalloc((void**)&dB_colchk_r<T>, size);
  cudaMemset(dB_colchk_r<T>, 0, size);
  
  cudaMalloc((void**)&dB_rowchk<T>, size);
  cudaMemset(dB_rowchk<T>, 0, size);
  cudaMalloc((void**)&dB_rowchk_r<T>, size);
  cudaMemset(dB_rowchk_r<T>, 0, size);
  //std::cout << "  finish dB." << std::endl;

  size = num_head * 2 * n * sizeof(T);
  cudaMalloc((void**)&dC_colchk<T>, size);
  cudaMemset(dC_colchk<T>, 0, size);
  cudaMalloc((void**)&dC_colchk_r<T>, size);
  cudaMemset(dC_colchk_r<T>, 0, size);
  
  if(QKV == 'k'){
    cudaMalloc((void**)&K_colchk<T>, size);
    cudaMemset(K_colchk<T>, 0, size);
  }
  else if (QKV == 'v'){
    cudaMalloc((void**)&V_colchk<T>, size);
    cudaMemset(V_colchk<T>, 0, size);
  }
  
  size = num_batches * 2 * m * sizeof(T);
  cudaMalloc((void**)&dC_rowchk<T>, size);
  cudaMemset(dC_rowchk<T>, 0, size);
  cudaMalloc((void**)&dC_rowchk_r<T>, size);
  cudaMemset(dC_rowchk_r<T>, 0, size);
  
  if(QKV == 'q'){
    cudaMalloc((void**)&Q_rowchk<T>, size);
    cudaMemset(Q_rowchk<T>, 0, size);
  }
  else if(QKV == 'k'){
    cudaMalloc((void**)&K_rowchk<T>, size);
    cudaMemset(K_rowchk<T>, 0, size);
  }
  else{
    // cudaMalloc((void**)&V_rowchk<T>, size);
    // cudaMemset(V_rowchk<T>, 0, size);
  }

  int64_t len = m / num_head;
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
  // outputChk(chk_v_a, 1, ld_chk_v, 2*len, 2, len);
  free(h_matrix);

  len = n / num_batches;
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
  // outputChk(chk_v_b, 1, ld_chk_v, 2*len, len, 2);
  free(h_matrix);
  
  bool COL_FT = true;
  bool ROW_FT = true;
  bool DEBUG = true;
  bool CHECK_BEFORE = true;
  bool CHECK_AFTER = true;
  bool INJECTION = false;

  char flag;
  destinationFile = "abftbgemm/control/abftCOL_FT.txt";
  fullPath = homePath / destinationFile;
  std::ifstream colFile(fullPath);
  if (colFile.is_open()){
    colFile.get(flag);
    if(flag == 'f'){
      COL_FT = false;
    }
    // printf("%c", flag);
  }
  else{
    printf("COL_FT: Cannot open file, using default setting.\n");
  }
  colFile.close();
  
  destinationFile = "abftbgemm/control/abftROW_FT.txt";
  fullPath = homePath / destinationFile;
  std::ifstream rowFile(fullPath);
  if (rowFile.is_open()){
    rowFile.get(flag);
    if(flag == 'f'){
      ROW_FT = false;
    }
    // printf("%c", flag);
  }
  else{
    printf("ROW_FT: Cannot open file, using default setting.\n");
  }
  rowFile.close();

  destinationFile = "abftbgemm/control/Injection.txt";
  fullPath = homePath / destinationFile;
  std::ifstream injFile(fullPath);
  if (injFile.is_open()){
    injFile.get(flag);
    if(flag == 't'){
      INJECTION = true;
    }
    // printf("%c", flag);
  }
  else{
    printf("Injection: Cannot open file, using default setting.\n");
  }
  injFile.close();

  destinationFile = "abftbgemm/records/time/abftgemm.txt";
  fullPath = homePath / destinationFile;

  auto start = high_resolution_clock::now();
  if constexpr (std::is_same<T, float>::value) {
    abftGemmPassChk<float>(transa, transb, m, n, k,
      alpha, dA_, lda,
      dB_, ldb, beta,
      c, ldc,
      chk_v_a, chk_v_b, ld_chk_v,
      num_batches, num_head,
      COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER,QKV,INJECTION, homePath);
  }
  else if constexpr (std::is_same<T, at::Half>::value) {
    abftGemmPassChk<at::Half>(transa, transb, m, n, k,
      alpha, dA_, lda,
      dB_, ldb, beta,
      c, ldc,
      chk_v_a, chk_v_b, ld_chk_v,
      num_batches, num_head,
      COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER,QKV, INJECTION, homePath);
  }
  cudaDeviceSynchronize();
  
  auto stop = high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(stop - start);
  std::cout << "abftGemm: " << duration.count() / 1000.0 << std::endl;
  destinationFile = "abftbgemm/records/time/abftgemm";
  fullPath = homePath / destinationFile;
  recordTime(fullPath, (duration.count() / 1000.0), DEBUG);

  cudaFree(dA_colchk<T>);
  cudaFree(dA_rowchk<T>);
  cudaFree(dA_colchk_r<T>);
  cudaFree(dA_rowchk_r<T>);
  cudaFree(dB_colchk<T>);
  cudaFree(dB_rowchk<T>);
  cudaFree(dB_colchk_r<T>);
  cudaFree(dB_rowchk_r<T>);
  cudaFree(dC_colchk<T>);
  cudaFree(dC_rowchk<T>);
  cudaFree(dC_colchk_r<T>);
  cudaFree(dC_rowchk_r<T>);
  cudaFree(chk_v_a);
  cudaFree(chk_v_b);
}

template void myGemmPassChk<float>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<float> alpha,
            const float *a, int64_t lda, const float *b, int64_t ldb, at::opmath_type<float> beta,
            float *c, int64_t ldc);

template void myGemmPassChk<at::Half>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<at::Half> alpha,
            const at::Half *a, int64_t lda, const at::Half *b, int64_t ldb, at::opmath_type<at::Half> beta,
            at::Half *c, int64_t ldc);

template void myGemmPassChk<double>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<double> alpha,
            const double *a, int64_t lda, const double *b, int64_t ldb, at::opmath_type<double> beta,
            double *c, int64_t ldc);
  
template void myGemmPassChk<c10::complex<double>>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<c10::complex<double>> alpha,
            const c10::complex<double> *a, int64_t lda, const c10::complex<double> *b, int64_t ldb, at::opmath_type<c10::complex<double>> beta,
            c10::complex<double> *c, int64_t ldc);

template void myGemmPassChk<c10::complex<float>>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<c10::complex<float>> alpha,
            const c10::complex<float> *a, int64_t lda, const c10::complex<float> *b, int64_t ldb, at::opmath_type<c10::complex<float>> beta,
            c10::complex<float> *c, int64_t ldc);

template void myGemmPassChk<at::BFloat16>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<at::BFloat16> alpha,
            const at::BFloat16 *a, int64_t lda, const at::BFloat16 *b, int64_t ldb, at::opmath_type<at::BFloat16> beta,
            at::BFloat16 *c, int64_t ldc);

template <typename T>
void abftGemmPassChk(char transa, char transb, int64_t m, int64_t n, int64_t k, 
      at::opmath_type<T> alpha, T *a, int64_t lda, 
      T *b, int64_t ldb, at::opmath_type<T> beta,
      T *c, int64_t ldc,
      T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,
      int64_t num_batches, int64_t num_head,                                      
      bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
      char QKV, bool INJECTION, fs::path homePath){
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  
  cublasHandle_t handle_colchk;
  cublasCreate(&handle_colchk);
  cublasHandle_t handle_rowchk;
  cublasCreate(&handle_rowchk);

  cudaStream_t stream_main, stream_colchk, stream_rowchk;
  cudaStreamCreate(&stream_main);
  cudaStreamCreate(&stream_colchk);
  cudaStreamCreate(&stream_rowchk);
  cublasSetStream(handle, stream_main);
  cublasSetStream(handle_colchk, stream_colchk);
  cublasSetStream(handle_rowchk, stream_rowchk);

  cudaEvent_t main_compute_done;
  cudaEventCreate(&main_compute_done);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t, t1, t_Achk, t_Bchk;

  float falpha = at::opmath_type<T>(1);
  float fbeta = at::opmath_type<T>(0);
  
  fs::path destinationFile, fullPath;
  
  // printf("m:%d, n:%d, k:%d \n", m,n,k);
  // printf("lda: %d, ldb: %d, ldc: %d \n", lda, ldb, ldc);

  // printf("A: \n");
  // outputChk(a, 1, lda, 0, k, m);

  // printf("B: \n");
  // outputChk(b, 1, ldb, 0, k, n);

  int64_t nb = 0;

  T *A_copy, *B_copy;
  T *A_copy1, *B_copy1;
  
  int64_t m_copy = m;
  int64_t n_copy = n;

  int64_t RowOffset = m/num_head;
  int64_t ColOffset = n/num_batches;
  
  //A check col
  if (COL_FT){
    if (DEBUG) std::cout << "dA_chk" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_colchk);
    nb = m / num_head;
    if(opa == CUBLAS_OP_N){
      // printf("A no T\n");
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemmStridedBatched(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, nb,
                                    &falpha, chk_v_a, ld_chk_v, 0,
                                    a, (lda), nb, &fbeta,
                                    dA_colchk_r<T>, ldda_colchk_r, k*2,
                                    num_head);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmStridedBatchedEx(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                                    &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 0,
                                    a, CUDA_R_16F, (lda), nb, &fbeta,
                                    dA_colchk_r<T>, CUDA_R_16F, ldda_colchk_r, k*2,
                                    num_head, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      // printf("dA_colchk: \n");
      // outputChk(dA_colchk<T>, 1, ldda_colchk, 0, 2*num_head, k);
    }
    else{
      // printf("A T\n");
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemmStridedBatched(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                                    &falpha, a, lda, k*nb,
                                    chk_v_a, ld_chk_v, 0, &fbeta,
                                    dA_rowchk_r<T>, ldda_rowchk_r, k*2,
                                    num_head);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmStridedBatchedEx(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                                    &falpha, a, CUDA_R_16F, lda, k*nb,
                                    chk_v_a, CUDA_R_16F, ld_chk_v, 0, &fbeta,
                                    dA_rowchk_r<T>, CUDA_R_16F, ldda_rowchk_r, k*2,
                                    num_head, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      // printf("dA_rowchk: \n");
      // outputChk(dA_rowchk_r<T>, num_head, ldda_rowchk_r, k*2, k, 2);  
      cudaStreamSynchronize(stream_colchk);
      
      // make A and check_sum together
      size_t size = (m + 2*num_head) * k * sizeof(T);
      cudaMalloc((void**)&A_copy, size);
      // cudaMalloc((void**)&A_copy1, size);
      // for(int i = 0; i < num_head; i++){
      //   cudaMemcpy(A_copy+(i*(nb*k+2*k)), a+(i*nb*k), nb*k*sizeof(T), cudaMemcpyDeviceToDevice);
      //   cudaMemcpy(A_copy+(nb*k)+(i*(nb*k+2*k)), dA_rowchk_r<T>+(i*k*2), 2*k*sizeof(T), cudaMemcpyDeviceToDevice);
      // }
      for(int i = 0; i < num_head; i++){
        cublasSetMatrixAsync(k, nb, sizeof(T),
                          a+(i*nb*k), lda,
                          A_copy+(i*(nb*k+2*k)), lda, stream_colchk);
        cublasSetMatrixAsync(k, 2, sizeof(T),
                          dA_rowchk_r<T>+(i*k*2), ldda_rowchk_r,
                          A_copy+(nb*k)+(i*(nb*k+2*k)), lda, stream_colchk);
      }
      // GemmMatrxiChkMerge<<<num_head, 2, 0, stream_colchk>>>(A_copy, a, dA_rowchk_r<T>, k, (m/num_head), k, 2);
      m_copy = (m + 2*num_head);
      RowOffset += 2;
      // printf("A_copy: \n");
      // outputChk(A_copy, 1, (lda), (m+2*num_head)*k, k, (m+2*num_head));  

      // printf("A_copy1: \n");
      // outputChk(A_copy1, 1, (lda), (m+2*num_head)*k, k, (m+2*num_head));  

    }
    // cudaStreamSynchronize(stream_colchk);
    if (DEBUG) {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Achk, start, stop);
    }
  }
  else{
    A_copy = a;
  }

  // B check row
  if (ROW_FT){
    if (DEBUG) std::cout << "dB_chk" << std::endl;
    nb = n / num_batches;
    if (DEBUG) cudaEventRecord(start, stream_rowchk);
    if (opb == CUBLAS_OP_N){
      // printf("B no T\n");
      if constexpr (std::is_same<T, float>::value){
        cublasSgemmStridedBatched(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                                    &falpha, b, ldb, k*nb,
                                    chk_v_b, ld_chk_v, 0, &fbeta,
                                    dB_rowchk_r<T>, lddb_rowchk_r, k*2,
                                    num_batches);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmStridedBatchedEx(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                                    &falpha, b, CUDA_R_16F, ldb, k*nb,
                                    chk_v_b, CUDA_R_16F, ld_chk_v, 0, &fbeta,
                                    dB_rowchk_r<T>, CUDA_R_16F, lddb_rowchk_r, k*2,
                                    num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      // printf("dB_rowchk: \n");
      // outputChk(dB_rowchk_r<T>, num_batches, lddb_rowchk_r, k*2, k, 2);
      
      // make B and check_sum together
      size_t size = (n + 2*num_batches) * k * sizeof(T);
      cudaMalloc((void**)&B_copy, size);
      // cudaMalloc((void**)&B_copy1, size);
      // for(int i = 0; i < num_batches; i++){
      //   cudaMemcpy(B_copy+(i*(nb*k+2*k)), b+(i*nb*k), nb*k*sizeof(T), cudaMemcpyDeviceToDevice);
      //   cudaMemcpy(B_copy+(nb*k)+(i*(nb*k+2*k)), dB_rowchk_r<T>+(i*k*2), 2*k*sizeof(T), cudaMemcpyDeviceToDevice);
      // }
      for(int i = 0; i < num_head; i++){
        cublasSetMatrixAsync(k, nb, sizeof(T),
                          b+(i*nb*k), ldb,
                          B_copy+(i*(nb*k+2*k)), ldb, stream_rowchk);
        cublasSetMatrixAsync(k, 2, sizeof(T),
                          dB_rowchk_r<T>+(i*k*2), lddb_rowchk_r,
                          B_copy+(nb*k)+(i*(nb*k+2*k)), ldb, stream_rowchk);
      }
      // GemmMatrxiChkMerge<<<num_batches, 2, 0, stream_rowchk>>>(B_copy, b, dB_rowchk_r<T>, k, (n/num_batches), k, 2);
      n_copy = (n + 2*num_batches);
      ColOffset += 2;

      // printf("B_copy: \n");
      // outputChk(B_copy, 1, (ldb), (n+2*num_batches)*k, k, (n+2*num_batches));     
      // printf("B_copy1: \n");
      // outputChk(B_copy1, 1, (ldb), (n+2*num_batches)*k, k, (n+2*num_batches));  
    } 
    else{
      // printf("B T\n");
      if constexpr (std::is_same<T, float>::value){
          for(int i=0; i<n; i+=nb){
            cublasSgemm(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, nb, 
                        &falpha, chk_v_b, ld_chk_v, 
                        b+i, ldb, &fbeta, 
                        dB_colchk<T>+(i/nb)*2, lddb_colchk);
          }
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        for(int i=0; i<n; i+=nb){
          cublasGemmEx(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, nb,
                      &falpha, chk_v_b, CUDA_R_16F, ld_chk_v, 
                      b+i, CUDA_R_16F, ldb,
                      &fbeta, dB_colchk<T>+(i/nb)*2, CUDA_R_16F, lddb_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
      }
      // printf("dB_colchk: \n");
      // outputChk(dB_colchk<T>, 1, lddb_colchk, 0, 2*num_batches, k);
    }
    // cudaStreamSynchronize(stream_rowchk);
    if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
    }
  }
  else{
    B_copy = b;
  }

  int64_t mem_row = 0;
  int64_t mem_col = 0;
  falpha = alpha;
  fbeta = beta;

  // A * B
  // allocate a C_copy with col and row chk
  T *C_copy;
  size_t size = m_copy * n_copy * sizeof(T);
  cudaMalloc((void**)&C_copy, size);
  printf("%d, %d\n", m_copy, n_copy);

  if (DEBUG)  cudaEventRecord(start, stream_main);
  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  if constexpr (std::is_same<T, float>::value) {
      // for C_copy
      cublasSgemm(handle, opa, opb, m_copy, n_copy, k, 
                    &alpha, A_copy, lda, 
                    B_copy, ldb, &beta, 
                    C_copy, m_copy);
  } else if constexpr(std::is_same<T, at::Half>::value) {
      cublasGemmEx(handle, opa, opb, m, n, k,
        &alpha, a, CUDA_R_16F, lda, 
        b, CUDA_R_16F, ldb,
        &beta, c, CUDA_R_16F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaStreamSynchronize(stream_main);

  if(INJECTION){
    if(DEBUG) printf("Injection.\n");
    int64_t pos = getInjPos();
    bitflip<<<1,1,0,stream_main>>>(c, pos);
  }
  
  if (DEBUG)  {
    cudaEventRecord(stop, stream_main);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);
    destinationFile = "abftbgemm/records/effeciency/abftgemm.txt";
    fullPath = homePath / destinationFile;
    
    printf("  gemm: %f (%f)(%f)\n", t, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
    recordEffeciency(fullPath, t, 1, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
    if(COL_FT){
      printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", 
              t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
      recordEffeciency(fullPath, 
              t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
    }
    if(ROW_FT){
      printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", 
              t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
      recordEffeciency(fullPath, 
              t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
    }
  }

  // C copy back
  // T *tmpC;
  // cudaMalloc((void**)&tmpC, m*n*sizeof(T));
  // dim3 blocks((m_copy+32-1)/32, (n_copy+32-1)/32);
  // dim3 threads(32, 32);
  // GemmCopyBack_v2<<<blocks, threads, 0, stream_main>>>(C_copy, m_copy, n_copy, num_head, 
  //                                                       c, m/num_head, n/num_batches, 
  //                                                       dC_colchk<T>, COL_FT, dC_rowchk<T>, ROW_FT);
  
  // GemmResCopyBack<<<1, dim3(num_batches, num_head), 0, stream_main>>>(tmpC, C_copy, 
  //                                                                     ldc, m_copy, (m/num_head), (n/num_batches),
  //                                                                     COL_FT, ROW_FT); 
  // printf("tmpC: \n");
  // outputChk(tmpC, 1, ldc, m*n, m, n); 
 
  // for(int col = 0; col < num_batches; col++){
  //   for(int row = 0; row < num_head; row++){
  //     cublasSetMatrixAsync((m/num_head), (n/num_batches), sizeof(T),
  //                     C_copy + row*RowOffset + col*ColOffset*m_copy, m_copy, 
  //                     c + row*(m/num_head) + col*(n/num_batches)*ldc, ldc, stream_main);
  //   }
  // }

  // checkMatrix<<<blocks, threads, 0, stream_main>>>(c, tmpC, ldc, n, 0);

  // printf("c: \n");
  // outputChk(c, 1, ldc, m*n, m, n);

  // printf("C_copy: \n");
  // outputChk(C_copy, 1, m_copy, m_copy*n_copy, m_copy, n_copy);

  // printf("dC_colchk: \n");
  // outputChk(dC_colchk<T>, num_head*num_batches, 2, 2*m/num_head, 2,  m/num_head);

  // printf("dC_rowchk: \n");
  // outputChk(dC_rowchk<T>, num_head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);

  dim3 blocks((n_copy+32-1)/32, (m_copy+32-1)/32);
  dim3 threads(32, 32);
  if(QKV == 'q'){
    GemmCopyBack_v2<<<blocks, threads, 0, stream_main>>>(C_copy, m_copy, n_copy, num_head, 
                                                          c, m/num_head, n/num_batches, 
                                                          dC_colchk<T>, COL_FT, Q_rowchk<T>, ROW_FT);
    // printf("c: \n");
    // outputChk(c, 1, ldc, m*n, m, n);
    
    // GemmChkCopyBack<<<1, dim3(num_batches, num_head)>>>(Q_rowchk<T>, C_copy, m_copy, 
    //                                                     (m/num_head), 2, (m/num_head), (n/num_batches), 
    //                                                     num_head, false, COL_FT, ROW_FT);
    // for(int c = 0; c < num_batches; c++){  
    //   for(int r = 0; r < num_head; r++){
    //     cublasSetMatrixAsync((m/num_head), 2, sizeof(T),
    //                     C_copy + (r*RowOffset + c*ColOffset*m_copy) + (n/num_batches)*m_copy, m_copy, 
    //                     Q_rowchk<T> + ((m/num_head)*2)*(r+c*num_head) , m/num_head, stream_rowchk);
    //   }
    // }
    // printf("Q_rowchk: \n");
    // outputChk(Q_rowchk<T>, num_head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
    // printf("dC_rowchk: \n");
    // outputChk(dC_rowchk<T>, num_head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
  }
  else if(QKV == 'k'){
    // K_rowchk<T> = dC_rowchk<T>;
    GemmCopyBack_v2<<<blocks, threads, 0, stream_main>>>(C_copy, m_copy, n_copy, num_head, 
                                                          c, m/num_head, n/num_batches, 
                                                          dC_colchk<T>, COL_FT, K_rowchk<T>, ROW_FT);
    
    // GemmChkCopyBack<<<1, dim3(num_batches, num_head)>>>(K_rowchk<T>, C_copy, m_copy, 
    //                                                     (m/num_head), 2, (m/num_head), (n/num_batches), 
    //                                                     num_head, false, COL_FT, ROW_FT);
    // for(int c = 0; c < num_batches; c++){  
    //   for(int r = 0; r < num_head; r++){
    //     cublasSetMatrixAsync((m/num_head), 2, sizeof(T),
    //                     C_copy + (r*RowOffset + c*ColOffset*m_copy) + (n/num_batches)*m_copy, m_copy, 
    //                     K_rowchk<T> + ((m/num_head)*2)*(r+c*num_head) , m/num_head, stream_rowchk);
    //   }
    // }
    MatrixTranspose<<<1, num_head*num_batches>>>(K_rowchk<T>, K_colchk<T>, m/num_head, 2);
    // printf("K_colchk: \n");
    // outputChk(K_colchk<T>, num_head*num_batches, 2, 2*m/num_head, 2,  m/num_head);
    // printf("dC_rowchk: \n");
    // outputChk(dC_rowchk<T>, num_head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
  }
  else{
    // V_colchk<T> = dC_colchk<T>;
    GemmCopyBack_v2<<<blocks, threads, 0, stream_main>>>(C_copy, m_copy, n_copy, num_head, 
                                                          c, m/num_head, n/num_batches, 
                                                          V_colchk<T>, COL_FT, dC_rowchk<T>, ROW_FT);
    
    // GemmChkCopyBack<<<1, dim3(num_batches, num_head)>>>(V_colchk<T>, C_copy, m_copy, 
    //                                                     2, (n/num_batches), (m/num_head), (n/num_batches), 
    //                                                     num_head, true, COL_FT, ROW_FT);
    // for(int c = 0; c < num_batches; c++){  
    //   for(int r = 0; r < num_head; r++){
    //     cublasSetMatrixAsync(2, (n/num_batches), sizeof(T),
    //                     C_copy + (r*RowOffset + c*ColOffset*m_copy) + (m/num_head), m_copy, 
    //                     V_colchk<T> + ((n/num_batches)*2)*(r+c*num_head) , 2, stream_colchk);
    //   }
    // }
    // printf("V_colchk: \n");
    // outputChk(V_colchk<T>, num_head*num_batches, 2, 2*n/num_batches, 2, n/num_batches);
    // printf("dC_colchk: \n");
    // outputChk(dC_colchk<T>, num_head*num_batches, 2, 2*n/num_batches, 2, n/num_batches);
  }
}

template<typename T>
void myGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha,
            const T *a, int64_t lda, const T *b, int64_t ldb, at::opmath_type<T> beta,
            T *c, int64_t ldc){

  printf("m:%d, n:%d, k:%d\n", m, n, k);

  const char* homeDir = nullptr;
  homeDir = getenv("HOME");
  fs::path homePath(homeDir);
  fs::path destinationFile = "abftbgemm/control/QKV.txt";
  fs::path fullPath = homePath / destinationFile;
  
  char QKV;
  std::ifstream qkvFile(fullPath);
  if (qkvFile.is_open()){
    qkvFile.get(QKV);
  }
  else{
    printf("QKV: Cannot open file, using default setting.\n");
  }
  qkvFile.close();

  T *dA_ = const_cast<T*>(a);
  T *dB_ = const_cast<T*>(b);

  ldda_colchk = 2;
  ldda_colchk_r = 2;
  ldda_rowchk = k;
  ldda_rowchk_r = k;

  lddb_rowchk = k;
  lddb_rowchk_r = k;
  lddb_colchk = 2;
  lddb_colchk_r = 2;

  lddc_colchk = 2;
  lddc_colchk_r = 2;
  lddc_rowchk = m;
  lddc_rowchk_r = m;
  int64_t ld_chk_v = 2;

  // T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
  // T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
  // T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
  T *chk_v_a;
  T *chk_v_b;

  size_t size = 2 * k * sizeof(T);
  cudaMalloc((void**)&dA_colchk<T>, size);
  cudaMemset(dA_colchk<T>, 0, size);
  cudaMalloc((void**)&dA_colchk_r<T>, size);
  cudaMemset(dA_colchk_r<T>, 0, size);

  cudaMalloc((void**)&dA_rowchk<T>, size);
  cudaMemset(dA_rowchk<T>, 0, size);
  cudaMalloc((void**)&dA_rowchk_r<T>, size);
  cudaMemset(dA_rowchk_r<T>, 0, size);
  //std::cout << "  finish dA." << std::endl;
  
  cudaMalloc((void**)&dB_colchk<T>, size);
  cudaMemset(dB_colchk<T>, 0, size);
  cudaMalloc((void**)&dB_colchk_r<T>, size);
  cudaMemset(dB_colchk_r<T>, 0, size);
  
  cudaMalloc((void**)&dB_rowchk<T>, size);
  cudaMemset(dB_rowchk<T>, 0, size);
  cudaMalloc((void**)&dB_rowchk_r<T>, size);
  cudaMemset(dB_rowchk_r<T>, 0, size);
  //std::cout << "  finish dB." << std::endl;

  size = 2 * n * sizeof(T);
  cudaMalloc((void**)&dC_colchk<T>, size);
  cudaMemset(dC_colchk<T>, 0, size);
  cudaMalloc((void**)&dC_colchk_r<T>, size);
  cudaMemset(dC_colchk_r<T>, 0, size);
  
  size = 2 * m * sizeof(T);
  cudaMalloc((void**)&dC_rowchk<T>, size);
  cudaMemset(dC_rowchk<T>, 0, size);
  cudaMalloc((void**)&dC_rowchk_r<T>, size);
  cudaMemset(dC_rowchk_r<T>, 0, size);

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
  // outputChk(chk_v_a, 1, ld_chk_v, 2*len, 2, len);
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
  // outputChk(chk_v_b, 1, ld_chk_v, 2*len, len, 2);
  free(h_matrix);
  
  bool COL_FT = true;
  bool ROW_FT = true;
  bool DEBUG = true;
  bool CHECK_BEFORE = true;
  bool CHECK_AFTER = true;
  // bool ifPassChk = false;
  bool INJECTION = false;

  destinationFile = "abftbgemm/control/abftCOL_FT.txt";
  fullPath = homePath / destinationFile;
  char flag;
  std::ifstream colFile(fullPath);
  if (colFile.is_open()){
    colFile.get(flag);
    if(flag == 'f'){
      COL_FT = false;
    }
    // printf("%c", flag);
  }
  else{
    printf("COL_FT: Cannot open file, using default setting.\n");
  }
  colFile.close();
  
  destinationFile = "abftbgemm/control/abftROW_FT.txt";
  fullPath = homePath / destinationFile;
  std::ifstream rowFile(fullPath);
  if (rowFile.is_open()){
    rowFile.get(flag);
    if(flag == 'f'){
      ROW_FT = false;
    }
    // printf("%c", flag);
  }
  else{
    printf("ROW_FT: Cannot open file, using default setting.\n");
  }
  rowFile.close();

  destinationFile = "abftbgemm/control/Injection.txt";
  fullPath = homePath / destinationFile;
  std::ifstream injFile(fullPath);
  if (injFile.is_open()){
    injFile.get(flag);
    if(flag == 't'){
      INJECTION = true;
    }
    // printf("%c", flag);
  }
  else{
    printf("Injection: Cannot open file, using default setting.\n");
  }
  injFile.close();
  
  auto start = high_resolution_clock::now();
  if constexpr (std::is_same<T, float>::value) {
    abftGemm<float>(transa, transb, m, n, k,
      alpha, dA_, lda,
      dB_, ldb, beta,
      c, ldc,
      chk_v_a, chk_v_b, ld_chk_v,
      COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, QKV, INJECTION, homePath);
  }
  else if constexpr (std::is_same<T, at::Half>::value) {
    abftGemm<at::Half>(transa, transb, m, n, k,
      alpha, dA_, lda,
      dB_, ldb, beta,
      c, ldc,
      chk_v_a, chk_v_b, ld_chk_v,
      COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, QKV, INJECTION, homePath);
  }
  cudaDeviceSynchronize();
  auto stop = high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(stop - start);
  std::cout << "abftGemm: " << duration.count() / 1000.0 << std::endl;
  
  destinationFile = "abftbgemm/records/time/abftgemm.txt";
  fullPath = homePath / destinationFile;
  recordTime(fullPath, (duration.count() / 1000.0), DEBUG);

  cudaFree(dA_colchk<T>);
  cudaFree(dA_rowchk<T>);
  cudaFree(dA_colchk_r<T>);
  cudaFree(dA_rowchk_r<T>);
  cudaFree(dB_colchk<T>);
  cudaFree(dB_rowchk<T>);
  cudaFree(dB_colchk_r<T>);
  cudaFree(dB_rowchk_r<T>);
  cudaFree(dC_colchk<T>);
  cudaFree(dC_rowchk<T>);
  cudaFree(dC_colchk_r<T>);
  cudaFree(dC_rowchk_r<T>);
  cudaFree(chk_v_a);
  cudaFree(chk_v_b);
}

template void myGemm<float>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<float> alpha,
            const float *a, int64_t lda, const float *b, int64_t ldb, at::opmath_type<float> beta,
            float *c, int64_t ldc);

template void myGemm<at::Half>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<at::Half> alpha,
            const at::Half *a, int64_t lda, const at::Half *b, int64_t ldb, at::opmath_type<at::Half> beta,
            at::Half *c, int64_t ldc);

template void myGemm<double>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<double> alpha,
            const double *a, int64_t lda, const double *b, int64_t ldb, at::opmath_type<double> beta,
            double *c, int64_t ldc);
  
template void myGemm<c10::complex<double>>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<c10::complex<double>> alpha,
            const c10::complex<double> *a, int64_t lda, const c10::complex<double> *b, int64_t ldb, at::opmath_type<c10::complex<double>> beta,
            c10::complex<double> *c, int64_t ldc);

template void myGemm<c10::complex<float>>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<c10::complex<float>> alpha,
            const c10::complex<float> *a, int64_t lda, const c10::complex<float> *b, int64_t ldb, at::opmath_type<c10::complex<float>> beta,
            c10::complex<float> *c, int64_t ldc);

template void myGemm<at::BFloat16>(char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<at::BFloat16> alpha,
            const at::BFloat16 *a, int64_t lda, const at::BFloat16 *b, int64_t ldb, at::opmath_type<at::BFloat16> beta,
            at::BFloat16 *c, int64_t ldc);

template <typename T>
void abftGemm(char transa, char transb, int64_t m, int64_t n, int64_t k, 
      at::opmath_type<T> alpha, T *a, int64_t lda, 
      T *b, int64_t ldb, at::opmath_type<T> beta,
      T *c, int64_t ldc,
      T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,                                      
      bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
      char QKV, bool INJECTION, fs::path homePath){
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  
  cublasHandle_t handle_colchk;
  cublasCreate(&handle_colchk);
  cublasHandle_t handle_rowchk;
  cublasCreate(&handle_rowchk);

  cudaStream_t stream_main, stream_colchk, stream_rowchk;
  cudaStreamCreate(&stream_main);
  cudaStreamCreate(&stream_colchk);
  cudaStreamCreate(&stream_rowchk);
  cublasSetStream(handle, stream_main);
  cublasSetStream(handle_colchk, stream_colchk);
  cublasSetStream(handle_rowchk, stream_rowchk);

  cudaEvent_t main_compute_done;
  cudaEventCreate(&main_compute_done);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t, t1, t_Achk, t_Bchk;
  fs::path destinationFile, fullPath;

  float falpha = at::opmath_type<T>(1);
  float fbeta = at::opmath_type<T>(0);

  printf("m:%d, n:%d, k:%d \n", m,n,k);
  // printf("lda: %d, ldb: %d, ldc: %c\n", lda, ldb, ldc);

  //A check col
  if (COL_FT){
    if (DEBUG) std::cout << "dA_colchk" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_colchk);
    if(opa == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemm(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m, 
                      &falpha, chk_v_a, ld_chk_v, 
                      a, lda, &fbeta, 
                      dA_colchk<T>, ldda_colchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmEx(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      a, CUDA_R_16F, lda,
                      &fbeta, dA_colchk<T>, CUDA_R_16F, ldda_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      // printf("dA_colchk: \n");
      // outputChk(dA_colchk<T>, 1, ldda_colchk, 0, k, 2);   
    }
    else{
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemm(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, m, 
                      &falpha, a, lda, 
                      chk_v_a, ld_chk_v, &fbeta, 
                      dA_rowchk<T>, ldda_rowchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmEx(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, k,2,m,
                      &falpha, a, CUDA_R_16F, lda, 
                      chk_v_a, CUDA_R_16F, ld_chk_v,
                      &fbeta, dA_rowchk<T>, CUDA_R_16F, ldda_rowchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } 
      // printf("dA_rowchk: \n");
      // outputChk(dA_rowchk<T>, 1, ldda_rowchk, 0, k, 2);  
    }
    cudaStreamSynchronize(stream_colchk);
    if (DEBUG) {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Achk, start, stop);
    }
  }

  // B check row

  // printf("CL_rowchk: \n");
  // outputChk(CL_rowchk<T>, 1, lddb_rowchk, 0, k, 2);  

  if (ROW_FT){
    if (DEBUG) std::cout << "dB_rowchk" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_rowchk);
    if(QKV != 'c'){
      if (opb == CUBLAS_OP_N){
        printf("  Navie Calculate.\n");
        if constexpr (std::is_same<T, float>::value){
          cublasSgemm(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n, 
                      &falpha, b, ldb, 
                      chk_v_b, ld_chk_v, &fbeta, 
                      dB_rowchk<T>, lddb_rowchk);
        }
        else if constexpr(std::is_same<T, at::Half>::value){
          cublasGemmEx(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n,
                        &falpha, b, CUDA_R_16F, ldb, 
                        chk_v_b, CUDA_R_16F, ld_chk_v,
                        &fbeta, dB_rowchk<T>, CUDA_R_16F, lddb_rowchk,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // printf("dB_rowchk: \n");
        // outputChk(dB_rowchk<T>, 1, lddb_rowchk, 0, k, 2);  
      } 
      else{
        if constexpr (std::is_same<T, float>::value){
            cublasSgemm(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, n, 
                        &falpha, chk_v_b, ld_chk_v, 
                        b, ldb, &fbeta, 
                        dB_colchk<T>, lddb_colchk);
          }
        else if constexpr(std::is_same<T, at::Half>::value){
            cublasGemmEx(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, n,
                          &falpha, chk_v_b, CUDA_R_16F, ld_chk_v, 
                          b, CUDA_R_16F, ldb,
                          &fbeta, dB_colchk<T>, CUDA_R_16F, lddb_colchk,
                          CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // printf("dB_colchk: \n");
        // outputChk(dB_colchk<T>, 1, lddb_rowchk, 0, 2, k);  
      }
    }
    else{
      printf("  Pass Chk.\n");
      dB_rowchk<T> = CL_rowchk<T>;
      // printf("dB_rowchk: \n");
      // outputChk(dB_rowchk<T>, 1, lddb_rowchk, 0, k, 2);  
    }
    cudaStreamSynchronize(stream_rowchk);
    if (DEBUG) {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
    }
  }
  
  int64_t mem_row = 0;
  int64_t mem_col = 0;
  falpha = alpha;
  fbeta = beta;


  // A * B
  if (DEBUG)  cudaEventRecord(start, stream_main);
  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  if constexpr (std::is_same<T, float>::value) {
      cublasSgemm(handle, opa, opb, m, n, k, 
                    &alpha, a, lda, 
                    b, ldb, &beta, 
                    c, ldc);
  } else if constexpr(std::is_same<T, at::Half>::value) {
      cublasGemmEx(handle, opa, opb, m, n, k,
        &alpha, a, CUDA_R_16F, lda, 
        b, CUDA_R_16F, ldb,
        &beta, c, CUDA_R_16F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaStreamSynchronize(stream_main);
  
  // printf("Before injection C: \n");
  // outputChk(c, 1, ldc, 0, m, n);

  if(INJECTION){
    if(DEBUG) printf("Injection.\n");
    int64_t pos = getInjPos();
    bitflip<<<1, 1, 0, stream_main>>>(c, pos);
  }

  // printf("After injection C: \n");
  // outputChk(c, 1, ldc, 0, m, n);
  
  
  if (DEBUG)  {
    cudaEventRecord(stop, stream_main);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);
    destinationFile = "abftbgemm/records/effeciency/abftgemm.txt";
    fullPath = homePath / destinationFile;

    printf("  gemm: %f (%f)(%f)\n", t, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
    recordEffeciency(fullPath, t, 1, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
    if(COL_FT){
      printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
      recordEffeciency(fullPath, t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
    }
    if(ROW_FT){
      printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
      recordEffeciency(fullPath, t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
    }
  }


  if(COL_FT){
    if (DEBUG)  cudaEventRecord(start, stream_colchk);
    //std::cout << "  COL_FT" << std::endl;
    if (opa == CUBLAS_OP_N) {
      if (DEBUG) std::cout << "dA_colchk * dB = dC_colchk" << std::endl;
      // K*4 must be greater then 2 * N
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle_colchk, opa, opb, 2, n, k,
                    &falpha, dA_colchk<T>, ldda_colchk, 
                    b, ldb, &fbeta, 
                    dC_colchk<T>, lddc_colchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmEx(handle_colchk, opa, opb, 2, n, k,
                      &falpha, dA_colchk<T>, CUDA_R_16F, ldda_colchk, 
                      b, CUDA_R_16F, ldb,
                      &fbeta, dC_colchk<T>, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    else{
      if (DEBUG) std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle_colchk, opa, opb, 2, n, k,
                    &falpha, dA_rowchk<T>, ldda_rowchk, 
                    b, ldb, &fbeta, 
                    dC_colchk<T>, lddc_colchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmEx(handle_colchk, opa, opb, 2, n, k,
                      &falpha, dA_rowchk<T>, CUDA_R_16F, ldda_rowchk, 
                      b, CUDA_R_16F, ldb,
                      &fbeta, dC_colchk<T>, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    cudaStreamSynchronize(stream_colchk);
    // std::cout << "Output dC_colchk: " << std::endl;
    // outputMatrixChk(dC_colchk, ldda_colchk, n*2, num_batches, 2, n);
    if (DEBUG)  {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      destinationFile = "abftbgemm/records/effeciency/abftgemm.txt";
      fullPath = homePath / destinationFile;
      printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*2*n*k*2/t1/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath,  t1, t1/t, (double)1*2*n*k*2/t1/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t1/1e6);
    }
  }

  if (ROW_FT) {
    if (DEBUG)  cudaEventRecord(start, stream_rowchk);
    //std::cout << "  ROW_FT" << std::endl;
    if (opb == CUBLAS_OP_N) {
      if (DEBUG) std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
      //we can further work on this to support trans A
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle_rowchk, opa, opb,  m, 2, k,
                    &alpha, a, lda, 
                    dB_rowchk<T>, lddb_rowchk, &beta, 
                    dC_rowchk<T>, lddc_rowchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmEx(handle_rowchk, opa, opb,  m, 2, k,
                      &falpha, a, CUDA_R_16F, lda, 
                      dB_rowchk<T>, CUDA_R_16F, lddb_rowchk,
                      &fbeta, dC_rowchk<T>, CUDA_R_16F, lddc_rowchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    } 
    else{
      if (DEBUG) std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle_rowchk, opa, opb,  m, 2, k,
                    &alpha, a, lda, 
                    dB_colchk<T>, lddb_colchk, &beta, 
                    dC_rowchk<T>, lddc_rowchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmEx(handle_rowchk, opa, opb,  m, 2, k,
                      &falpha, a, CUDA_R_16F, lda, 
                      dB_colchk<T>, CUDA_R_16F, lddb_colchk,
                      &fbeta, dC_rowchk<T>, CUDA_R_16F, lddc_rowchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    cudaStreamSynchronize(stream_rowchk);
    // std::cout << "Output dC_rowchk: " << std::endl;
    // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
    if (DEBUG)  {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);

      destinationFile = "abftbgemm/records/effeciency/abftgemm.txt";
      fullPath = homePath / destinationFile;
      printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*m*2*k*2/t1/1e6, (double)1*(m*k+k*2+m*2)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath, t1, t1/t, (double)1*m*2*k*2/t1/1e6, (double)1*(m*k+k*2+m*2)*sizeof(T)/t1/1e6);      
    }
  }
    
  // --- check check-sum of C---//

  if (DEBUG) std::cout << "------Check check-sum-------" << std::endl;
  falpha = at::opmath_type<T>(1);
  fbeta = at::opmath_type<T>(0);
  int threadNum = ((n+128-1)/128);

  if (COL_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-col\n");
    if (DEBUG) cudaEventRecord(start, stream_colchk);
    if constexpr (std::is_same<T, float>::value){
      cublasSgemm(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &falpha, chk_v_a, ld_chk_v, 
                      c, ldc, &fbeta, 
                      dC_colchk_r<T>, lddc_colchk_r);
    }
    else if constexpr(std::is_same<T, at::Half>::value){
      cublasGemmEx(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      c, CUDA_R_16F, ldc,
                      &fbeta, dC_colchk_r<T>, CUDA_R_16F, lddc_colchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    T E = 1;
    detect_correct_col_Gemm<T><<<dim3(128), dim3(threadNum), 0, stream_colchk>>>(c ,ldc, E, n,
                                                                                      dC_colchk<T>, lddc_colchk,
                                                                                      dC_colchk_r<T>, lddc_colchk_r);

    // cudaStreamSynchronize(stream_colchk);

    if (DEBUG)  {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);

      destinationFile = "abftbgemm/records/effeciency/abftgemm.txt";
      fullPath = homePath / destinationFile;
      printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(1)*2*n*m*2/t1/1e6, (double)1*(m*n+2*m+2*n)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath,  t1, t1/t, (double)(1)*2*n*m*2/t1/1e6, (double)1*(m*n+2*m+2*n)*sizeof(T)/t1/1e6);      
    }
  }

  
  if (ROW_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    
    if (DEBUG) printf("dgemm-after-check-C-row\n");
    if (DEBUG)  cudaEventRecord(start, stream_rowchk);
    if constexpr (std::is_same<T, float>::value){
      cublasSgemm(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &falpha, c, ldc, 
                      chk_v_b, ld_chk_v, &fbeta, 
                      dC_rowchk_r<T>, lddc_rowchk_r);
    }
    else if constexpr(std::is_same<T, at::Half>::value){
      cublasGemmEx(handle_rowchk, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &falpha, c, CUDA_R_16F, ldc, 
                      chk_v_b, CUDA_R_16F, ld_chk_v,
                      &fbeta, dC_rowchk_r<T>, CUDA_R_16F, lddc_rowchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    T E = 1;
    threadNum = (m+128-1)/128;
    detect_correct_row_Gemm<T><<<dim3(128), dim3(threadNum), 0, stream_rowchk>>>(c, ldc, E, m, n,
                                                                          dC_rowchk<T>, lddc_rowchk,
                                                                          dC_rowchk_r<T>, lddc_rowchk_r);
  
    cudaStreamSynchronize(stream_rowchk);
    if (DEBUG)  {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);

      destinationFile = "abftbgemm/records/effeciency/abftgemm.txt";
      fullPath = homePath / destinationFile;
      printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(1)*m*2*n*2/t1/1e6, (double)1*(m*n+2*n+2*m)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath,  t1, t1/t, (double)(1)*m*2*n*2/t1/1e6, (double)1*(m*n+2*n+2*m)*sizeof(T)/t1/1e6);      
    }
  }
  // printf("Final C: \n");
  // outputChk(c, 1, ldc, 0, m, n);
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
void myGemmBiasPassChk (
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

    T *mat1_ptr_ = const_cast<T*>(mat1_ptr);
    T *mat2_ptr_ = const_cast<T*>(mat2_ptr);
    T *bias_ = const_cast<T*>(bias);

    const char* homeDir = nullptr;
    homeDir = getenv("HOME");
    fs::path homePath(homeDir);
    fs::path destinationFile = "abftbgemm/control/QKV.txt";
    fs::path fullPath = homePath / destinationFile;

    char QKV;
    std::ifstream qkvFile(fullPath);
    if (qkvFile.is_open()){
      qkvFile.get(QKV);
    }
    else{
      printf("QKV: Cannot open file, using default setting.\n");
    }
    qkvFile.close();

    bool Together = false;
    char flag;
    destinationFile = "abftbgemm/control/together.txt";
    fullPath = homePath / destinationFile;
    std::ifstream togetherFile(fullPath);
    if (togetherFile.is_open()){
      togetherFile.get(flag);
      if(flag == 't'){
        Together = true;
      }
    }
    else{
      printf("Together: Cannot open file, using default setting.\n");
    }
    togetherFile.close();

    std::vector<string> tokens;
    string line, token;
    destinationFile = "abftbgemm/control/Batch.txt";
    fullPath = homePath / destinationFile;
    std::ifstream batchFile(fullPath);
    while (std::getline(batchFile, line)) {
        std::istringstream iss(line);

        // Tokenize the line by space
        while (iss >> token) {
            tokens.push_back(token);
        }
    }
    batchFile.close();
    
    int64_t num_batches = std::stoll(tokens[0]);
    // for gpt-neo, bert, roberta
    int64_t num_head = std::stoll(tokens[1]);
    // for gpt-2
    if(Together){
      num_head = 3 * num_head;
    }

    printf("num_batch: %d, num_head: %d\n", num_batches, num_head);
    // printf("lda: %d, ldb: %d, ldc: %d\n", mat1_ld, mat2_ld, result_ld);


    // printf("m:%d, n:%d, k:%d\n", m, n, k);
    // printf("%d, %d, %d\n", mat1_ld, mat2_ld, result_ld);

    ldda_colchk = 2*num_head;
    ldda_colchk_r = 2*num_head;
    ldda_rowchk = k;
    ldda_rowchk_r = k;

    lddb_rowchk = k;
    lddb_rowchk_r = k;
    lddb_colchk = 2;
    lddb_colchk_r = 2;

    lddc_colchk = 2*num_head;
    lddc_colchk_r = 2*num_head;
    lddc_rowchk = m;
    lddc_rowchk_r = m;
    int64_t ld_chk_v = 2;

    // T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
    // T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
    // T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
    T *dBias_colchk, *dBias_rowchk, *dBias_colchk_r,*dBias_rowchk_r;
    T *chk_v_a;
    T *chk_v_b;
  
    size_t size = num_head * 2 * k * sizeof(T);
    cudaMalloc((void**)&dA_colchk<T>, size);
    cudaMemset(dA_colchk<T>, 0, size);
    cudaMalloc((void**)&dA_colchk_r<T>, size);
    cudaMemset(dA_colchk_r<T>, 0, size);

    cudaMalloc((void**)&dA_rowchk<T>, size);
    cudaMemset(dA_rowchk<T>, 0, size);
    cudaMalloc((void**)&dA_rowchk_r<T>, size);
    cudaMemset(dA_rowchk_r<T>, 0, size);
    // std::cout << "  finish dA." << std::endl;
    
    size = num_batches * 2 * k * sizeof(T);
    cudaMalloc((void**)&dB_colchk<T>, size);
    cudaMemset(dB_colchk<T>, 0, size);
    cudaMalloc((void**)&dB_colchk_r<T>, size);
    cudaMemset(dB_colchk_r<T>, 0, size);
    
    cudaMalloc((void**)&dB_rowchk<T>, size);
    cudaMemset(dB_rowchk<T>, 0, size);
    cudaMalloc((void**)&dB_rowchk_r<T>, size);
    cudaMemset(dB_rowchk_r<T>, 0, size);
    // std::cout << "  finish dB." << std::endl;

    size = num_head * 2 * n * sizeof(T);
    cudaMalloc((void**)&dC_colchk<T>, size);
    cudaMemset(dC_colchk<T>, 0, size);
    cudaMalloc((void**)&dC_colchk_r<T>, size);
    cudaMemset(dC_colchk_r<T>, 0, size);

    cudaMalloc((void**)&dBias_colchk, size);
    cudaMemset(dBias_colchk, 0, size);
    cudaMalloc((void**)&dBias_colchk_r, size);
    cudaMemset(dBias_colchk_r, 0, size);
    
    // gpt-2
    if(Together){
      cudaMalloc((void**)&K_colchk<T>, size)/3;
      cudaMemset(K_colchk<T>, 0, size/3);
      cudaMalloc((void**)&V_colchk<T>, size/3);
      cudaMemset(V_colchk<T>, 0, size/3);
      cudaMalloc((void**)&tmp_colchk<T>, size);
      cudaMemset(tmp_colchk<T>, 0, size);
    }
    else{
      // bert
      if(QKV == 'k'){
        cudaMalloc((void**)&K_colchk<T>, size);
        cudaMemset(K_colchk<T>, 0, size);
      }
      else if(QKV == 'v'){
        cudaMalloc((void**)&V_colchk<T>, size);
        cudaMemset(V_colchk<T>, 0, size);
      }
    }
    
    size = num_batches * 2 * m * sizeof(T);
    cudaMalloc((void**)&dC_rowchk<T>, size);
    cudaMemset(dC_rowchk<T>, 0, size);
    cudaMalloc((void**)&dC_rowchk_r<T>, size);
    cudaMemset(dC_rowchk_r<T>, 0, size);

    cudaMalloc((void**)&dBias_rowchk, size);
    cudaMemset(dBias_rowchk, 0, size);
    cudaMalloc((void**)&dBias_rowchk_r, size);
    cudaMemset(dBias_rowchk_r, 0, size);
    
    // gpt-2
    if(Together){
      cudaMalloc((void**)&Q_rowchk<T>, size/3);
      cudaMemset(Q_rowchk<T>, 0, size/3);
      cudaMalloc((void**)&K_rowchk<T>, size/3);
      cudaMemset(K_rowchk<T>, 0, size/3);
      cudaMalloc((void**)&tmp_rowchk<T>, size);
      cudaMemset(tmp_rowchk<T>, 0, size);
    }
    else{
      if(QKV == 'q'){
        cudaMalloc((void**)&Q_rowchk<T>, size);
        cudaMemset(Q_rowchk<T>, 0, size);
      }
      else if(QKV == 'k'){
        cudaMalloc((void**)&K_rowchk<T>, size);
        cudaMemset(K_rowchk<T>, 0, size);
      }
    }

    int64_t len = m / num_head;
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

    len = n / num_batches;
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
    bool INJECTION = false;
    
    destinationFile = "abftbgemm/control/abftCOL_FT.txt";
    fullPath = homePath / destinationFile;
    std::ifstream colFile(fullPath);
    if (colFile.is_open()){
      colFile.get(flag);
      if(flag == 'f'){
        COL_FT = false;
      }
      // printf("%c", flag);
    }
    else{
      printf("COL_FT: Cannot open file, using default setting.\n");
    }
    colFile.close();

    destinationFile = "abftbgemm/control/abftROW_FT.txt";
    fullPath = homePath / destinationFile;
    std::ifstream rowFile(fullPath);
    if (rowFile.is_open()){
      rowFile.get(flag);
      if(flag == 'f'){
        ROW_FT = false;
      }
      // printf("%c", flag);
    }
    else{
      printf("ROW_FT: Cannot open file, using default setting.\n");
    }
    rowFile.close();

    destinationFile = "abftbgemm/control/Injection.txt";
    fullPath = homePath / destinationFile;
    std::ifstream injFile(fullPath);
    if (injFile.is_open()){
      injFile.get(flag);
      if(flag == 't'){
        INJECTION = true;
      }
      // printf("%c", flag);
    }
    else{
      printf("Injection: Cannot open file, using default setting.\n");
    }
    injFile.close();


    auto start = high_resolution_clock::now();
    if constexpr (std::is_same<T, float>::value) {
      abftGemmBiasPassChk<float>(transpose_mat1, transpose_mat2, m, n, k,
        alpha_val, mat1_ptr_, mat1_ld,
        mat2_ptr_, mat2_ld, bias_, 
        result_ptr, result_ld,
        activation,
        chk_v_a, chk_v_b, ld_chk_v,
        dBias_colchk, dBias_rowchk, dBias_colchk_r, dBias_rowchk_r,
        num_batches, num_head,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, QKV, INJECTION, Together, homePath);
    }
    else if constexpr (std::is_same<T, at::Half>::value) {
      abftGemmBiasPassChk<at::Half>(transpose_mat1, transpose_mat2, m, n, k,
        alpha_val, mat1_ptr_, mat1_ld,
        mat2_ptr_, mat2_ld, bias_, 
        result_ptr, result_ld,
        activation,
        chk_v_a, chk_v_b, ld_chk_v,
        dBias_colchk, dBias_rowchk, dBias_colchk_r, dBias_rowchk_r,
        num_batches, num_head,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, QKV, INJECTION, Together, homePath);
    }
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<microseconds>(stop - start);
    std::cout << "abftGemmBias: " << duration.count() / 1000.0 << std::endl;
    destinationFile = "abftbgemm/records/time/abftBias.txt";
    fullPath = homePath / destinationFile;
    recordTime(fullPath, (duration.count() / 1000.0), DEBUG);

    cudaFree(dA_colchk<T>);
    cudaFree(dA_rowchk<T>);
    cudaFree(dA_colchk_r<T>);
    cudaFree(dA_rowchk_r<T>);
    cudaFree(dB_colchk<T>);
    cudaFree(dB_rowchk<T>);
    cudaFree(dB_colchk_r<T>);
    cudaFree(dB_rowchk_r<T>);
    cudaFree(dC_colchk<T>);
    cudaFree(dC_rowchk<T>);
    cudaFree(dC_colchk_r<T>);
    cudaFree(dC_rowchk_r<T>);
    cudaFree(chk_v_a);
    cudaFree(chk_v_b);
    cudaFree(dBias_colchk);
    cudaFree(dBias_rowchk);
    cudaFree(dBias_colchk_r);
    cudaFree(dBias_rowchk_r);

    // printf("Bias Test\n");
}

template void myGemmBiasPassChk<float>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<float> alpha_val,
  const float* mat1_ptr, int64_t mat1_ld,
  const float* mat2_ptr, int64_t mat2_ld,  const float* bias,
  float* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template void myGemmBiasPassChk<at::Half>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<at::Half> alpha_val,
  const at::Half* mat1_ptr, int64_t mat1_ld,
  const at::Half* mat2_ptr, int64_t mat2_ld,  const at::Half* bias,
  at::Half* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template void myGemmBiasPassChk<double>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<double> alpha_val,
  const double* mat1_ptr, int64_t mat1_ld,
  const double* mat2_ptr, int64_t mat2_ld,  const double* bias,
  double* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template void myGemmBiasPassChk<at::BFloat16>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<at::BFloat16> alpha_val,
  const at::BFloat16* mat1_ptr, int64_t mat1_ld,
  const at::BFloat16* mat2_ptr, int64_t mat2_ld,  const at::BFloat16* bias,
  at::BFloat16* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template <typename T>
void abftGemmBiasPassChk(
    bool transpose_mat1, bool transpose_mat2,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<T> alpha_val, T* mat1_ptr, int64_t mat1_ld,
    T* mat2_ptr, int64_t mat2_ld, T* bias,
    T* result_ptr, int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation,
    T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,
    T *dBias_colchk, T *dBias_rowchk, T *dBias_colchk_r, T *dBias_rowchk_r,
    int64_t num_batches, int64_t num_head,                             
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
    char QKV, bool INJECTION, bool Together, fs::path homePath) {
  
  // std::cout << "Using gemm_and_bias." << std::endl;
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
  // std::cout << epilogue << std::endl;
  if (activation == GEMMAndBiasActivationEpilogue::RELU) {
    // printf("relu.\n");
    epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
  } else if (activation == GEMMAndBiasActivationEpilogue::GELU) {
    // printf("gelu.\n");
#if CUDA_VERSION >= 11040 || defined(USE_ROCM)
    epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
#endif
  }

  if (bias != nullptr) {
    // printf("bias not null.\n");
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
  cudaStream_t stream_main;
  cudaStreamCreate(&stream_main);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream_main);

  // cublasHandle_t handle_colchk;
  // cublasCreate(&handle_colchk);
  // cublasHandle_t handle_rowchk;
  // cublasCreate(&handle_rowchk);

  // cudaStream_t stream_main, stream_colchk, stream_rowchk;
  // cudaStreamCreate(&stream_main);
  // cudaStreamCreate(&stream_colchk);
  // cudaStreamCreate(&stream_rowchk);

  // cublasSetStream(handle_colchk, stream_colchk);
  // cublasSetStream(handle_rowchk, stream_rowchk);


  cudaEvent_t main_compute_done;
  cudaEventCreate(&main_compute_done);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t, t1, t_Achk, t_Bchk, t_Biasrowchk, t_Biascolchk;
  int64_t nb = 0;
  fs::path destinationFile, fullPath;

  float falpha = at::opmath_type<T>(1);
  float fbeta = at::opmath_type<T>(0);
  printf("m:%d, n:%d, k:%d\n ", m, n, k);

  // printf("alpha: %f, beta: %f \n", alpha_val, beta_val);

  // printf("mat1: \n");
  // outputChk(mat1_ptr, 1, mat1_ld, m*k, m, k);
  // printf("mat2:\n");
  // outputChk(mat2_ptr,1, mat2_ld, k*n, k, n);

  // A chk
  if(COL_FT){
    if (DEBUG) std::cout << "dA_checksum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    nb = m / num_head;
    if(transa == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value) {
        for(int i=0; i<m; i+=nb){
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, nb, 
                      &falpha, chk_v_a, ld_chk_v, 
                      mat1_ptr + i, mat1_ld, &fbeta, 
                      dA_colchk<T>+(i/nb)*2, ldda_colchk);
        }
        // cublasSgemmStridedBatched(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, nb,
        //                             &falpha, chk_v_a, ld_chk_v, 0,
        //                             mat1_ptr, (mat1_ld/num_head), nb, &fbeta,
        //                             dA_colchk_r<T>, (ldda_colchk_r/num_head), k*2,
        //                             num_head);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        for(int i=0; i<m; i+=nb){
          cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, nb,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      mat1_ptr + i, CUDA_R_16F, mat1_ld,
                      &fbeta, dA_colchk<T>+(i/nb)*2, CUDA_R_16F, ldda_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // cublasGemmStridedBatchedEx(handle_colchk, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, nb,
        //                 &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 0,
        //                 mat1_ptr, CUDA_R_16F, (mat1_ld/num_head), nb, &fbeta,
        //                 dA_colchk_r<T>, CUDA_R_16F, (ldda_colchk_r/num_head), k*2,
        //                 num_head, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } 
    }
    else{
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                                    &falpha, mat1_ptr, mat1_ld, k*nb,
                                    chk_v_a, ld_chk_v, 0, &fbeta,
                                    dA_rowchk_r<T>, ldda_rowchk_r, k*2,
                                    num_head);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                        &falpha, mat1_ptr, CUDA_R_16F, mat1_ld, k*nb,
                        chk_v_a, CUDA_R_16F, ld_chk_v, 0, &fbeta,
                        dA_rowchk_r<T>, CUDA_R_16F, ldda_rowchk_r, k*2,
                        num_head, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } 
    }
    // cudaStreamSynchronize(stream_colchk);
    if (DEBUG) {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Achk, start, stop);
    }
  }
  // printf("A chk\n");

  // B chk
  if (ROW_FT){
    if (DEBUG) std::cout << "dB_checksum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    nb = n / num_batches;
    if (transb == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value){
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                                    &falpha, mat2_ptr, mat2_ld, k*nb,
                                    chk_v_b, ld_chk_v, 0, &fbeta,
                                    dB_rowchk_r<T>, lddb_rowchk_r, k*2,
                                    num_batches);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, nb,
                          &falpha, mat2_ptr, CUDA_R_16F, mat2_ld, k*nb,
                          chk_v_b, CUDA_R_16F, ld_chk_v, 0, &fbeta,
                          dB_rowchk_r<T>, CUDA_R_16F, lddb_rowchk_r, k*2,
                          num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    else{
      if constexpr (std::is_same<T, float>::value){
          for(int i=0; i<n; i+=nb){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, nb, 
                        &falpha, chk_v_b, ld_chk_v, 
                        mat2_ptr+i, mat2_ld, &fbeta, 
                        dB_colchk<T>+(i/nb)*2, lddb_colchk);
          }
        }
        else if constexpr(std::is_same<T, at::Half>::value){
          for(int i=0; i<n; i+=nb){
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, nb,
                        &falpha, chk_v_b, CUDA_R_16F, ld_chk_v, 
                        mat2_ptr+i, CUDA_R_16F, mat2_ld,
                        &fbeta, dB_colchk<T>+(i/nb)*2, CUDA_R_16F, lddb_colchk,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          }
        }
    }
    // cudaStreamSynchronize(stream_rowchk);
    if (DEBUG) {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
    }
  }
  // printf("B chk\n");

  MatrixMerge<<<1, dim3(num_head, 1), 0, stream_main>>>(dA_rowchk_r<T>, dA_rowchk<T>, k, 2, k, 2*num_head, 1);
  MatrixMerge<<<1, dim3(num_batches, 1), 0, stream_main>>>(dB_rowchk_r<T>, dB_rowchk<T>, k, 2, k, 2*num_batches, 1);


  T *biasMatrix;
  size_t size =  m * n * sizeof(T);
  cudaMalloc((void**)&biasMatrix, size);
  getBiasMatrix<<<dim3(128), dim3((n+128-1)/128), 0, stream_main>>>(bias, biasMatrix, m);
  
  // printf("bias:\n");
  // outputChk(biasMatrix, 1, result_ld, m*n, m, n);
  
  // Bias col chk
  if (COL_FT){
    if (DEBUG) std::cout << "bias_check_col_sum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    nb = m / num_head;
    if constexpr (std::is_same<T, float>::value) {
      cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, nb,
                                    &falpha, chk_v_a, ld_chk_v, 0,
                                    biasMatrix, (result_ld), nb, &fbeta,
                                    dBias_colchk_r, (lddc_colchk_r/num_head), n*2,
                                    num_head);                                    
    }
    else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, nb,
                                  &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 0,
                                  biasMatrix, CUDA_R_16F, (result_ld), nb, &fbeta,
                                  dBias_colchk_r, CUDA_R_16F, (lddc_colchk_r/num_head), n*2,
                                  num_head, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    // cudaStreamSynchronize(stream_colchk); 
    if (DEBUG) {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Biascolchk, start, stop);
        t_Biascolchk /= 1.0;
    }
    // printf("bias colchk:\n");
    // outputChk(dBias_colchk_r, num_head, lddc_colchk/num_head, 2*n, 2, n);
  }
  
  // Bias row chk
  if (ROW_FT){
    if (DEBUG) std::cout << "bias_check_row_sum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    nb = n / num_batches;
    if constexpr (std::is_same<T, float>::value){
      cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, 2, nb,
                                    &falpha, biasMatrix, result_ld, m*nb,
                                    chk_v_b, ld_chk_v, 0, &fbeta,
                                    dBias_rowchk_r, lddc_rowchk_r, m*2,
                                    num_batches);
    }
    else if constexpr(std::is_same<T, at::Half>::value){
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, 2, nb,
                          &falpha, biasMatrix, CUDA_R_16F, result_ld, m*nb,
                          chk_v_b, CUDA_R_16F, ld_chk_v, 0, &fbeta,
                          dBias_rowchk_r, CUDA_R_16F, lddc_rowchk_r, m*2,
                          num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    // cudaStreamSynchronize(stream_rowchk);
    if (DEBUG) {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Biasrowchk, start, stop);
        t_Biasrowchk /= 1.0;
    }
  // printf("bias rowchk:\n");
  // outputMatrixChk(dBias_rowchk, lddc_rowchk, num_batches*2*n, 1, m, 2*num_batches);
  }
  
  MatrixMerge<<<1, dim3(1, num_head), 0, stream_main>>>(dBias_colchk_r, dBias_colchk, 2, n, 2*num_head, n, num_head);
  // outputChk(dBias_colchk, 1, lddc_colchk, 0, 2*num_head, n);
  MatrixMerge<<<1, dim3(num_batches, 1), 0, stream_main>>>(dBias_rowchk_r, dBias_rowchk, m, 2, m, 2*num_batches, 1);

  cudaStreamSynchronize(stream_main);

  int64_t mem_row = 0;
  int64_t mem_col = 0;

  // falpha = alpha_val;
  // fbeta = beta_val;

  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  if (DEBUG)  cudaEventRecord(start, stream_main);
  // std::cout << "cublasLtMatmul" << std::endl;
  cublasLtMatmul(ltHandle, computeDesc.descriptor(), &alpha_val,
      mat1_ptr, Adesc.descriptor(),
      mat2_ptr, Bdesc.descriptor(), &beta_val,
      result_ptr, Cdesc.descriptor(), 
      result_ptr, Cdesc.descriptor(),
      &heuristicResult.algo, workspace.mutable_get(), workspaceSize, stream_main);
  cudaStreamSynchronize(stream_main);

  // printf("result:\n");
  // outputMatrix(result_ptr,result_ld, m*n, 1, m, n);
  
  // printf("Before injection C: \n");
  // outputChk(result_ptr, 1, result_ld, 0, m, n);

  if(INJECTION){
    if(Together){
      int64_t pos = getInjPos();
      if(QKV == 'q'){
        if(DEBUG) printf("Injection.\n");
        bitflip<<<1, 1, 0, stream_main>>>(result_ptr, pos);
      }
      else if(QKV == 'k'){
        if(DEBUG) printf("Injection.\n");
        bitflip<<<1, 1, 0, stream_main>>>(result_ptr, pos + (m/3)*n);
      }
      else{
        if(DEBUG) printf("Injection.\n");
        bitflip<<<1, 1, 0, stream_main>>>(result_ptr, pos + 2*(m/3)*n);
      }
    }
    else{
      if(DEBUG) printf("Injection.\n");
      int64_t pos = getInjPos();
      bitflip<<<1, 1, 0, stream_main>>>(result_ptr, pos);
    }
  }
  
  // printf("After injection C: \n");
  // outputChk(result_ptr, 1, result_ld, 0, m, n);

  if (DEBUG) {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t, start, stop);
      destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
      fullPath = homePath / destinationFile;
      printf("  gemm: %f (%f)(%f)\n", t, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
      recordEffeciency(fullPath,  t, 1, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
      if(COL_FT){
        printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", 
                t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
        
        recordEffeciency(fullPath,  
                t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
        
        printf("dBias_colchk_gemm: %f (%f)(%f)(%f)\n", 
                t_Biascolchk, t_Biascolchk/t, (double)1*m*2*n*2/t_Biascolchk/1e6, (double)1*(2*n+2*m+n*m)*sizeof(T)/t_Biascolchk/1e6);
        recordEffeciency(fullPath,  
                t_Biascolchk, t_Biascolchk/t, (double)1*m*2*n*2/t_Biascolchk/1e6, (double)1*(2*n+2*m+n*m)*sizeof(T)/t_Biascolchk/1e6);      
      }
      if(ROW_FT){
        printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", 
                t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
        recordEffeciency(fullPath,  
                t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
        
        printf("dBias_rowchk_gemm: %f (%f)(%f)(%f)\n",
                t_Biasrowchk, t_Biasrowchk/t, (double)1*2*n*m*2/t_Biasrowchk/1e6, (double)1*(2*m+m*n+2*n)*sizeof(T)/t_Biasrowchk/1e6);
        recordEffeciency(fullPath,  
                t_Biasrowchk, t_Biasrowchk/t, (double)1*2*n*m*2/t_Biasrowchk/1e6, (double)1*(2*m+m*n+2*n)*sizeof(T)/t_Biasrowchk/1e6);
      }
  }

  if (COL_FT){
    //std::cout << "  COL_FT" << std::endl;
    if (DEBUG)  cudaEventRecord(start, stream_main);
    if (transa == CUBLAS_OP_N) {
      if (DEBUG) std::cout << "dA_colchk * dB = dC_colchk" << std::endl;
      // K*4 must be greater then 2 * N
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, transa, transb, 2*num_head, n, k,
                    &falpha, dA_colchk<T>, ldda_colchk, 
                    mat2_ptr, mat2_ld, &fbeta, 
                    dC_colchk<T>, lddc_colchk);
      }
      else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, transa, transb, 2*num_head, n, k,
                      &falpha, dA_colchk<T>, CUDA_R_16F, ldda_colchk, 
                      mat2_ptr, CUDA_R_16F, mat2_ld,
                      &fbeta, dC_colchk<T>, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    else{
      if (DEBUG) std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, transa, transb, 2*num_head, n, k,
                    &falpha, dA_rowchk<T>, ldda_rowchk, 
                    mat2_ptr, mat2_ld, &fbeta, 
                    dC_colchk<T>, lddc_colchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmEx(handle, transa, transb, 2*num_head, n, k,
                      &falpha, dA_rowchk<T>, CUDA_R_16F, ldda_rowchk, 
                      mat2_ptr, CUDA_R_16F, mat2_ld,
                      &fbeta, dC_colchk<T>, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    // cudaStreamSynchronize(stream_colchk);
    // std::cout << "Output org dC_colchk: " << std::endl;
    // outputMatrixChk(dC_colchk, lddc_colchk, n*2, 1, 2, n);
    
    // dC_colchk + dBias_colchk
    // printf("dC_colchk + d Bias_colchk\n");
    if constexpr (std::is_same<T, float>::value){
      cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2*num_head, n,
                    &falpha, dBias_colchk, lddc_colchk, &falpha,
                    dC_colchk<T>, lddc_colchk,
                    dC_colchk<T>, lddc_colchk);
    }
    else if constexpr (std::is_same<T, at::Half>::value){
      dim3 blockSize(16, 16);
      dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (2 + blockSize.y - 1) / blockSize.y);
      addVector<T><<<blockSize, gridSize, (2*num_head*n)*sizeof(T), stream_main>>>(dC_colchk<T>, dBias_colchk, 2*num_head, n); 
    }
    // cudaStreamSynchronize(stream_colchk);
    // std::cout << "Output dC_colchk: " << std::endl;
    // outputMatrixChk(dC_colchk, lddc_colchk, n*2, 1, 2, n);
    
    if (DEBUG)  {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
      fullPath = homePath / destinationFile;
      
      printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", 
                  t1, t1/t, (double)(1*(2*num_head)*n*k*2 + 2*num_head*n)/t1/1e6, (double)(1*((2*num_head)*k+k*n+(2*num_head)*n) + 2*num_head*n)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath,  
                  t1, t1/t, (double)(1*(2*num_head)*n*k*2 + 2*num_head*n)/t1/1e6, (double)(1*((2*num_head)*k+k*n+(2*num_head)*n) + 2*num_head*n)*sizeof(T)/t1/1e6);     
    }
  }

  if (ROW_FT) {
      //std::cout << "  ROW_FT" << std::endl;
      if (DEBUG)  cudaEventRecord(start, stream_main);
      if (transb == CUBLAS_OP_N) {
        if (DEBUG) std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
        //we can further work on this to support trans A
        if constexpr (std::is_same<T, float>::value){
          cublasSgemm(handle, transa, transb,  m, 2*num_batches, k,
                      &falpha, mat1_ptr, mat1_ld, 
                      dB_rowchk<T>, lddb_rowchk, &fbeta, 
                      dC_rowchk<T>, lddc_rowchk);
        }
        else if constexpr(std::is_same<T, at::Half>::value){
          cublasGemmEx(handle, transa, transb,  m, 2*num_batches, k,
                        &falpha, mat1_ptr, CUDA_R_16F, mat1_ld, 
                        dB_rowchk<T>, CUDA_R_16F, lddb_rowchk,
                        &fbeta, dC_rowchk<T>, CUDA_R_16F, lddc_rowchk,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
      } else{
        if (DEBUG) std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
          if constexpr (std::is_same<T, float>::value){
            cublasSgemm(handle, transa, transb,  m, 2*num_batches, k,
                        &falpha, mat1_ptr, mat1_ld, 
                        dB_colchk<T>, lddb_colchk, &fbeta, 
                        dC_rowchk<T>, lddc_rowchk);
          }
          else if constexpr(std::is_same<T, at::Half>::value){
            cublasGemmEx(handle, transa, transb,  m, 2*num_batches, k,
                          &falpha, mat1_ptr, CUDA_R_16F, mat1_ld, 
                          dB_colchk<T>, CUDA_R_16F, lddb_colchk,
                          &fbeta, dC_rowchk<T>, CUDA_R_16F, lddc_rowchk,
                          CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          }
      }
      // dC_rowchk + dBias_rowchk
      // std::cout << "Output org dC_colchk: " << std::endl;
      // outputMatrixChk(dC_rowchk, lddc_rowchk, m*2, 1, m, 2);
      // dC_colchk + dBias_colchk
      // cudaStreamSynchronize(stream_rowchk);
      if constexpr (std::is_same<T, float>::value){ 
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 2*num_batches,
                  &falpha, dBias_rowchk, lddc_rowchk, &falpha,
                  dC_rowchk<T>, lddc_rowchk,
                  dC_rowchk<T>, lddc_rowchk);
      }
      else if constexpr (std::is_same<T,at::Half >::value){
        dim3 blockSize(16, 16);
        dim3 gridSize((2 + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        addVector<T><<<blockSize, gridSize, (m*num_batches*2)*sizeof(T),stream_main>>>(dC_rowchk<T>, dBias_rowchk, m, 2*num_batches);
      }
      // cudaStreamSynchronize(stream_rowchk);
      // std::cout << "Output dC_rowchk: " << std::endl;
      // outputMatrixChk(dC_rowchk, lddc_rowchk, m*2, 1, m, 2);
      // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
      if (DEBUG)  {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t1, start, stop);

        destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
        fullPath = homePath / destinationFile;
        printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", 
                  t1, t1/t, (double)(1*m*(2*num_batches)*k*2 + (2*num_batches*m))/t1/1e6, (double)(1*(m*k+k*(2*num_batches)+m*(2*num_batches)) + (2*num_batches*m))*sizeof(T)/t1/1e6);
        recordEffeciency(fullPath,
                  t1, t1/t, (double)(1*m*(2*num_batches)*k*2 + (2*num_batches*m))/t1/1e6, (double)(1*(m*k+k*(2*num_batches)+m*(2*num_batches)) + (2*num_batches*m))*sizeof(T)/t1/1e6);
      }
  }

  // printf("dC_colchk: \n");
  // outputChk(dC_colchk<T>, 1, lddc_colchk, 0, 2*num_head, n);
  // printf("dC_rowchk: \n");
  // outputChk(dC_rowchk<T>, 1, lddc_rowchk, 0, m, 2*num_batches);

  if(!Together){
      // for gpt-neo, bert, roberta
    if(QKV == 'q'){
      MatrixSplit<<<1, dim3(num_batches, num_head), 0, stream_main>>>(dC_rowchk<T>, Q_rowchk<T>, m/num_head, 2, lddc_rowchk, num_head);
      // printf("Q_rowchk: \n");
      // outputChk(Q_rowchk<T>, num_head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
    }
    else if(QKV == 'k'){
      MatrixSplit<<<1, dim3(num_batches, num_head), 0, stream_main>>>(dC_rowchk<T>, K_rowchk<T>, m/num_head, 2, lddc_rowchk, num_head);
      // printf("K_colchk: \n");
      // outputChk(K_colchk<T>, num_head*num_batches, 2, 2*n/num_batches, 2, n/num_batches);
      MatrixTranspose<<<1, num_head*num_batches, 0, stream_main>>>(K_rowchk<T>, K_colchk<T>, m/num_head, 2);
    }
    else {
      MatrixSplit<<<1, dim3(num_batches, num_head), 0, stream_main>>>(dC_colchk<T>, V_colchk<T>, 2, n/num_batches, lddc_colchk, num_head);
      // MatrixSplit<<<1, dim3(num_batches, num_head)>>>(dC_rowchk<T>, V_rowchk<T>, m/num_head, 2, lddc_rowchk, num_head);
      // printf("V_colchk: \n");
      // outputChk(V_colchk<T>, num_head*num_batches, 2, 2*n/num_batches, 2, n/num_batches);
      // printf("V_rowchk: \n");
      // outputChk(V_rowchk<T>, num_head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
    }
  }
  else{
    // For GPT-2
    int head = num_head / 3;
    MatrixSplit<<<1, dim3(num_batches, num_head), 0, stream_main>>>(dC_rowchk<T>, tmp_rowchk<T>, m/num_head, 2, lddc_rowchk, num_head);
    // outputChk(tmp_rowchk<T>, num_head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
    MatrixSplit<<<1, dim3(num_batches, num_head), 0, stream_main>>>(dC_colchk<T>, tmp_colchk<T>, 2, n/num_batches, lddc_colchk, num_head);
    // outputChk(tmp_colchk<T>, num_head*num_batches, 2, 2*n/num_batches, 2, n/num_batches);

    // for Q
    assignChk<<<1, dim3(num_batches * head), 0, stream_main>>>(tmp_rowchk<T>, Q_rowchk<T>, m/num_head, 2, (head*num_batches), head, 0);
    // printf("Q_rowchk: \n");
    // outputChk(Q_rowchk<T>, head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
    
    // for K
    assignChk<<<1, dim3(num_batches * head), 0, stream_main>>>(tmp_rowchk<T>, K_rowchk<T>, m/num_head, 2, (head*num_batches), head, 1);
    // printf("K_rowchk: \n");
    // outputChk(K_rowchk<T>, head*num_batches, m/num_head, 2*m/num_head, m/num_head, 2);
    MatrixTranspose<<<1, head*num_batches, 0, stream_main>>>(K_rowchk<T>, K_colchk<T>, m/num_head, 2);
    // printf("k_colchk: \n");
    // outputChk(K_colchk<T>, head*num_batches, 2, 2*m/num_head, 2,  m/num_head);
    // cudaStreamSynchronize(stream_rowchk);
    
    // for V
    assignChk<<<1, dim3(num_batches * head), 0, stream_main>>>(tmp_colchk<T>, V_colchk<T>, 2, n/num_batches, (head*num_batches), head, 2);
    // printf("V_colchk: \n");
    // outputChk(V_colchk<T>, head*num_batches, 2, 2*n/num_batches, 2, n/num_batches);

    // cudaStreamSynchronize(stream_colchk);
  }
}

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

    const char* homeDir = nullptr;
    homeDir = getenv("HOME");
    fs::path homePath(homeDir);
    fs::path destinationFile = "abftbgemm/control/QKV.txt";
    fs::path fullPath = homePath / destinationFile;

    char QKV;
    std::ifstream qkvFile(fullPath);
    if (qkvFile.is_open()){
      qkvFile.get(QKV);
    }
    else{
      printf("QKV: Cannot open file, using default setting.\n");
    }
    qkvFile.close();

    T *mat1_ptr_ = const_cast<T*>(mat1_ptr);
    T *mat2_ptr_ = const_cast<T*>(mat2_ptr);
    T *bias_ = const_cast<T*>(bias);

    // printf("m:%d, n:%d, k:%d\n", m, n, k);
    // printf("%d, %d, %d\n", mat1_ld, mat2_ld, result_ld);

    ldda_colchk = 2;
    ldda_colchk_r = 2;
    ldda_rowchk = k;
    ldda_rowchk_r = k;

    lddb_rowchk = k;
    lddb_rowchk_r = k;
    lddb_colchk = 2;
    lddb_colchk_r = 2;

    lddc_colchk = 2;
    lddc_colchk_r = 2;
    lddc_rowchk = m;
    lddc_rowchk_r = m;
    int64_t ld_chk_v = 2;

    // T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
    // T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
    // T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
    T *dBias_colchk, *dBias_rowchk, *dBias_colchk_r,*dBias_rowchk_r;
    T *chk_v_a;
    T *chk_v_b;
  
    size_t size = 2 * k * sizeof(T);
    cudaMalloc((void**)&dA_colchk<T>, size);
    cudaMemset(dA_colchk<T>, 0, size);
    cudaMalloc((void**)&dA_colchk_r<T>, size);
    cudaMemset(dA_colchk_r<T>, 0, size);

    cudaMalloc((void**)&dA_rowchk<T>, size);
    cudaMemset(dA_rowchk<T>, 0, size);
    cudaMalloc((void**)&dA_rowchk_r<T>, size);
    cudaMemset(dA_rowchk_r<T>, 0, size);
    //std::cout << "  finish dA." << std::endl;
    
    cudaMalloc((void**)&dB_colchk<T>, size);
    cudaMemset(dB_colchk<T>, 0, size);
    cudaMalloc((void**)&dB_colchk_r<T>, size);
    cudaMemset(dB_colchk_r<T>, 0, size);
    
    cudaMalloc((void**)&dB_rowchk<T>, size);
    cudaMemset(dB_rowchk<T>, 0, size);
    cudaMalloc((void**)&dB_rowchk_r<T>, size);
    cudaMemset(dB_rowchk_r<T>, 0, size);
    //std::cout << "  finish dB." << std::endl;

    size = 2 * n * sizeof(T);
    cudaMalloc((void**)&dC_colchk<T>, size);
    cudaMemset(dC_colchk<T>, 0, size);
    cudaMalloc((void**)&dC_colchk_r<T>, size);
    cudaMemset(dC_colchk_r<T>, 0, size);

    cudaMalloc((void**)&dBias_colchk, size);
    cudaMemset(dBias_colchk, 0, size);
    cudaMalloc((void**)&dBias_colchk_r, size);
    cudaMemset(dBias_colchk_r, 0, size);
    
    size = 2 * m * sizeof(T);
    cudaMalloc((void**)&dC_rowchk<T>, size);
    cudaMemset(dC_rowchk<T>, 0, size);
    cudaMalloc((void**)&dC_rowchk_r<T>, size);
    cudaMemset(dC_rowchk_r<T>, 0, size);

    cudaMalloc((void**)&dBias_rowchk, size);
    cudaMemset(dBias_rowchk, 0, size);
    cudaMalloc((void**)&dBias_rowchk_r, size);
    cudaMemset(dBias_rowchk_r, 0, size);

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
    bool INJECTION = false;
    
    destinationFile = "abftbgemm/control/abftCOL_FT.txt";
    fullPath = homePath / destinationFile;
    char flag;
    std::ifstream colFile(fullPath);
    if (colFile.is_open()){
      colFile.get(flag);
      if(flag == 'f'){
        COL_FT = false;
      }
      // printf("%c", flag);
    }
    else{
      printf("COL_FT: Cannot open file, using default setting.\n");
    }
    colFile.close();
    
    destinationFile = "abftbgemm/control/abftROW_FT.txt";
    fullPath = homePath / destinationFile;
    std::ifstream rowFile(fullPath);
    if (rowFile.is_open()){
      rowFile.get(flag);
      if(flag == 'f'){
        ROW_FT = false;
      }
      // printf("%c", flag);
    }
    else{
      printf("ROW_FT: Cannot open file, using default setting.\n");
    }
    rowFile.close();

    destinationFile = "abftbgemm/control/Injection.txt";
    fullPath = homePath / destinationFile;
    std::ifstream injFile(fullPath);
    if (injFile.is_open()){
      injFile.get(flag);
      if(flag == 't'){
        INJECTION = true;
      }
      // printf("%c", flag);
    }
    else{
      printf("Injection: Cannot open file, using default setting.\n");
    }
    injFile.close();

    auto start = high_resolution_clock::now();
    if constexpr (std::is_same<T, float>::value) {
      abftGemmBias<float>(transpose_mat1, transpose_mat2, m, n, k,
        alpha_val, mat1_ptr_, mat1_ld,
        mat2_ptr_, mat2_ld, bias_, 
        result_ptr, result_ld,
        activation,
        chk_v_a, chk_v_b, ld_chk_v,
        dBias_colchk, dBias_rowchk, dBias_colchk_r, dBias_rowchk_r,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, QKV, INJECTION, homePath);
    }
    else if constexpr (std::is_same<T, at::Half>::value) {
      abftGemmBias<at::Half>(transpose_mat1, transpose_mat2, m, n, k,
        alpha_val, mat1_ptr_, mat1_ld,
        mat2_ptr_, mat2_ld, bias_, 
        result_ptr, result_ld,
        activation,
        chk_v_a, chk_v_b, ld_chk_v,
        dBias_colchk, dBias_rowchk, dBias_colchk_r, dBias_rowchk_r,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER, QKV, INJECTION, homePath);
    }
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<microseconds>(stop - start);
    std::cout << "abftGemmBias: " << duration.count() / 1000.0 << std::endl;
    
    destinationFile = "abftbgemm/records/time/abftBias.txt";
    fullPath = homePath / destinationFile;
    recordTime(fullPath, (duration.count() / 1000.0), DEBUG);

    cudaFree(dA_colchk<T>);
    cudaFree(dA_rowchk<T>);
    cudaFree(dA_colchk_r<T>);
    cudaFree(dA_rowchk_r<T>);
    cudaFree(dB_colchk<T>);
    cudaFree(dB_rowchk<T>);
    cudaFree(dB_colchk_r<T>);
    cudaFree(dB_rowchk_r<T>);
    cudaFree(dC_colchk<T>);
    cudaFree(dC_rowchk<T>);
    cudaFree(dC_colchk_r<T>);
    cudaFree(dC_rowchk_r<T>);
    cudaFree(chk_v_a);
    cudaFree(chk_v_b);
    cudaFree(dBias_colchk);
    cudaFree(dBias_rowchk);
    cudaFree(dBias_colchk_r);
    cudaFree(dBias_rowchk_r);

    // printf("Bias Test\n");
}

template void myGemmBias<float>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<float> alpha_val,
  const float* mat1_ptr, int64_t mat1_ld,
  const float* mat2_ptr, int64_t mat2_ld,  const float* bias,
  float* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template void myGemmBias<at::Half>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<at::Half> alpha_val,
  const at::Half* mat1_ptr, int64_t mat1_ld,
  const at::Half* mat2_ptr, int64_t mat2_ld,  const at::Half* bias,
  at::Half* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template void myGemmBias<double>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<double> alpha_val,
  const double* mat1_ptr, int64_t mat1_ld,
  const double* mat2_ptr, int64_t mat2_ld,  const double* bias,
  double* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template void myGemmBias<at::BFloat16>( bool transpose_mat1, bool transpose_mat2, int64_t m, int64_t n, int64_t k, at::opmath_type<at::BFloat16> alpha_val,
  const at::BFloat16* mat1_ptr, int64_t mat1_ld,
  const at::BFloat16* mat2_ptr, int64_t mat2_ld,  const at::BFloat16* bias,
  at::BFloat16* result_ptr, int64_t result_ld, GEMMAndBiasActivationEpilogue activation);

template <typename T>
void abftGemmBias(
    bool transpose_mat1, bool transpose_mat2,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<T> alpha_val, T* mat1_ptr, int64_t mat1_ld,
    T* mat2_ptr, int64_t mat2_ld, T* bias,
    T* result_ptr, int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation,
    T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,
    T *dBias_colchk, T *dBias_rowchk, T *dBias_colchk_r, T *dBias_rowchk_r,                             
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, 
    char QKV, bool INJECTION, fs::path homePath) {
  
  // std::cout << "Using gemm_and_bias." << std::endl;
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
    printf("relue\n");
    epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
  } else if (activation == GEMMAndBiasActivationEpilogue::GELU) {
    printf("gelu\n");
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

  cudaStream_t stream_main;
  cudaStreamCreate(&stream_main);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream_main);
  //
  // cublasHandle_t handle_colchk;
  // cublasCreate(&handle_colchk);
  // cublasHandle_t handle_rowchk;
  // cublasCreate(&handle_rowchk);

  // cudaStream_t stream_main, stream_colchk, stream_rowchk;
  // cudaStreamCreate(&stream_main);
  // cudaStreamCreate(&stream_colchk);
  // cudaStreamCreate(&stream_rowchk);

  // cublasSetStream(handle_colchk, stream_colchk);
  // cublasSetStream(handle_rowchk, stream_rowchk);


  cudaEvent_t main_compute_done;
  cudaEventCreate(&main_compute_done);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t, t1, t_Achk, t_Bchk, t_Biasrowchk, t_Biascolchk;

  fs::path destinationFile, fullPath;

  float falpha = at::opmath_type<T>(1);
  float fbeta = at::opmath_type<T>(0);
  printf("m:%d, n:%d, k:%d\n ", m, n, k);

  // printf("alpha: %f, beta: %f \n", alpha_val, beta_val);

  // printf("mat1: \n");
  // outputChk(mat1_ptr, 1, mat1_ld, m*k, m, k);
  // printf("mat2:\n");
  // outputChk(mat2_ptr,1, mat2_ld, k*n, k, n);

  // A chk
  if(COL_FT){
    if (DEBUG) std::cout << "dA_checksum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    if(transa == CUBLAS_OP_N){
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m, 
                      &falpha, chk_v_a, ld_chk_v, 
                      mat1_ptr, mat1_ld, &fbeta, 
                      dA_colchk<T>, ldda_colchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      mat1_ptr, CUDA_R_16F, mat1_ld,
                      &fbeta, dA_colchk<T>, CUDA_R_16F, ldda_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } 
    }
    else{
      if constexpr (std::is_same<T, float>::value) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, m, 
                      &falpha, mat1_ptr, mat1_ld, 
                      chk_v_a, ld_chk_v, &fbeta, 
                      dA_rowchk<T>, ldda_rowchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, k,2,m,
                      &falpha, mat1_ptr, CUDA_R_16F, mat1_ld, 
                      chk_v_a, CUDA_R_16F, ld_chk_v,
                      &fbeta, dA_rowchk<T>, CUDA_R_16F, ldda_rowchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } 
    }
    // cudaStreamSynchronize(stream_colchk);
    if (DEBUG) {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Achk, start, stop);
    }
  }

  // B chk
  if (ROW_FT){
    if (DEBUG) std::cout << "dB_checksum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    if (QKV != 'c'){
      if (DEBUG) printf(" Navive Calculate.\n");
      if (transb == CUBLAS_OP_N){
        if constexpr (std::is_same<T, float>::value){
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n, 
                      &falpha, mat2_ptr, mat2_ld, 
                      chk_v_b, ld_chk_v, &fbeta, 
                      dB_rowchk<T>, lddb_rowchk);
        }
        else if constexpr(std::is_same<T, at::Half>::value){
          cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n,
                        &falpha, mat2_ptr, CUDA_R_16F, mat2_ld, 
                        chk_v_b, CUDA_R_16F, ld_chk_v,
                        &fbeta, dB_rowchk<T>, CUDA_R_16F, lddb_rowchk,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // printf("dB_rowchk: \n");
        // outputChk(dB_rowchk<T>, 1, lddb_rowchk, 0, k, 2);  
      }
      else{
        if constexpr (std::is_same<T, float>::value){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, n, 
                        &falpha, chk_v_b, ld_chk_v, 
                        mat2_ptr, mat2_ld, &fbeta, 
                        dB_colchk<T>, lddb_colchk);
          }
          else if constexpr(std::is_same<T, at::Half>::value){
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, k, n,
                          &falpha, chk_v_b, CUDA_R_16F, ld_chk_v, 
                          mat2_ptr, CUDA_R_16F, mat2_ld,
                          &fbeta, dB_colchk<T>, CUDA_R_16F, lddb_colchk,
                          CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          }
      }
    }
    else{
      if (DEBUG) printf(" Pass Chk.\n");
      dB_rowchk<T> = CL_rowchk<T>;
      // printf("dB_rowchk: \n");
      // outputChk(dB_rowchk<T>, 1, lddb_rowchk, 0, k, 2);  
    }
    
    // cudaStreamSynchronize(stream_rowchk);
    if (DEBUG) {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_Bchk, start, stop);
      t_Bchk /= 1.0;
    }
  }
  
  T *biasMatrix;
  size_t size =  m * n * sizeof(T);
  cudaMalloc((void**)&biasMatrix, size);
  getBiasMatrix<<<dim3(128), dim3((n+128-1)/128)>>>(bias, biasMatrix, m);
  
  // printf("bias:\n");
  // outputMatrixChk(biasMatrix, m, m, 1, m, n);
  // outputChk(biasMatrix, 1, result_ld, m*n, m, n);
  
  // Bias col chk
  if (COL_FT){
    if (DEBUG) std::cout << "bias_check_col_sum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    if constexpr (std::is_same<T, float>::value) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, m, 
                    &falpha, chk_v_a, ld_chk_v, 
                    biasMatrix, result_ld, &fbeta, 
                    dBias_colchk, lddc_colchk);
    }
    else if constexpr(std::is_same<T, at::Half>::value) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, m,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      biasMatrix, CUDA_R_16F, result_ld,
                      &fbeta, dBias_colchk, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    // cudaStreamSynchronize(stream_colchk); 
    if (DEBUG) {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Biascolchk, start, stop);
        t_Biascolchk /= 1.0;
    }
    // printf("bias colchk:\n");
    // outputMatrixChk(dBias_colchk, lddc_colchk, 2*m, 1, 2, n);
  }
  
  // Bias row chk
  if (ROW_FT){
    if (DEBUG) std::cout << "bias_check_row_sum" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_main);
    if constexpr (std::is_same<T, float>::value){
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, 2, n, 
                  &falpha, biasMatrix, result_ld, 
                  chk_v_b, ld_chk_v, &fbeta, 
                  dBias_rowchk, lddc_rowchk);
    }
    else if constexpr(std::is_same<T, at::Half>::value){
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, 2, n,
                    &falpha, biasMatrix, CUDA_R_16F, result_ld, 
                    chk_v_b, CUDA_R_16F, ld_chk_v,
                    &fbeta, dBias_rowchk, CUDA_R_16F, lddc_rowchk,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    // cudaStreamSynchronize(stream_rowchk);
    if (DEBUG) {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Biasrowchk, start, stop);
        t_Biasrowchk /= 1.0;
    }
  // printf("bias rowchk:\n");
  // outputMatrixChk(dBias_rowchk, lddc_rowchk, 2*n, 1, m, 2);
  }
  
  int64_t mem_row = 0;
  int64_t mem_col = 0;

  // falpha = alpha_val;
  // fbeta = beta_val;

  if (DEBUG) std::cout<<"A*B=C." << std::endl;
  if (DEBUG)  cudaEventRecord(start, stream_main);
  // std::cout << "cublasLtMatmul" << std::endl;
  cublasLtMatmul(ltHandle, computeDesc.descriptor(), &alpha_val,
      mat1_ptr, Adesc.descriptor(),
      mat2_ptr, Bdesc.descriptor(), &beta_val,
      result_ptr, Cdesc.descriptor(), 
      result_ptr, Cdesc.descriptor(),
      &heuristicResult.algo, workspace.mutable_get(), workspaceSize, stream_main);
  cudaStreamSynchronize(stream_main);
  // printf("result:\n");
  // outputMatrix(result_ptr,result_ld, m*n, 1, m, n);
  if (DEBUG) {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t, start, stop);
      
      destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
      fullPath = homePath / destinationFile;
      printf("  gemm: %f (%f)(%f)\n", t, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
      recordEffeciency(fullPath,  t, 1, (double)1*m*n*k*2/t/1e6, (double)1*(m*k+k*n+m*n)/t/1e6);
      
      if(COL_FT){
        printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
        recordEffeciency(fullPath,  t_Achk, t_Achk/t, 
                                        (double)1*m*2*k*2/t_Achk/1e6, (double)1*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
        
        printf("dBias_colchk_gemm: %f (%f)(%f)(%f)\n", t_Biascolchk, t_Biascolchk/t, (double)1*m*2*n*2/t_Biascolchk/1e6, (double)1*(2*n+2*m+n*m)*sizeof(T)/t_Biascolchk/1e6);
        recordEffeciency(fullPath,  t_Biascolchk, t_Biascolchk/t, 
                                        (double)1*m*2*n*2/t_Biascolchk/1e6, (double)1*(2*n+2*m+n*m)*sizeof(T)/t_Biascolchk/1e6);      
      }
      
      if(ROW_FT){
        printf("dB_chk_gemm: %f (%f)(%f)(%f)\n",t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
        recordEffeciency(fullPath,  t_Bchk, t_Bchk/t, (double)1*2*n*k*2/t_Bchk/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
        
        printf("dBias_rowchk_gemm: %f (%f)(%f)(%f)\n", t_Biasrowchk, t_Biasrowchk/t, (double)1*2*n*m*2/t_Biasrowchk/1e6, (double)1*(2*m+m*n+2*n)*sizeof(T)/t_Biasrowchk/1e6);
        recordEffeciency(fullPath,  
          t_Biasrowchk, t_Biasrowchk/t, (double)1*2*n*m*2/t_Biasrowchk/1e6, (double)1*(2*m+m*n+2*n)*sizeof(T)/t_Biasrowchk/1e6);
      }
  }

  if(INJECTION){
    if(DEBUG) printf("Injection.\n");
    int64_t pos = getInjPos();
    bitflip<<<1, 1, 0, stream_main>>>(result_ptr, pos);
  }

  if (COL_FT){
    //std::cout << "  COL_FT" << std::endl;
    if (DEBUG)  cudaEventRecord(start, stream_main);
    if (transa == CUBLAS_OP_N) {
      if (DEBUG) std::cout << "dA_colchk * dB = dC_colchk" << std::endl;
      // K*4 must be greater then 2 * N
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, transa, transb, 2, n, k,
                    &falpha, dA_colchk<T>, ldda_colchk, 
                    mat2_ptr, mat2_ld, &fbeta, 
                    dC_colchk<T>, lddc_colchk);
      }
      else if constexpr(std::is_same<T, half>::value){
        cublasGemmEx(handle, transa, transb, 2, n, k,
                      &falpha, dA_colchk<T>, CUDA_R_16F, ldda_colchk, 
                      mat2_ptr, CUDA_R_16F, mat2_ld,
                      &fbeta, dC_colchk<T>, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    else{
      if (DEBUG) std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;
      if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, transa, transb, 2, n, k,
                    &falpha, dA_rowchk<T>, ldda_rowchk, 
                    mat2_ptr, mat2_ld, &fbeta, 
                    dC_colchk<T>, lddc_colchk);
      }
      else if constexpr(std::is_same<T, at::Half>::value){
        cublasGemmEx(handle, transa, transb, 2, n, k,
                      &falpha, dA_rowchk<T>, CUDA_R_16F, ldda_rowchk, 
                      mat2_ptr, CUDA_R_16F, mat2_ld,
                      &fbeta, dC_colchk<T>, CUDA_R_16F, lddc_colchk,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
    }
    // std::cout << "Output org dC_colchk: " << std::endl;
    // outputMatrixChk(dC_colchk, lddc_colchk, n*2, 1, 2, n);
    
    // dC_colchk + dBias_colchk
    if constexpr (std::is_same<T, float>::value){
      cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n,
                    &falpha, dBias_colchk, lddc_colchk, &falpha,
                    dC_colchk<T>, lddc_colchk,
                    dC_colchk<T>, lddc_colchk);
    }
    else if constexpr (std::is_same<T, at::Half>::value){
      dim3 blockSize(16, 16);
      dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (2 + blockSize.y - 1) / blockSize.y);
      addVector<T><<<blockSize, gridSize, (2*n)*sizeof(T), stream_main>>>(dC_colchk<T>, dBias_colchk, 2, n); 
    }
    // std::cout << "Output dC_colchk: " << std::endl;
    // outputMatrixChk(dC_colchk, lddc_colchk, n*2, 1, 2, n);
    
    if (DEBUG)  {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
      fullPath = homePath / destinationFile;
      printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*2*n*k*2/t1/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath,  t1, t1/t, (double)1*2*n*k*2/t1/1e6, (double)1*(2*k+k*n+2*n)*sizeof(T)/t1/1e6);     
    }
  }

  if (ROW_FT) {
      //std::cout << "  ROW_FT" << std::endl;
      if (DEBUG)  cudaEventRecord(start, stream_main);
      if (transb == CUBLAS_OP_N) {
        if (DEBUG) std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
        //we can further work on this to support trans A
        if constexpr (std::is_same<T, float>::value){
          cublasSgemm(handle, transa, transb,  m, 2, k,
                      &falpha, mat1_ptr, mat1_ld, 
                      dB_rowchk<T>, lddb_rowchk, &fbeta, 
                      dC_rowchk<T>, lddc_rowchk);
        }
        else if constexpr(std::is_same<T, at::Half>::value){
          cublasGemmEx(handle, transa, transb,  m, 2, k,
                        &falpha, mat1_ptr, CUDA_R_16F, mat1_ld, 
                        dB_rowchk<T>, CUDA_R_16F, lddb_rowchk,
                        &fbeta, dC_rowchk<T>, CUDA_R_16F, lddc_rowchk,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
      } else{
        if (DEBUG) std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
          if constexpr (std::is_same<T, float>::value){
            cublasSgemm(handle, transa, transb,  m, 2, k,
                        &falpha, mat1_ptr, mat1_ld, 
                        dB_colchk<T>, lddb_colchk, &fbeta, 
                        dC_rowchk<T>, lddc_rowchk);
          }
          else if constexpr(std::is_same<T, at::Half>::value){
            cublasGemmEx(handle, transa, transb,  m, 2, k,
                          &falpha, mat1_ptr, CUDA_R_16F, mat1_ld, 
                          dB_colchk<T>, CUDA_R_16F, lddb_colchk,
                          &fbeta, dC_rowchk<T>, CUDA_R_16F, lddc_rowchk,
                          CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          }
          // dC_rowchk + dBias_rowchk
          
      }
      // dC_rowchk + dBias_rowchk
      // std::cout << "Output org dC_colchk: " << std::endl;
      // outputMatrixChk(dC_rowchk, lddc_rowchk, m*2, 1, m, 2);
      // dC_colchk + dBias_colchk

      if constexpr (std::is_same<T, float>::value){ 
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 2,
                  &falpha, dBias_rowchk, lddc_rowchk, &falpha,
                  dC_rowchk<T>, lddc_rowchk,
                  dC_rowchk<T>, lddc_rowchk);
      }
      else if constexpr (std::is_same<T,at::Half >::value){
        dim3 blockSize(16, 16);
        dim3 gridSize((2 + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        addVector<T><<<blockSize, gridSize, (m*2)*sizeof(T),stream_main>>>(dC_rowchk<T>, dBias_rowchk, m, 2);
      }
      // std::cout << "Output dC_rowchk: " << std::endl;
      // outputMatrixChk(dC_rowchk, lddc_rowchk, m*2, 1, m, 2);
      // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
      if (DEBUG)  {
        cudaEventRecord(stop, stream_main);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t1, start, stop);
        destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
        fullPath = homePath / destinationFile;
        printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)1*m*2*k*2/t1/1e6, (double)1*(m*k+k*2+m*2)*sizeof(T)/t1/1e6);
        recordEffeciency(fullPath,  t1, t1/t, (double)1*m*2*k*2/t1/1e6, (double)1*(m*k+k*2+m*2)*sizeof(T)/t1/1e6);
      }
  }
  

  // --- check check-sum of C---//
  int threadNum = 0;
  if (DEBUG) std::cout << "------Check check-sum-------" << std::endl;
  if (COL_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    if (DEBUG) printf("dgemm-after-check-C-col\n");
    if (DEBUG) cudaEventRecord(start, stream_main);
    if constexpr (std::is_same<T, float>::value){
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &falpha, chk_v_a, ld_chk_v, 
                      result_ptr, result_ld, &fbeta, 
                      dC_colchk_r<T>, lddc_colchk_r);
    }
    else if constexpr(std::is_same<T, half>::value){
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,  2, mem_col, mem_row,
                      &falpha, chk_v_a, CUDA_R_16F, ld_chk_v, 
                      result_ptr, CUDA_R_16F, result_ld,
                      &fbeta, dC_colchk_r<T>, CUDA_R_16F, lddc_colchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    T E = 1;
    threadNum = (n+128-1)/128;
    detect_correct_col_Gemm<T><<<dim3(128), dim3(threadNum), 0, stream_main>>>(result_ptr, result_ld, E, n,
                                                                                      dC_colchk<T>, lddc_colchk, 
                                                                                      dC_colchk_r<T>, lddc_colchk_r);
    
    // cudaStreamSynchronize(stream_colchk);

    if (DEBUG)  {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
      fullPath = homePath / destinationFile;
      printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(1)*2*n*m*2/t1/1e6, (double)1*(m*n+2*m+2*n)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath,  t1, t1/t, (double)(1)*2*n*m*2/t1/1e6, (double)1*(m*n+2*m+2*n)*sizeof(T)/t1/1e6);
    }
  }
  

  if (ROW_FT && CHECK_AFTER) {
    mem_row = m;
    mem_col = n;
    
    if (DEBUG) printf("dgemm-after-check-C-row\n");
    if (DEBUG)  cudaEventRecord(start, stream_main);
    if constexpr (std::is_same<T, float>::value){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &falpha, result_ptr, result_ld, 
                      chk_v_b, ld_chk_v, &fbeta, 
                      dC_rowchk_r<T>, lddc_rowchk_r);
    }
    else if constexpr(std::is_same<T, half>::value){
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,  mem_row, 2, mem_col,
                      &falpha, result_ptr, CUDA_R_16F, result_ld, 
                      chk_v_b, CUDA_R_16F, ld_chk_v,
                      &fbeta, dC_rowchk_r<T>, CUDA_R_16F, lddc_rowchk_r,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    T E = 1;

    detect_correct_row_Gemm<T><<<dim3(128), dim3((m+128-1)/128), 0, stream_main>>>(result_ptr, result_ld, E, m, n,
                                                                          dC_rowchk<T>, lddc_rowchk,
                                                                          dC_rowchk_r<T>, lddc_rowchk_r);

    // cudaStreamSynchronize(stream_rowchk);
    if (DEBUG)  {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      destinationFile = "abftbgemm/records/effeciency/abftBias.txt";
      fullPath = homePath / destinationFile;
      printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(1)*m*2*n*2/t1/1e6, (double)1*(m*n+2*n+2*m)*sizeof(T)/t1/1e6);
      recordEffeciency(fullPath,  t1, t1/t, (double)(1)*m*2*n*2/t1/1e6, (double)1*(m*n+2*n+2*m)*sizeof(T)/t1/1e6);
    }
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
  
  // std::cout << "Using gemm_and_bias." << std::endl;
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
  // std::cout << "cublasLtMatmulAlgoGetHeuristic()" << std::endl;
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

  // std::cout << "cublasLtMatmul" << std::endl;
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