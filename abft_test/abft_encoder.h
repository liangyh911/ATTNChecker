#include <cstdint>
#include <cublas_v2.h>
#include <cuda_fp16.h>

void col_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
                 float *A, int64_t lda, int64_t stridea, 
                 float * chk_v, int64_t ld_chk_v,
                 float * dcolchk, int64_t ld_dcolchk,
                 int64_t num_batches);

void row_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
                 float * A, int64_t lda, int64_t stridea,
                 float * chk_v, int64_t ld_chk_v,
                 float * drowchk, int64_t ld_drowchk,
                 int64_t num_batches);

// void col_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
//                  half *A, int64_t lda, int64_t stridea, 
//                  half * chk_v, int64_t ld_chk_v,
//                  half * dcolchk, int64_t ld_dcolchk,
//                  int64_t num_batches);

// void row_chk_enc(cublasHandle_t handle, int64_t m, int64_t n,
//                  half * A, int64_t lda, int64_t stridea,
//                  half * chk_v, int64_t ld_chk_v,
//                  half * drowchk, int64_t ld_drowchk,
//                  int64_t num_batches);
