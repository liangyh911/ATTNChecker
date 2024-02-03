#include <cstdint>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>

void col_chk_enc(cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n,
                 float *A, int64_t lda, int64_t stridea, 
                 float * chk_v, int64_t ld_chk_v,
                 float * dcolchk, int64_t ld_dcolchk,
                 int64_t num_batches);

void row_chk_enc(cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n,
                 float * A, int64_t lda, int64_t stridea,
                 float * chk_v, int64_t ld_chk_v,
                 float * drowchk, int64_t ld_drowchk,
                 int64_t num_batches);

void col_chk_enc(cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n,
                 at::Half *A, int64_t lda, int64_t stridea, 
                 at::Half * chk_v, int64_t ld_chk_v,
                 at::Half * dcolchk, int64_t ld_dcolchk,
                 int64_t num_batches);

void row_chk_enc(cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n,
                 at::Half * A, int64_t lda, int64_t stridea,
                 at::Half * chk_v, int64_t ld_chk_v,
                 at::Half * drowchk, int64_t ld_drowchk,
                 int64_t num_batches);
