#include <cstdint>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>

void abft_checker_colchk(cublasHandle_t handle, char transa, char transb,
							float * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
							float * dA_colchk,    int64_t ldda_colchk, 
							float * dA_colchk_r,  int64_t ldda_colchk_r,
							float * dev_chk_v,    int64_t ld_dev_chk_v,
							bool DEBUG, cudaStream_t stream, int64_t num_batches);

void abft_checker_rowchk(cublasHandle_t handle, char transa, char transb, 
							float * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
							float * dA_rowchk,    int64_t ldda_rowchk,
							float * dA_rowchk_r,  int64_t ldda_rowchk_r,
							float * dev_chk_v,    int64_t ld_dev_chk_v,
							bool DEBUG, cudaStream_t stream, int64_t num_batches);


void abft_checker_colchk(cublasHandle_t handle, char transa, char transb,
							at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
							at::Half * dA_colchk,    int64_t ldda_colchk, 
							at::Half * dA_colchk_r,  int64_t ldda_colchk_r,
							at::Half * dev_chk_v,    int64_t ld_dev_chk_v,
							bool DEBUG, cudaStream_t stream, int64_t num_batches);

void abft_checker_rowchk(cublasHandle_t handle, char transa, char transb, 
							at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
							at::Half * dA_rowchk,    int64_t ldda_rowchk,
							at::Half * dA_rowchk_r,  int64_t ldda_rowchk_r,
							at::Half * dev_chk_v,    int64_t ld_dev_chk_v,
							bool DEBUG, cudaStream_t stream, int64_t num_batches);

