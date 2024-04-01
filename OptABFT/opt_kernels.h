#include <cstdint>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <cuda_fp16.h>

/*
template<typename T, int NROW, int NCOL, int C>
void encode_col_lancher(int num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					T *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk,
					cudaStream_t stream_colchk);

template<typename T, int NROW, int NCOL>
void encode_row_lancher(int num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					 T *dA_rowchk, int64_t ldda_rowchk, int64_t strideA_rowchk,
					 cudaStream_t stream_rowchk);

template<typename T, int NROW, int NCOL, int C>
void update_col_lancher(int num_batches,
					T *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk, 
					T *dB, int64_t lddb, int64_t strideB, 
					T *dC_colchk, int64_t lddc_colchk, int64_t strideC_colchk,
					cudaStream_t stream_colchk);

template<typename T, int NROW, int NCOL>
void update_row_lancher(int num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					T *dB_rowchk, int64_t lddb_rowchk, int64_t strideB_rowchk,
					T *dC_rowchk, int64_t lddc_rowchk, int64_t strideC_rowchk,
					cudaStream_t stream_rowchk);
*/
void detect_correct_col_lancher(float * dA, int64_t ldda, float E, int64_t stridea,
						     float * dA_colchk, 	int64_t ldda_colchk,	int64_t stride_colchk,
						     float * dA_colchk_r, int64_t ldda_colchk_r,	int64_t stride_colchk_r,
							 int numBlock, int64_t numThread, cudaStream_t stream_colchk);

void detect_correct_row_lancher(float * dA, int64_t ldda, float E, int64_t stridea,
						     float * dA_rowchk, 	int64_t ldda_rowchk,	int64_t stride_rowchk,
						     float * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r,
							 int numBlock, int64_t numThread, cudaStream_t stream_rowchk);

void detect_correct_col_lancher(at::Half * dA, int64_t ldda, float E, int64_t stridea,
						     at::Half * dA_colchk, 	int64_t ldda_colchk,	int64_t stride_colchk,
						     at::Half * dA_colchk_r, int64_t ldda_colchk_r,	int64_t stride_colchk_r,
							 int numBlock, int64_t numThread, cudaStream_t stream_colchk);

void detect_correct_row_lancher(at::Half * dA, int64_t ldda, float E, int64_t stridea,
						     at::Half * dA_rowchk, 	int64_t ldda_rowchk,	int64_t stride_rowchk,
						     at::Half * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r,
							 int numBlock, int64_t numThread, cudaStream_t stream_rowchk);
