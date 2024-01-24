#include <ATen/ATen.h>
#include <cstdint>
#include <cuda_runtime.h>


void colchk_detect_correct(at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
			  				at::Half * dA_colchk,		int64_t ldda_colchk,
		          			at::Half * dA_colchk_r, 		int64_t ldda_colchk_r,
	         	  			int64_t num_batches,
                   	  		cudaStream_t stream);

void rowchk_detect_correct(at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
							at::Half * dA_rowchk,	int64_t ldda_rowchk,
							at::Half * dA_rowchk_r,	int64_t ldda_rowchk_r,
							int64_t num_batches,
							cudaStream_t stream);
