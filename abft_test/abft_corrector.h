#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

void colchk_detect_correct(float * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
			  				float * dA_colchk,		int64_t ldda_colchk,
		          			float * dA_colchk_r, 		int64_t ldda_colchk_r,
	         	  			int64_t num_batches,
                   	  		cudaStream_t stream);
void rowchk_detect_correct(float * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
							float * dA_rowchk,	int64_t ldda_rowchk,
							float * dA_rowchk_r,	int64_t ldda_rowchk_r,
							int64_t num_batches,
							cudaStream_t stream);


// void colchk_detect_correct(half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
// 			  				half * dA_colchk,		int64_t ldda_colchk,
// 		          			half * dA_colchk_r, 		int64_t ldda_colchk_r,
// 	         	  			int64_t num_batches,
//                    	  		cudaStream_t stream);
// void rowchk_detect_correct(half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
// 							half * dA_rowchk,	int64_t ldda_rowchk,
// 							half * dA_rowchk_r,	int64_t ldda_rowchk_r,
// 							int64_t num_batches,
// 							cudaStream_t stream);
