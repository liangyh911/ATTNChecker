#include <cstdio>
#undef max
#undef min
#include "./abft_encoder.h"
#include "./abft_corrector.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


void abft_checker_colchk(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
						 float * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
						 float * dA_colchk,    int64_t ldda_colchk, 
						 float * dA_colchk_r,  int64_t ldda_colchk_r,
						 float * dev_chk_v,    int64_t ld_dev_chk_v,
						 bool DEBUG, cudaStream_t stream, int64_t num_batches){
	if (DEBUG) printf("abft_checker_colchk\n");
	col_chk_enc(handle, m, n, 
                dA, ldda, stridea,
                dev_chk_v, ld_dev_chk_v, 
                dA_colchk_r, ldda_colchk_r,
                num_batches);

    printf("colchk_detect_correct.\n");
    colchk_detect_correct(dA, ldda, m, n, stridea,
				          dA_colchk,	ldda_colchk,
				          dA_colchk_r, 	ldda_colchk_r,
						  num_batches,
						  stream);
}

void abft_checker_rowchk(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
					float * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
					float * dA_rowchk,    int64_t ldda_rowchk,
    				float * dA_rowchk_r,  int64_t ldda_rowchk_r,
    				float * dev_chk_v,    int64_t ld_dev_chk_v,
    				bool DEBUG, cudaStream_t stream, int64_t num_batches){
	if (DEBUG) printf("abft_checker_rowchk\n");
	row_chk_enc(handle, m, n, 
                dA, ldda, stridea,
                dev_chk_v, ld_dev_chk_v, 
                dA_rowchk_r, ldda_rowchk_r,
                num_batches);

	printf("rowchk_detect_correct.\n");
	rowchk_detect_correct(dA, ldda, m, n, stridea,
							dA_rowchk,		ldda_rowchk,
							dA_rowchk_r, 	ldda_rowchk_r,
							num_batches,
							stream);
}


// void abft_checker_colchk(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
// 						 half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
// 						 half * dA_colchk,    int64_t ldda_colchk, 
// 						 half * dA_colchk_r,  int64_t ldda_colchk_r,
// 						 half * dev_chk_v,    int64_t ld_dev_chk_v,
// 						 bool DEBUG, cudaStream_t stream, int64_t num_batches){
// 	if (DEBUG) printf("abft_checker_colchk\n");
// 	col_chk_enc(handle, m, n, 
//                 dA, ldda, stridea,
//                 dev_chk_v, ld_dev_chk_v, 
//                 dA_colchk_r, ldda_colchk_r,
//                 num_batches);

//     printf("colchk_detect_correct.\n");
//     colchk_detect_correct(dA, ldda, m, n, stridea,
// 				          dA_colchk,	ldda_colchk,
// 				          dA_colchk_r, 	ldda_colchk_r,
// 						  num_batches,
// 						  stream);
// }

// void abft_checker_rowchk(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
// 					half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
// 					half * dA_rowchk,    int64_t ldda_rowchk,
//     				half * dA_rowchk_r,  int64_t ldda_rowchk_r,
//     				half * dev_chk_v,    int64_t ld_dev_chk_v,
//     				bool DEBUG, cudaStream_t stream, int64_t num_batches){
// 	if (DEBUG) printf("abft_checker_rowchk\n");
// 	row_chk_enc(handle, m, n, 
//                 dA, ldda, stridea,
//                 dev_chk_v, ld_dev_chk_v, 
//                 dA_rowchk_r, ldda_rowchk_r,
//                 num_batches);

// 	printf("rowchk_detect_correct.\n");
// 	rowchk_detect_correct(dA, ldda, m, n, stridea,
// 							dA_rowchk,		ldda_rowchk,
// 							dA_rowchk_r, 	ldda_rowchk_r,
// 							num_batches,
// 							stream);
// }