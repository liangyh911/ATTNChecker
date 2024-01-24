#include <cstdio>
#undef max
#undef min
#include "./abft_encoder.h"
#include "./abft_corrector.h"
#include <cstdint>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

void abft_checker_colchk(char transa, char transb,
						 at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
						 at::Half * dA_colchk,    int64_t ldda_colchk, 
						 at::Half * dA_colchk_r,  int64_t ldda_colchk_r,
						 at::Half * dev_chk_v,    int64_t ld_dev_chk_v,
						 bool DEBUG, cudaStream_t stream, int64_t num_batches){
	if (DEBUG) printf("abft_checker_colchk\n");
	col_chk_enc(transa, transb, m, n, 
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

void abft_checker_rowchk(char transa, char transb, 
					at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
					at::Half * dA_rowchk,    int64_t ldda_rowchk,
    				at::Half * dA_rowchk_r,  int64_t ldda_rowchk_r,
    				at::Half * dev_chk_v,    int64_t ld_dev_chk_v,
    				bool DEBUG, cudaStream_t stream, int64_t num_batches){
	if (DEBUG) printf("abft_checker_rowchk\n");
	row_chk_enc(transa, transb, m, n, 
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

