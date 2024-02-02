#include <stdio.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>


__global__ void
colchk_detect_correct_kernel(at::Half * dA, int64_t ldda, at::Half E, int64_t stridea,
						     at::Half * dA_colchk, 	int64_t ldda_colchk,	int64_t stride_colchk,
						     at::Half * dA_colchk_r, int64_t ldda_colchk_r,	int64_t stride_colchk_r){
    //printf("col_chk kernel func. \n");
	//determin the block to process
	// printf("determin the block to process. \n");
    dA = dA + blockIdx.x * stridea;
	dA_colchk = dA_colchk + blockIdx.x * stride_colchk;
	dA_colchk_r = dA_colchk_r + blockIdx.x * stride_colchk_r;
    
    //determine the specific colum to process
	// printf("determin the specific colum to process. \n");
    dA = dA + threadIdx.x * ldda;
    dA_colchk   = dA_colchk   + threadIdx.x * ldda_colchk;
    dA_colchk_r = dA_colchk_r + threadIdx.x * ldda_colchk_r;
	
    at::Half d1 = (*dA_colchk)       - (*dA_colchk_r);
    at::Half d2 = (*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
	
    //error detected
	// printf("error detected. \n");
    if(fabs(d1) > E) {
    	//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (double)d1, (double)d2, loc);
			
		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		//*(dA + loc) = *dA_colchk - sum;
    }
}

__global__ void
rowchk_detect_correct_kernel(at::Half * dA, int64_t ldda, at::Half E, int64_t stridea,
						     at::Half * dA_rowchk, 	int64_t ldda_rowchk,	int64_t stride_rowchk,
						     at::Half * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r){
    // printf("row_chk kernel func. \n");
	//determin the block to process
	// printf("determin the block to process. \n");
    dA = dA + blockIdx.x * stridea;
    dA_rowchk = dA_rowchk + blockIdx.x * stride_rowchk;
    dA_rowchk_r = dA_rowchk_r + blockIdx.x * stride_rowchk_r;
        
    //determine the specific row to process
	// printf("determin the specific row to process. \n");
	dA = dA + threadIdx.x;
    dA_rowchk   = dA_rowchk   + threadIdx.x;
    dA_rowchk_r = dA_rowchk_r + threadIdx.x;
	
    at::Half d1 = (*dA_rowchk)                 - (*dA_rowchk_r);
    at::Half d2 = (*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
	
    //error detected
	// printf("error detected. \n");
    if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[row check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n",(double)d1, (double)d2, loc);
			
		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for (int i = 0; i < ldda; i++) {
		    if (i != loc) {
			sum +=	*(dA + i * ldda); 
		    }
		}
        //correct the error
		// *(dA + loc * ldda) = *dA_rowchk - sum;
     }
}

void outputMatrix(at::Half *A, int64_t ld, int64_t stride, int64_t num_batches, int64_t row, int64_t col){
  size_t size = num_batches * (row * col) * sizeof(at::Half);
  at::Half *tensor;
  tensor = (at::Half *)malloc(size);
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

void colchk_detect_correct(at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
				           at::Half * dA_colchk,	int64_t ldda_colchk,
				           at::Half * dA_colchk_r, 	int64_t ldda_colchk_r,
						   int64_t num_batches,
                   		   cudaStream_t stream) {
	printf("col_detect_correct called \n");
	//error threshold 
	at::Half E = 1e-3;

	// at::Half *col_chk, *col_chk_r;
	// size_t size = num_batches * 2 * n * sizeof(at::Half);
  	// col_chk = (at::Half *)malloc(size);
	// col_chk_r = (at::Half *)malloc(size);
	// cudaMemcpy(col_chk, dA_colchk, size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(col_chk_r, dA_colchk_r, size, cudaMemcpyDeviceToHost);
	
	// printf("col_chk: %d \n", ldda_colchk);
	// outputMatrix(dA_colchk, ldda_colchk, 2*n, num_batches, 2, n);
	// printf("col_chk_r: %d \n",ldda_colchk_r);
	// outputMatrix(dA_colchk_r, ldda_colchk_r, 2*n, num_batches, 2, n);

	// int idxl = 0;
	// int idx = 0;
	// int idx_r = 0;
	// for(int num = 0; num < num_batches; num++){
	// 	for(int i = 0; i < 2; i++){
	// 		for(int j = 0; j < n; j++){
	// 			idxl = num*(2*n) + i*n + j;
	// 			idx = num*(2*n) + (j*ldda_colchk) + i;
	// 			idx_r = num*(2*n) + (j*ldda_colchk_r) + i;
	// 			at::Half d = fabs(col_chk[idxl] - col_chk_r[idxl]);
	// 			if(d > E){
	// 				printf("* col error: (%d, %d, %d): (%d:%.6f, %d:%.6f: %.6f. \n", 	\
	// 									  num, i, j, idxl, (double)col_chk[idxl], idxl, (double)col_chk_r[idxl], (double)d);
	// 			}

	// 			d = fabs(col_chk[idx]-col_chk_r[idx_r]);
	// 			if(d > E){
	// 				printf("  col error: (%d, %d, %d): (%d:%.6f, %d:%.6f): %.6f. \n", \
	// 									  num, i, j, idx, (double)col_chk[idx], idx_r, (double)col_chk_r[idx_r], (double)d);
	// 			}	
	// 		}
	// 	}
	// }
	// free(col_chk);
	// free(col_chk_r);

	colchk_detect_correct_kernel<<<dim3(num_batches), dim3(n), 0, stream>>>(dA, ldda, E, stridea,
											dA_colchk,		ldda_colchk,	(2*n),
											dA_colchk_r, 	ldda_colchk_r,	(2*n));
}

void rowchk_detect_correct(at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
					 	   at::Half * dA_rowchk,		int64_t ldda_rowchk,
						   at::Half * dA_rowchk_r,		int64_t ldda_rowchk_r,
						   int64_t num_batches,
						   cudaStream_t stream) {
	printf("row_detect_correct called \n");

	//error threshold 
	at::Half E = 1e-3;

	// at::Half *row_chk, *row_chk_r;
	// size_t size = num_batches * 2 * m * sizeof(at::Half);
  	// row_chk = (at::Half *)malloc(size);
	// row_chk_r = (at::Half *)malloc(size);
	// cudaMemcpy(row_chk, dA_rowchk, size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(row_chk_r, dA_rowchk_r, size, cudaMemcpyDeviceToHost);

	// printf("row_chk ld: %d \n", ldda_rowchk);
	// outputMatrix(dA_rowchk, ldda_rowchk, 2*m, num_batches, m, 2);
	// printf("row_chk_r ld: %d \n", ldda_rowchk_r);
	// outputMatrix(dA_rowchk_r, ldda_rowchk_r, 2*m, num_batches, m, 2);

	// int idxl = 0;
	// int idx = 0;
	// int idx_r = 0;
	// for(int num = 0; num < num_batches; num++){
	// 	for(int i = 0; i < m; i++){
	// 		for(int j = 0; j < 2; j++){
	// 			idxl = num*(2*m) + i*2 + j;
	// 			idx = num*(2*m) + (j*ldda_rowchk) + i;
	// 			idx_r = num*(2*m) + (j*ldda_rowchk_r) + i;
	// 			at::Half d = fabs(row_chk[idxl]-row_chk_r[idxl]);
	// 			if(d > E){
	// 				printf("* row error: (%d, %d, %d): (%d:%.6f, %d:%.6f): %.6f. \n", \
	// 									  num, i, j, idxl, (double)row_chk[idxl], idxl, (double)row_chk_r[idxl], (double)d);
	// 			}

	// 			d = fabs(row_chk[idx]-row_chk_r[idx_r]);
	// 			if(d > E){
	// 				printf("  row error: (%d, %d, %d): (%d:%.6f, %d:%.6f): %.6f. \n", \
	// 									  num, i, j, idx, (double)row_chk[idx], idx_r, (double)row_chk_r[idx_r], (double)d);
	// 			}	
	// 		}
	// 	}
	// }
	// free(row_chk);
	// free(row_chk_r);
	
	rowchk_detect_correct_kernel<<<dim3(num_batches), dim3(m), 0, stream>>>(dA, ldda, E, stridea,
											dA_rowchk, ldda_rowchk,		(2*m),
											dA_rowchk_r, ldda_rowchk_r,	(2*m));
}
