#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdint>
#include <stdlib.h>
#include <iostream>
#include <ATen/ATen.h>

template <typename T>
__global__ void
detect_correct_col(T * dA, int64_t ldda, T E, int64_t stridea,
						     T * dA_colchk, 	int64_t ldda_colchk,	int64_t stride_colchk,
						     T * dA_colchk_r, int64_t ldda_colchk_r,	int64_t stride_colchk_r){
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
	
    T d1 = (*dA_colchk)       - (*dA_colchk_r);
    T d2 = (*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
	
    //error detected
	// printf("error detected. \n");
    if(fabs(d1) > E) {
    	//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
    }
}

template<typename T>
__global__ void
detect_correct_row(T * dA, int64_t ldda, T E, int64_t stridea,
						     T * dA_rowchk, 	int64_t ldda_rowchk,	int64_t stride_rowchk,
						     T * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r){
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
	
    T d1 = (*dA_rowchk)                 - (*dA_rowchk_r);
    T d2 = (*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
	
    //error detected
	// printf("error detected. \n");
    if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[row check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for (int i = 0; i < ldda; i++) {
		    if (i != loc) {
				sum +=	*(dA + i * ldda); 
		    }
		}
        //correct the error
		*(dA + loc * ldda) = *dA_rowchk - sum;
     }
}
// void detect_correct_row_lancher(float * dA, int64_t ldda, float E, int64_t stridea,
// 						     float * dA_rowchk, 	int64_t ldda_rowchk,	int64_t stride_rowchk,
// 						     float * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r,
// 							 int numBlock, int64_t numThread, cudaStream_t stream_rowchk){
	
// 	detect_correct_row<<<dim3(numBlock), dim3(numThread), 0, stream_rowchk>>>(dA, ldda, E, stridea,
//                                           dA_rowchk, ldda_rowchk,     stride_rowchk,
//                                           dA_rowchk_r, ldda_rowchk_r, stride_rowchk_r);
// }

__global__ void
detect_correct_col(at::Half * dA, int64_t ldda, at::Half E, int64_t stridea,
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
	
    float d1 = (float)((*dA_colchk)       - (*dA_colchk_r));
    float d2 = (float)((*(dA_colchk + 1)) - (*(dA_colchk_r + 1)));
	
    //error detected
	// printf("error detected. \n");
    if(fabs(d1) > E) {
    	//locate the error
		// int loc = __half2int_rn(d2 / d1) - 1;
		int loc = round(d2 / d1) - 1;
		printf("[col check]error detected (val1 = %.6f, val2 = %.6f), (d1 = %.6f, d2 = %.6f, loc = %d) \n", \
												(float)(*dA_colchk), (float)(*dA_colchk_r), (float)(d1), (float)(d2), loc);
			
		//the sum of the rest correct number except the error one
		at::Half sum = 0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum = sum + (*(dA + i)); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
    }
}
// void detect_correct_col_lancher(at::Half * dA, int64_t ldda, float E, int64_t stridea,
// 						     at::Half * dA_colchk, 	int64_t ldda_colchk,	int64_t stride_colchk,
// 						     at::Half * dA_colchk_r, int64_t ldda_colchk_r,	int64_t stride_colchk_r,
// 							 int numBlock, int64_t numThread, cudaStream_t stream_colchk){
	
// 	detect_correct_col<<<dim3(numBlock), dim3(numThread), 0, stream_colchk>>>(dA, ldda, E, stridea,
//                                             dA_colchk,      ldda_colchk,    stride_colchk,
//                                             dA_colchk_r,    ldda_colchk_r,  stride_colchk_r);
// }

__global__ void
detect_correct_row(at::Half * dA, int64_t ldda, at::Half E, int64_t stridea,
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
	
    float d1 =  (float)((*dA_rowchk)                 - (*dA_rowchk_r));
    float d2 =  (float)((*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r)));
	
    //error detected
	// printf("error detected. \n");
    if(float(d1) > E) {
		//locate the error
		// int loc = __half2int_rn(d2 / d1) - 1;
		int loc = round(d2 / d1) - 1;
		printf("[row check]error detected (val1 = %.6f, val2 = %.6f), (d1 = %.6f, d2 = %.6f, loc = %d) \n", \
												(float)(*dA_rowchk), (float)(*dA_rowchk_r), (float)(d1), (float)(d2), loc);
			
		//the sum of the rest correct number except the error one
		at::Half sum = 0.0;
		for (int i = 0; i < ldda; i++) {
		    if (i != loc) {
				sum += *(dA + i * ldda); 
		    }
		}
        //correct the error
		*(dA + loc * ldda) = *dA_rowchk - sum;
     }
}
// void detect_correct_row_lancher(at::Half * dA, int64_t ldda, float E, int64_t stridea,
// 						     at::Half * dA_rowchk, 	int64_t ldda_rowchk,	int64_t stride_rowchk,
// 						     at::Half * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r,
// 							 int numBlock, int64_t numThread, cudaStream_t stream_rowchk){
	
// 	detect_correct_row<<<dim3(numBlock), dim3(numThread), 0, stream_rowchk>>>(dA, ldda, E, stridea,
//                                           dA_rowchk, ldda_rowchk,     stride_rowchk,
//                                           dA_rowchk_r, ldda_rowchk_r, stride_rowchk_r);
// }