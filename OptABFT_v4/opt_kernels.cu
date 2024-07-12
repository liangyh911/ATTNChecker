#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdint>
#include <stdlib.h>
#include <iostream>
#include <ATen/ATen.h>
#include <cmath>


/*
__constant__ float CHK_V_A[2*72];

__global__ void encode_col_v1(int m, int k, int num_batches,
					float *dA, int64_t ldda, int64_t strideA, 
                     float *chk_v, int64_t ld_chk_v, int64_t stride_chk_v,
					 float *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk) {

	const int batch_id = blockIdx.x;
	const int x = threadIdx.x;
	const int y = threadIdx.y;
	dA = dA + batch_id * strideA;
	chk_v = chk_v + batch_id * stride_chk_v;
	dA_colchk = dA_colchk + batch_id * strideA_colchk;

	float res = 0.0;
	for (int i = 0; i < m; i++) {
		res += chk_v[y + i * ld_chk_v] * dA[i + x * ldda];
	}
	dA_colchk[y + x * ldda_colchk] = res;
}

__global__ void encode_col_v2(int m, int k, int num_batches,
					float *dA, int64_t ldda, int64_t strideA, 
                     float *chk_v, int64_t ld_chk_v, int64_t stride_chk_v,
					 float *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk) {

	const int batch_id = blockIdx.x;
	const int x = threadIdx.x;
	const int y = threadIdx.y;
	dA = dA + batch_id * strideA;
	// chk_v = chk_v + batch_id * stride_chk_v;
	dA_colchk = dA_colchk + batch_id * strideA_colchk;

	float res = 0.0;
	for (int i = 0; i < m; i++) {
		res += CHK_V_A[y + i * ld_chk_v] * dA[i + x * ldda];
	}
	dA_colchk[y + x * ldda_colchk] = res;
}

__global__ void encode_col_v3(int m, int k, int num_batches,
					float *dA, int64_t ldda, int64_t strideA, 
                     float *chk_v, int64_t ld_chk_v, int64_t stride_chk_v,
					 float *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk) {

	extern __shared__ float dA_sm [];
	int ldda_sm = m; 

	const int batch_id = blockIdx.x;
	const int x = threadIdx.x;
	const int y = threadIdx.y;
	dA = dA + batch_id * strideA;
	chk_v = chk_v + batch_id * stride_chk_v;
	dA_colchk = dA_colchk + batch_id * strideA_colchk;

	// if (y == 0)	{
		for (int i = 0; i < k/2; i++) {
			dA_sm[x + (i+y*k/2) * ldda_sm] = dA[x + (i+y*k/2) * ldda];
		}
	// }
	__syncthreads();

	if (x < k) {
		float res = 0.0;
		for (int i = 0; i < m; i++) {
			// if (x == 0 && y == 0)
			// printf("%f %f %f\n", chk_v[y + i * ld_chk_v], dA_sm[i + x * ldda_sm], res);
			res += chk_v[y + i * ld_chk_v] * dA_sm[i + x * ldda_sm];
		}
		dA_colchk[y + x * ldda_colchk] = res;
	}
}

__global__ void encode_col_v4(int m, int k, int num_batches,
					float *dA, int64_t ldda, int64_t strideA, 
                     float *chk_v, int64_t ld_chk_v, int64_t stride_chk_v,
					 float *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk) {

	extern __shared__ float dA_sm [];
	int ldda_sm = m; 

	const int batch_id = blockIdx.x;
	const int x = threadIdx.x;
	const int y = threadIdx.y;
	dA = dA + batch_id * strideA;
	chk_v = chk_v + batch_id * stride_chk_v;
	dA_colchk = dA_colchk + batch_id * strideA_colchk;

	for (int i = 0; i < k/2; i++) {
		dA_sm[x + (i+y*k/2) * ldda_sm] = dA[x + (i+y*k/2) * ldda];
	}

	__syncthreads();

	if (x < k) {
		float res = 0.0;
		for (int i = 0; i < m; i++) {
			// if (x == 0 && y == 0)
			// printf("%f %f %f\n", chk_v[y + i * ld_chk_v], dA_sm[i + x * ldda_sm], res);
			res += 1 * dA_sm[i + x * ldda_sm];
		}
		dA_colchk[y + x * ldda_colchk] = res;
	}
}
*/

template <typename T>
struct SharedMemory
{
    // Ensure that we won't compile any un-specialized types
    __device__ T *getPointer()
    {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};
template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};
template <>
struct SharedMemory <at::Half>
{
    __device__ at::Half *getPointer()
    {
        extern __shared__ at::Half s_half[];
        return s_half;
    }
};

template<class T, int64_t NROW, int64_t NCOL, int64_t C>
__global__ void encode_col_v5(int64_t num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					 T *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk) {

	SharedMemory<T> smem;
 	T* dA_sm = smem.getPointer();
	
	// extern __shared__ T dA_sm [];

	const int batch_id = blockIdx.x;
	const int tid = threadIdx.x;
	const int y_load = tid / NROW;
	const int x_load = tid % NROW;
	const int y_compute = tid / NCOL;
	const int x_compute = tid % NCOL;
	dA = dA + batch_id * strideA;
	dA_colchk = dA_colchk + batch_id * strideA_colchk;

	for (int i = 0; i < NCOL; i += C) {
		dA_sm[x_load+(NROW+1)*(i+y_load)] = dA[x_load+(NROW)*(i+y_load)];
	}	
	__syncthreads();

	if (x_compute < NCOL && y_compute < 2) {
		T res = 0.0;
		T * dA_col = &dA_sm[x_compute * (NROW+1)];
		if (y_compute == 0) {
			for (int i = 0; i < NROW; i++) {
				res += dA_col[i];
			}
		}
		if (y_compute == 1) {
			for (int i = 0; i < NROW; i++) {
				res += (T)(i+1) * dA_col[i];
			}
		}
		dA_colchk[y_compute + x_compute * ldda_colchk] = res;
	}
}

template<typename T, int64_t NROW, int64_t NCOL>
__global__ void encode_row_v5(int num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					 T *dA_rowchk, int64_t ldda_rowchk, int64_t strideA_rowchk) {

	const int batch_id = blockIdx.x;
	const int tid = threadIdx.x;
	const int y = tid / NROW;
	const int x = tid % NROW;
	dA = dA + batch_id * strideA;
	dA_rowchk = dA_rowchk + batch_id * strideA_rowchk;

	// printf("%d %d\n", x, y);

	if (x < NROW && y < 2) {
		T res = 0.0;
		T * dA_row = &dA[x];
		if (y == 0) {
			for (int i = 0; i < NCOL; i++) {
				res += dA_row[i * NROW];
			}
		}
		if (y == 1) {
			for (int i = 0; i < NCOL; i++) {
				res += (T)(i+1) * dA_row[i * NROW];
			}
		}
		dA_rowchk[y * NROW + x] = res;
	}
}


template<class T, int64_t NROW, int64_t NCOL, int C>
__global__ void update_col_v5(int64_t num_batches,
					T *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk, 
					T *dB, int64_t lddb, int64_t strideB, 
					T *dC_colchk, int64_t lddc_colchk, int64_t strideC_colchk) {

	// extern __shared__ T sm [];

	SharedMemory<T> smem;
 	T* sm = smem.getPointer();

	T * dA_colchk_sm = sm;
	T * dB_sm = sm + 2*NROW;

	const int batch_id = blockIdx.x;
	const int tid = threadIdx.x;
	int y_load = tid / NROW;
	int x_load = tid % NROW;

	const int y_compute = tid / NCOL;
	const int x_compute = tid % NCOL;
	dA_colchk = dA_colchk + batch_id * strideA_colchk;
	dB = dB + batch_id * strideB;
	dC_colchk = dC_colchk + batch_id * strideC_colchk;

	if (tid < 2*NROW) {
		dA_colchk_sm[tid] = dA_colchk[tid];
	}
	for (int i = 0; i < NCOL; i += C) {
		dB_sm[x_load+(NROW+1)*(i+y_load)] = dB[x_load+(NROW)*(i+y_load)];
	}	
	__syncthreads();

	// printf("%d %d\n", x, y);

	if (x_compute < NCOL && y_compute < 2) {
		T res = 0.0;
		T * row = &dA_colchk_sm[y_compute];
		// T * row = &dA_colchk[y_compute];
		T * col = &dB_sm[x_compute * (NROW+1)];
		for (int i = 0; i < NROW; i++) {
			res += row[i * 2] * col[i];
			// res += 1 * col[i];
		}

		dC_colchk[y_compute + x_compute * 2] = res;
	}
}


template<typename T, int64_t NROW, int64_t NCOL>
__global__ void update_row_v5(int64_t num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					T *dB_rowchk, int64_t lddb_rowchk, int64_t strideB_rowchk,
					T *dC_rowchk, int64_t lddc_rowchk, int64_t strideC_rowchk) {

	// extern __shared__ T dB_rowchk [];

	const int batch_id = blockIdx.x;
	const int tid = threadIdx.x;
	const int y = tid / NROW;
	const int x = tid % NROW;
	dA = dA + batch_id * strideA;
	dB_rowchk = dB_rowchk + batch_id * strideB_rowchk;
	dC_rowchk = dC_rowchk + batch_id * strideC_rowchk;


	// printf("%d %d\n", x, y);

	if (x < NROW && y < 2) {
		T res = 0.0;
		T * row = &dA[x];
		T * col = &dB_rowchk[y*NCOL];
		for (int i = 0; i < NCOL; i++) {
			res += col[i] * row[i * NROW];
		}
		dC_rowchk[y * NROW + x] = res;
	}
}

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
	
    float d1 = (float)(*dA_colchk)       - (*dA_colchk_r);
    float d2 = (float)(*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
	float abs_d1 = fabs(d1);
	int loc = -1;
	int locT = -1;
	float MAX;

	// E < abs d1 < INF
	if(abs_d1 > E && !isinf(abs_d1) && !isnan(abs_d1)) {
		if(!isinf(d2)){
			// d2 != INF
			//locate the error
			
			int counter = 0;
			// if more than one large number in the col
			for(int i = 0; i < ldda; i++){
				if(fabs((float)*(dA+i)) > 1e10){
					counter++;
					if(counter > 1){
						printf("[col check]col chksum error, more than one large number.\n");
						return;
					}
				}
			}

			loc = round(d2 / d1) - 1;
			printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n",  (float)d1, (float)d2, loc);
			//correction
			// *(dA+loc) += d1;
			if(abs_d1 > (float)1e5){
				// printf("d1 > threshold.\n");
				T sum = 0.0;
				for(int i = 0; i < ldda; i++) {
					if (i != loc) {
						sum +=	*(dA + i); 
					}
				}
				//correct the error
				*(dA + loc) = *dA_colchk - sum;
			}
			else{
				// printf("d1 =< threshold.\n");
				*(dA + loc) += d1;
			} 
		}
		else{
			if(isinf(*(dA_colchk + 1))){
				// C1,j == INF
				printf("[col check]Error detected in INPUTS.\n");
				return;
			}
			else{
				// C1,j != INF
				MAX = 0;
				int counter = 0;
				for(int i = 0; i < ldda; i++) {
					if(fabs((float)*(dA+i)) > MAX){
						MAX = fabs((float)*(dA+i));
						loc = i;
					}
					// if((*(dA+i)) < MIN){
					// 	MIN = *(dA+i);
					// 	locT = i;
					// }
					if(fabs((float)*(dA+i)) > 1e10){
						counter++;
						if(counter > 1){
							printf("[col check]col chksum error, more than one large number.\n");
							return;
						}
					}
				}
				// if(fabs(((float)MAX)) < fabs(((float)MIN))){
				// 	loc = locT;
				// }
				printf("[col check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
				//correction
				// *(dA+loc) += d1;
				if(abs_d1 > (float)1e5){
					// printf("d1 > threshold.\n");
					T sum = 0.0;
					for(int i = 0; i < ldda; i++) {
						if (i != loc) {
							sum +=	*(dA + i); 
						}
					}
					//correct the error
					*(dA + loc) = *dA_colchk - sum;
				}
				else{
					// printf("d1 =< threshold.\n");
					*(dA + loc) += d1;
				} 
			}
		}
		return;
	}
	// abs = inf
	if(isinf(abs_d1)){
		MAX = 0;
		int64_t counter = 0;
		for(int i = 0; i < ldda; i++) {
			if(fabs((float)*(dA+i)) > MAX){
				MAX = fabs((float)*(dA+i));
				loc = i;
			}
			// if(*(dA+i) < MIN){
			// 	MIN = *(dA+i);
			// }
			if(isinf(*(dA+i)) || fabs((float)*(dA+i)) > 1e10){
				counter++;
				if(counter > 1){
					printf("[col check]Multi INFs or Large Number detected in one column.\n");
					return;
				}
			}
		}
		if(counter == 0){
			printf("[col chk]No INF or Large Number found.\n");
			return;
		}
		// if(fabs((T)MAX) < fabs((T)MIN)){
		// 	MAX = MIN;
		// }
		// for(int i = 0; i < ldda; i++) {
		// 	if (fabs((float)*(dA+i)) == MAX || isinf(*(dA+i))) {
		// 		loc = i;
		// 		break;
		// 	}
		// }
		printf("[col check]INF detected (d1 = %.6f, d2 = %.6f, loc = %d, %.6f, %.6f) \n", 
											(float)d1, (float)d2, loc,(float)*(dA+29), (float)*(dA+loc));
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
		return;
	}
	// abs == nan
	if(isnan(abs_d1)){
		int64_t counter = 0;
		for(int i = 0; i < ldda; i++) {
			if (isnan(*(dA+i))) {
				loc = i;
				counter++;
			}
			if(isinf(*(dA+i))){
				counter++;
			}
			if(counter > 1){
				printf("[col check]Multi INF or NAN detected in one column. \n");
				return;
			}
		}
		if(loc == -1){
			printf("[col check]No found NAN for d1 = NAN (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
			return;
		}
		printf("[col check]NAN detected (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",  
											blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
		return;
	}
}

template<typename T>
__global__ void
detect_correct_row(T * dA, int64_t ldda, T E, int64_t stridea, int64_t col,
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
	
    float d1 = (float)(*dA_rowchk)                 - (*dA_rowchk_r);
    float d2 = (float)(*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
	float abs_d1 = fabs(d1);
	int loc = -1;
	int locT = -1;
	float MAX;

	if(abs_d1 > E && !isinf(abs_d1) && !isnan(abs_d1)) {
		if(!isinf(d2)){
			// d2 != INF
			//locate the error
			int counter = 0;
			// if more than one large number
			for(int i = 0; i < col; i++){
				if(fabs((float)*(dA + i*ldda)) > 1e10){
					counter++;
					if(counter > 1){
						printf("[row check]row chksum error. More than one Large Number detected. \n");
						return;
					}
				}
			}
			if(counter == 0){
				printf("[row chk]Recaculate row chk. No Large Number detected for d1 = INF (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d, %.6f) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc, (float)*(dA+21*ldda));
				T sum = 0.0;
				T sumW = 0.0;
				for(int i = 0; i < col; i++) {
					sum +=	*(dA + i * ldda); 
					sumW += (i+1)*(*(dA + i * ldda));
				}
				*(dA_rowchk) = sum;
				*(dA_rowchk + ldda_rowchk) = sumW;
				return;
			}
			loc = round(d2 / d1) - 1;
			printf("[row check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			//correction
			// *(dA + loc * ldda) += d1;
			if(abs_d1 > (float)1e5){
				// printf("d1 > threshold.\n");
				T sum = 0.0;
				for (int i = 0; i < col; i++) {
					if (i != loc) {
						sum +=	*(dA + i * ldda); 
					}
				}
				*(dA + loc * ldda) = *dA_rowchk - sum;
			}
			else{
				// printf("d1 =< threshold.\n");
				*(dA + loc * ldda) += d1;
			} 	
		}
		else{
			if(isinf(*(dA_rowchk + ldda_rowchk))){
				// C1,j == INF
				// two cases: 1. inp matrxi has error; 2. inp matrix not error
				T sum = 0.0;
				T sumW = 0.0;
				for(int i = 0; i < col; i++) {
					if(fabs((float)*(dA+i*ldda)) > 1e10 || isinf(*(dA+i*ldda))){
						printf("[row check]Error detected in INPUTS.(loc = %d)\n", i);
						return;
					}
					else{
						sum +=	*(dA + i * ldda); 
						sumW += (i+1)*(*(dA + i * ldda));
					}
				}
				*(dA_rowchk) = sum;
				*(dA_rowchk + ldda_rowchk) = sumW;
				printf("[row check]Recaculate row chk. No found Large Number or INF for d1 = INF (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
				return;
			}
			else{
				// C1,j != INF
				MAX = 0;
				int counter = 0;
				for(int i = 0; i < col; i++) {
					if(fabs((float)*(dA + i * ldda)) > MAX){
						MAX = fabs((float)*(dA + i * ldda));
						loc = i;
					}
					// if(*(dA + i * ldda) < MIN){
					// 	MIN = *(dA+i*ldda);
					// 	locT = i;
					// }
					if(fabs((float)*(dA + i * ldda)) > 1e10){
						counter++;
						if(counter > 1){
							printf("[row check]row chksum error. More than one Large Number detected. \n");
							return;
						}
					}
				}
				// if(fabs((T)MAX) < fabs((T)MIN)){
				// 	loc = locT;
				// }
				printf("[row check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
				//correction
				// *(dA + loc * ldda) += d1;
				// correction
				if(abs_d1 > (float)1e5){
					// printf("d1 > threshold.\n");
					T sum = 0.0;
					for (int i = 0; i < col; i++) {
						if (i != loc) {
							sum +=	*(dA + i * ldda); 
						}
					}
					*(dA + loc * ldda) = *dA_rowchk - sum;
				}
				else{
					// printf("d1 =< threshold.\n");
					*(dA + loc * ldda) += d1;
				} 		
			}
		}
		return;
	}
	// abs d1 = INF
	if(isinf(abs_d1)){
		// abs == inf
		int64_t counter = 0;
		MAX = 0;
		for(int i = 0; i < col; i++) {
			if(fabs((float)*(dA + i * ldda)) > MAX){
				MAX = fabs((float)*(dA + i * ldda));
				loc = i;
			}
			// if(*(dA + i * ldda) < MIN){
			// 	MIN = *(dA + i * ldda);
			// }
			if(isinf(*(dA + i * ldda)) || fabs((float)*(dA + i * ldda)) > 1e10){
				counter++;
				if(counter > 1){
					printf("[row check]Multi INFs or Large Number detected in one row. \n");
					return;
				}
			}
		}
		if(counter == 0){
			printf("[row check]Recaculate row chk. No found INF for d1 = INF (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, -1);
			// printf("(C0: %.6f, C1: %.6f, R1: %.6f, R2: %.6f) \n", (float)(*(dA_rowchk)), (float)(*(dA_rowchk + ldda_rowchk)),
			// 													(float)(*(dA_rowchk_r)), (float)(*(dA_rowchk_r + ldda_rowchk_r)));
			T sum = 0.0;
			T sumW = 0.0;
			for(int i = 0; i < col; i++) {
				sum +=	*(dA + i * ldda); 
				sumW += (i+1)*(*(dA + i * ldda));
			}
			*(dA_rowchk) = sum;
			*(dA_rowchk + ldda_rowchk) = sumW;
			return;
		}
		// for(int i = 0; i < col; i++) {
		// 	if (*(dA + i * ldda) == MAX || isinf(*(dA + i * ldda))) {
		// 		loc = i;
		// 		break;
		// 	}
		// }
		printf("[row check]INF detected (idx = (%d, %d), d1 = %.6f, d2 = %.6f, loc = %d) \n", 
								 (blockIdx.x),(threadIdx.x),  (float)d1, (float)d2, loc);
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for (int i = 0; i < col; i++) {
			if (i != loc) {
				sum +=	*(dA + i * ldda); 
			}
		}
		*(dA + loc * ldda) = *dA_rowchk - sum;
		return;
	}
	// abs d1 = NAN
	if(isnan(abs_d1)){
		int64_t counter = 0;
		// abs == nan
		for(int i = 0; i < col; i++) {
			if (isnan(*(dA + i * ldda))) {
				loc = i;
				counter++;
			}
			if (isinf(*(dA + i * ldda))){
				counter++;
			}
			if(counter > 1){
				printf("[row check]Multi INF or NAN detected in one row. \n");
			}
		}
		if(loc == -1){
			printf("[row check]Recaculate row chk. No found NAN for d1 = NAN (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
			// printf("(C0: %.6f, C1: %.6f, R1: %.6f, R2: %.6f) \n", (float)(*(dA_rowchk)), (float)(*(dA_rowchk + ldda_rowchk)),
			// 													(float)(*(dA_rowchk_r)), (float)(*(dA_rowchk_r + ldda_rowchk_r)));
			T sum = 0.0;
			T sumW = 0.0;
			for(int i = 0; i < col; i++) {
				sum +=	*(dA + i * ldda); 
				sumW += (i+1)*(*(dA + i * ldda));
			}
			*(dA_rowchk) = sum;
			*(dA_rowchk + ldda_rowchk) = sumW;
			return;
		}
		printf("[row check]NAN detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for(int i = 0; i < col; i++) {
			if (i != loc) {
				sum +=	*(dA + i * ldda); 
			}
		}
		//correct the error
		*(dA + loc * ldda) = *dA_rowchk - sum;
		return;
	}
}

template<typename T>
__global__ void 
detect_correct_col_Gemm(T * dA, int64_t ldda, T E, int64_t num_col,
								T * dA_colchk, 	int64_t ldda_colchk,
								T * dA_colchk_r, int64_t ldda_colchk_r){
	int col_batchid = blockIdx.x * blockDim.x;
	int colid = col_batchid + threadIdx.x;
	if(colid < num_col){
		//determin the block to process
		dA 			= dA + col_batchid * ldda;
		dA_colchk   = dA_colchk + col_batchid * ldda_colchk;
		dA_colchk_r = dA_colchk_r + col_batchid * ldda_colchk_r;
		
		//determine the specific colum to process
		dA 			= dA + threadIdx.x * ldda;
		dA_colchk   = dA_colchk   + threadIdx.x * ldda_colchk;
		dA_colchk_r = dA_colchk_r + threadIdx.x * ldda_colchk_r;
		
		float d1 = (float)(*dA_colchk)       - (*dA_colchk_r);
		float d2 = (float)(*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
		float abs_d1 = fabs(d1);
		int loc = -1;
		float MAX;

		//error detected
		// abs == inf
		if(isinf(abs_d1)){
			int64_t counter = 0;
			MAX = 0;
			for(int i = 0; i < ldda; i++) {
				if(fabs((float)*(dA + i)) > MAX){
					MAX = fabs((float)*(dA + i));
				}
				if(isinf(*(dA + i)) || fabs((float)*(dA + i)) > 1e10){
					counter++;
					if(counter > 1){
						printf("[col check]Multi INFs or Large Numbers detected in one col. \n");
						return;
					}
				}
			}
			for(int i = 0; i < ldda; i++) {
				if (fabs((float)*(dA + i)) == MAX || isinf(*(dA + i))) {
					loc = i;
					break;
				}
			}
			printf("[col check]INF detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			//the sum of the rest correct number except the error one
			T sum = 0.0;
			for(int i = 0; i < ldda; i++) {
				if (i != loc) {
					sum +=	*(dA + i); 
				}
			}
			//correct the error
			*(dA + loc) = *dA_colchk - sum;
			return;
		}
		// abs == nan
		if(isnan(abs_d1)){
			int64_t counter = 0;
			for(int i = 0; i < ldda; i++) {
				if (isnan(*(dA+i))) {
					loc = i;
					counter++;
				}
				if(isinf(*(dA+i))){
					counter++;
				}
				if(counter > 1){
					printf("[col check]Multi INF or NAN detected in one col. \n");
					return;
				}
			}
			if(loc == -1){
				printf("[col check]No found NAN for d1 = NAN (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
																blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
				printf("(C0: %.6f, C1: %.6f, R1: %.6f, R2: %.6f) \n", (float)(*(dA_colchk)), (float)(*(dA_colchk + 1)),
																(float)(*(dA_colchk_r)), (float)(*(dA_colchk_r + 1)));
				return;
			}
			printf("[col check]NAN detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			//the sum of the rest correct number except the error one
			T sum = 0.0;
			for(int i = 0; i < ldda; i++) {
				if (i != loc) {
					sum +=	*(dA + i); 
				}
			}
			//correct the error
			*(dA + loc) = *dA_colchk - sum;
			return;
		}
		//
		if(fabs(d1) > E && !isinf(abs_d1) && !isnan(abs_d1)) {
			// d2 != INF
			if(!isinf(d2)){
				//locate the error
				loc = round(d2 / d1) - 1;
				printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
				//correction
				// *(dA + loc) += d1;
			}
			else{
				if(isinf(*(dA_colchk + 1))){
					// C1,j == INF
					printf("[col check]Error detected in INPUTS.\n");
					return;
				}
				else{
					// C1,j != INF
					MAX = 0;
					for(int i = 0; i < ldda; i++) {
						if(fabs((float)*(dA+i)) > MAX){
							MAX = fabs((float)*(dA+i));
							loc = i;
						}
					}
					printf("[col check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
					//correction
					// *(dA + loc) += d1;
				}
			}
			// correction
			if(abs_d1 > (float)1e5){
				// printf("d1 > threshold.\n");
				T sum = 0.0;
				for (int i = 0; i < ldda; i++) {
					if (i != loc) {
						sum +=  *(dA + i); 
					}
				}
				*(dA + loc) = *dA_colchk - sum;
			}
			else{
				// printf("d1 =< threshold.\n");
				*(dA + loc) += d1;
			}
			return; 		
		}
	}
}

template <typename T>
__global__ void 
detect_correct_row_Gemm(T * dA, int64_t ldda, T E, int64_t num_row, int64_t num_col,
						    	T * dA_rowchk, 	int64_t ldda_rowchk,
						    	T * dA_rowchk_r, int64_t ldda_rowchk_r){
	int row_batchid = blockIdx.x * blockDim.x;
	int rowid = row_batchid + threadIdx.x;
	if(rowid < num_row){
		//determin the block to process
		dA 			= dA + row_batchid;
		dA_rowchk   = dA_rowchk   + row_batchid;
		dA_rowchk_r = dA_rowchk_r + row_batchid;
			
		//determine the specific row to process
		dA 			= dA + threadIdx.x;
		dA_rowchk   = dA_rowchk   + threadIdx.x;
		dA_rowchk_r = dA_rowchk_r + threadIdx.x;
		
		float d1 = (float)(*dA_rowchk)                 - (*dA_rowchk_r);
		float d2 = (float)(*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
		float abs_d1 = fabs(d1);
		int loc = -1;
		float MAX;

		//error detected
		if(isinf(abs_d1)){
			int64_t counter = 0;
			MAX = 0;
			for(int i = 0; i < num_col; i++){
				if(fabs((float)*(dA + i * ldda)) > MAX){
					MAX = fabs((float)*(dA + i * ldda));
				}
				if(isinf(*(dA + i * ldda)) || fabs((float)*(dA + i * ldda)) > 1e10){
					counter++;
					if(counter > 1){
						printf("[row check]Multi INFs detected in one row. \n");
						return;
					}
				}
			}
			for(int i = 0; i < num_col; i++){
				if(isinf(*(dA + i * ldda)) || fabs((float)*(dA + i * ldda)) == MAX){
					loc = i;
					break;
				}
			}
			printf("[row check]INF detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			T sum = 0.0;
			for (int i = 0; i < num_col; i++) {
				if (i != loc) {
					sum +=  *(dA + i * ldda); 
				}
			}
			//correct the error
			*(dA + loc * ldda) = *dA_rowchk - sum;
			return;
		}
		if(isnan(abs_d1)){
			int64_t counter = 0;
			for(int i = 0; i < num_col; i++){
				if(isnan(*(dA + i * ldda))){
					loc = i;
					counter++;
				}
				if(isinf(*(dA + i * ldda))){
					counter++;
				}
				if(counter > 1){
					printf("[row check]Multi INF or NAN detected in one row. \n");
					return;
				}
			}
			if(loc == -1){
				printf("[row check]No found NAN for d1 = NAN (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
																blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
				printf("(C0: %.6f, C1: %.6f, R1: %.6f, R2: %.6f) \n", (float)(*(dA_rowchk)), (float)(*(dA_rowchk + ldda_rowchk)),
																(float)(*(dA_rowchk_r)), (float)(*(dA_rowchk_r + ldda_rowchk_r)));
				return;
			}
			printf("[col check]NAN detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			T sum = 0.0;
			for (int i = 0; i < num_col; i++) {
				if (i != loc) {
					sum +=  *(dA + i * ldda); 
				}
			}
			//correct the error
			*(dA + loc * ldda) = *dA_rowchk - sum;
			return;
		}
		if(abs_d1 > E && !isinf(abs_d1) && !isnan(abs_d1)) {
			if(!isinf(d2)){
				loc = round(d2 / d1) - 1;
				printf("[row check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
				// *(dA + loc * ldda) += d1;
			}
			else{
				if(isinf(*(dA_rowchk + ldda_rowchk))){
					printf("[row check]Error detected in INPUTS.\n");
					return;
				}
				else{
					MAX = 0;
					for(int i = 0; i < num_col; i++){
						if(fabs((float)*(dA + i * ldda)) > MAX){
							MAX = fabs((float)*(dA + i * ldda));
							loc = i;
						}
					}
					printf("[row check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
					//correction
					// *(dA + loc * ldda) += d1;
				}
			}
			if(abs_d1 > (float)1e5){
				// printf("d1 > threshold.\n");
				T sum = 0.0;
				for (int i = 0; i < num_col; i++) {
					if (i != loc) {
						sum +=  *(dA + i * ldda); 
					}
				}
				//correct the error
				*(dA + loc * ldda) = *dA_rowchk - sum;
			}
			else{
				// printf("d1 =< threshold.\n");
				*(dA + loc * ldda) += d1;
			} 
			return;	
		}
	}
}

template<typename T>
__global__ void addVector(T *dA_chk, T *biasMatrix, int row, int col) {
   
   	// extern __shared__ float dA_ChkSM[];
	SharedMemory<T> smem;
 	T* bias_ChkSM = smem.getPointer();

   	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = r * col + c;

	if (r < row && c < col) {
		bias_ChkSM[threadIdx.y * blockDim.x + threadIdx.x] = biasMatrix[idx];
	} else {
		bias_ChkSM[threadIdx.y * blockDim.x + threadIdx.x] = 0;
	}
	__syncthreads();

	if (r < row && c < col) {
		dA_chk[idx] += bias_ChkSM[threadIdx.y * blockDim.x + threadIdx.x];
	}
}

template<typename T>
__global__ void getBiasMatrix(T *biasVector, T *biasMatrix, int64_t row){
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	for(int r = 0; r < row; r++){
		biasMatrix[colIdx * row + r] = biasVector[r];
	}
}

template<typename T>
__global__ void MatrixSplit(T *inpMatrix, T *outMatrix, int64_t row, int64_t col, int64_t ld, int64_t R_Offset){
	int batchId = threadIdx.y + R_Offset * threadIdx.x;
	int startRow = threadIdx.y * row;
	int startCol = threadIdx.x * col;
	int stride = row * col;

	// printf("%d, %d, %d, %d\n", batchId, threadIdx.x, threadIdx.y, R_Offset);
	
	for(int c = 0; c < col; c++){
		for(int r = 0; r < row; r++){
			int inpIdx = (startRow+startCol*ld) + (r + c * ld);
			int outIdx = batchId * stride + r + c * row;
			// printf("%d\n", outIdx);
			outMatrix[outIdx] = inpMatrix[inpIdx];
		}
	}
}

template<typename T>
__global__ void MatrixTranspose(T *inpMatrix, T *outMatrix, int64_t iRow, int64_t iCol){
	// int batchId = threadIdx.y + iRow * threadIdx.x;
	int stride = iRow * iCol;
	int idx = threadIdx.x * stride;
	
	// printf("%d, %d, %d, %d\n", batchId, threadIdx.x, threadIdx.y, R_Offset);
	
	for(int c = 0; c < iCol; c++){
		for(int r = 0; r < iRow; r++){
			int inpIdx = idx + (r + c * iRow);
			int outIdx = idx + (c + r * iCol);
			outMatrix[outIdx] = inpMatrix[inpIdx];
		}
	}
}

template<typename T>
__global__ void ChkSumScale(T *chksum, int64_t row, int64_t scaleUnit, int64_t stride, int nb){
	int tid = threadIdx.x;
	int idx = threadIdx.x * stride;

	for(int r = 0; r < row; r++){
		int i1 = idx + r;
		int i2 = i1 + row;
		chksum[i2] += chksum[i1] * scaleUnit * (tid / nb);
		// if((idx % nb) == 1){
		// 	printf("%f, %f\n", chksum[i1], chksum[i2]);
		// }
	}
}

template<typename T>
__global__ void MatrixMerge(T *inpMatrix, T *outMatrix, int64_t iRow, int64_t iCol, int64_t oRow, int64_t oCol, int64_t R_Offset){
	int batchId = threadIdx.y + R_Offset * threadIdx.x;
	int startRow = threadIdx.y * iRow;
	int startCol = threadIdx.x * iCol;
	int stride = iCol * iRow;

	// printf("%d, %d, %d, %d\n", batchId, threadIdx.x, threadIdx.y, startRow, startCol);

	for(int c = 0; c < iCol; c++){
		for(int r = 0; r < iRow; r++){
			int outIdx = (startRow + startCol * oRow) + (r + c * oRow);
			int inpIdx = batchId * stride + r + c * iRow;
			// printf("%d\n", outIdx);
			outMatrix[outIdx] = inpMatrix[inpIdx];
		}
	}
}

template <typename T>
// __global__ void bitflip(T *dA, int64_t row, int64_t col, int64_t lda, int64_t batch){
__global__ void bitflip(T *dA, int64_t idx){
	// int stride = row * col;
	// int idx = batch * stride + row + col * lda;
	
	T value = NAN;
	// T value = (T)1e10;
	// T value = INFINITY;
	*(dA + idx) = value;

	// int64_t flipBit = 0;
	// float orgValue = (float)*(dA + idx);
	// if(fabs(orgValue) >= 2){
	// 	flipBit = 29;
	// }
	// else{
	// 	flipBit = 30;
	// }
	// uint32_t* intValue = reinterpret_cast<uint32_t*>(&orgValue);
    // *intValue ^= (1u << flipBit);
	// *(dA + idx) = (T) *reinterpret_cast<float*>(intValue);
	// printf("%.6f\n", (float)*(dA + idx));
}

template <typename T>
__global__ void assignChk(T *input, T *output, int64_t row, int64_t col, int64_t num_batch, int64_t num_head, int64_t N){
	int stride = row * col;
	int batchId = threadIdx.x / num_head;
	int inpIdx = N * stride * num_head + batchId * stride * num_head * 3 + (threadIdx.x % num_head)* stride;
	int outIdx = threadIdx.x * stride;

	for(int c = 0; c < col; c++){
		for(int r = 0; r < row; r++){
			output[outIdx + r + c * row] = input[inpIdx + r + c * row];
		}
	}
}

template <typename T>
__global__ void GemmMatrxiChkMerge(T *A_copy, T *A, T *chk, 
										int64_t Arow, int64_t Acol,
										int64_t Chkrow, int64_t Chkcol){
	// for A -> copy A
	if(threadIdx.x == 0){
		// printf("%d\n", threadIdx.x);
		int64_t inpIdx = blockIdx.x * (Arow*Acol);
		int64_t outIdx = blockIdx.x * ((Acol+2)*Arow);

		for(int c = 0; c < Acol; c++){
			for(int r = 0; r < Arow; r++){
				A_copy[outIdx + (r+c*Arow)] = A[inpIdx + (r+c*Arow)];
			}
		}
	}
	// for chk -> copy A
	else{
		// printf("%d\n", threadIdx.x);
		int64_t inpIdx = blockIdx.x * (Chkcol*Chkrow);
		int64_t outIdx = (Acol*Arow) + (blockIdx.x) * ((Acol+2)*Arow);
		
		for(int c = 0; c < Chkcol; c++){
			for(int r = 0; r < Chkrow; r++){
				A_copy[outIdx + (r+c*Chkrow)] = chk[inpIdx + (r+c*Chkrow)];
			}
		}
	}
	__syncthreads();
}

template <typename T>
__global__ void GemmResCopyBack(T *res, T *inp, 
								int64_t res_ld, int64_t inp_ld, 
								int64_t row, int64_t col, bool COL_FT, bool ROW_FT){
	int64_t inpR = row * threadIdx.y;
	int64_t inpC = col * threadIdx.x;
	if(COL_FT){
		inpR = (row + 2) * threadIdx.y;
	}
	if(ROW_FT){
		inpC = (col + 2) * threadIdx.x;
	}
	int64_t resR = row * threadIdx.y;
	int64_t resC = col * threadIdx.x;

	for(int c = 0; c < col; c++){
		for(int r = 0; r < row; r++){
			int64_t i = (inpR + inpC*inp_ld) + (r + c*inp_ld);
			int64_t o = (resR + resC*res_ld) + (r + c*res_ld);
			res[o] = inp[i];
		}
	}
}

template <typename T>
__global__ void GemmChkCopyBack(T *out, T *inp, int64_t inp_ld, 
								int64_t Orow, int64_t Ocol,
								int64_t Irow, int64_t Icol, 
								int64_t R_Offset, bool ifColChk, 
								bool COL_FT, bool ROW_FT){
	int64_t inpR = Irow * threadIdx.y;
	int64_t inpC = Icol * threadIdx.x;
	if(COL_FT){
		inpR = (Irow + 2) * threadIdx.y;
	}
	if(ROW_FT){
		inpC = (Icol + 2) * threadIdx.x;
	}
	if(ifColChk){
		inpR += Irow;
	}
	else{
		inpC += Icol;
	}

	int64_t batchId = threadIdx.y + R_Offset * threadIdx.x;
	int64_t stride = Ocol * Orow;

	for(int c = 0; c < Ocol; c++){
		for(int r = 0; r < Orow; r++){
			int inpIdx = (inpR + inpC * inp_ld) + (r + c * inp_ld);
			int outIdx = (batchId * stride) + (r + c * Orow);
			// printf("%d\n", outIdx);
			out[outIdx] = inp[inpIdx];
		}
	}
}

template <typename T>
__global__ void GemmMatrxiChkMerge_v2(T *outMatrix, int64_t ld_inp, int64_t N, int64_t num_head,
										T *A, int64_t m, int64_t n, 
										T *chk){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(inpR < ld_inp && inpC < N){
		int64_t aR = m;
		int64_t aC = n + 2;

		int64_t batchR = inpR / aR;
		int64_t batchC = inpC / aC;
		int64_t r = inpR % aR;
		int64_t c = inpC % aC;
		// copy to A
		if(r < m && c < n){
			int64_t idx = batchC * (n*m) + (r + c * m);
			outMatrix[inpR + inpC * ld_inp] = A[idx];
		}
		// copy to row check
		else if((c >= n && c < aC) && (r < m)){
			c -= n;
			int64_t idx = batchC * (2*m) + (r + c * m);
			outMatrix[inpR + inpC * ld_inp] = chk[idx];
		}
	}
}

template <typename T>
__global__ void GemmCopyBack_v2(T *inpMatrix, int64_t ld_inp, int64_t N, int64_t num_head,
								T *A, int64_t m, int64_t n, 
								T *col_chk, bool COL_FT,
								T *row_chk, bool ROW_FT){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(inpR < ld_inp && inpC < N){
		int64_t aR = m;
		int64_t aC = n;
		if(COL_FT){
			aR += 2;
		}
		if(ROW_FT){
			aC += 2;
		}
		int64_t batchR = inpR / aR;
		int64_t batchC = inpC / aC;
		int64_t r = inpR % aR;
		int64_t c = inpC % aC;
		// copy to A
		if(r < m && c < n){
			int64_t idx = (m*num_head) * n * batchC + c * (m*num_head) + batchR * m + r;
			A[idx] = inpMatrix[inpR + inpC * ld_inp];
		}
		// copy to col check
		else if((r >= m && r < aR) && (c < n)){
			r -= m;
			int64_t idx = (batchR + batchC * num_head) * (2*n) + (r + c * 2);
			col_chk[idx] = inpMatrix[inpR + inpC * ld_inp];
		}
		// copy to row check
		else if((c >= n && c < aC) && (r < m)){
			c -= n;
			int64_t idx = (batchR + batchC * num_head) * (2*m) + (r + c * m);
			row_chk[idx] = inpMatrix[inpR + inpC * ld_inp];
		}
	}
}

template <typename T>
__global__ void BGemmCopyBack_v2(T *inpMatrix, int64_t ld_inp, int64_t N,
								 T *A, int64_t m, int64_t n,
								 T* col_chk, T *row_chk){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;

	if(inpR < ld_inp && inpC < N){
		int64_t aR = m + 2;
		int64_t aC = n + 2;

		// int64_t batchR = inpR / aR;
		int64_t batchC = inpC / aC;
		int64_t r = inpR % aR;
		int64_t c = inpC % aC;
		// copy back to A
		if(r < m && c < n){
			int64_t idx = (m*1) * n * batchC + c * (m*1) + 0 * m + r;
			A[idx] = inpMatrix[inpR + inpC * ld_inp];
		}
		// copy back to col chk
		else if((r >= m && r < aR) && (c < n)){
			r -= m;
			int64_t idx = (0 + batchC * 1) * (2*n) + (r + c * 2);
			col_chk[idx] = inpMatrix[inpR + inpC * ld_inp];
		}
		// copy back to row chk
		else if((c >= n && c < aC) && (r < m)){
			c -= n;
			int64_t idx = (0 + batchC * 1) * (2*m) + (r + c * m);
			row_chk[idx] = inpMatrix[inpR + inpC * ld_inp];
		}
	}
}

template <typename T>
__global__ void BGemmMatrxiChkMerge_v2(T *outMatrix, int64_t ld_inp, int64_t N,
										T *A, int64_t m, int64_t n, 
										T *chk, bool ifColChk){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(inpR < ld_inp && inpC < N){
		int64_t aR = m;
		int64_t aC = n;
		if(ifColChk){
			aR += 2;
		}
		else{
			aC += 2;
		}
		// int64_t batchR = inpR / aR;
		int64_t batchC = inpC / aC;
		int64_t r = inpR % aR;
		int64_t c = inpC % aC;
		// copy to A
		if(r < m && c < n){
			int64_t idx = batchC * (n*m) + (r + c * m);
			outMatrix[inpR + inpC * ld_inp] = A[idx];
		}
		// copy to col check
		else if((r >= m && r < aR) && (c < n)){
			r -= m;
			int64_t idx = batchC * (2*n) + (r + c * 2);
			outMatrix[inpR + inpC * ld_inp] = chk[idx];
		}
		// copy to row check
		else if((c >= n && c < aC) && (r < m)){
			c -= n;
			int64_t idx = batchC * (2*m) + (r + c * m);
			outMatrix[inpR + inpC * ld_inp] = chk[idx];
		}
	}
}

template <typename T>
__global__ void BGemmChkMerge_v2(T *inpChk, int64_t R, int64_t C, int64_t num_head,
							  T *outChk, int64_t M, int64_t N){
	int64_t outC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t outR = blockDim.x * blockIdx.x + threadIdx.x;

	if(outR < M && outC < N){
		int64_t batchR = outR / R;
		int64_t batchC = outC / C;
		int64_t r = outR % R;
		int64_t c = outC % C;

		int64_t idx = (batchR + batchC * num_head) * (R * C) + (r + c * R); 
		outChk[outR + outC * M] = inpChk[idx];
	}
}

template <typename T>
__global__ void MatrixTranspose_v2(T *input, int64_t R, int64_t C,
								   T *output , int64_t m, int64_t n){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;

	if(inpR < R && inpC < C){
		int64_t b = inpC / m;
		int64_t c = inpR;
		int64_t r = inpC % m;

		output[b*m*n + r+c*m] = input[inpR + inpC * R];
	} 
}

template <typename T>
__global__ void GemmMatrxiChkMerge_v3(T *A, int64_t A_r, int64_t A_c,
								T *chk, int64_t nb,
								T *outMatrix, int64_t out_r, int64_t out_c){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;

	if(inpR < out_r && inpC < out_c){
		// copy from A
		if(inpR < A_r && inpC < nb*A_c){
			int64_t b = inpC / A_c;
			int64_t c = inpC + b * 2;
			outMatrix[inpR + c * A_r] = A[inpR + inpC * A_r];
		}
		// copy from row chk
		else if(inpR < A_r && inpC >= nb*A_c){
			int64_t tc = inpC - nb*A_c;
			int64_t b = tc / 2;
			// printf("%d\n", b);
			int64_t c = inpC - (nb-b-1)*A_c;
			outMatrix[inpR + c * A_r] = chk[inpR + tc * A_r];
		}
	}
}

template <typename T>
__global__ void GemmMatrxiColChkMerge_v3(T *A, int64_t A_r, int64_t A_c,
								T *chk, int64_t num_head,
								T *outMatrix, int64_t out_r, int64_t out_c){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;

	if(inpR < out_r && inpC < out_c){
		int64_t aR = A_r + 2;
		int64_t batchR = inpR / aR;
		int64_t r = inpR % aR;
		// copy from A
		if(r < A_r && inpC < A_c){
			int64_t idx = (A_r*num_head) * A_c * 0 + inpC * (A_r*num_head) + batchR * A_r + r;
			outMatrix[inpR + inpC * out_r] = A[idx];
		}
		// copy from col chk
		else if((r >= A_r && r < aR) && (inpC < A_c)){
			r -= A_r;
			int64_t idx = (batchR + 0 * num_head) * (2*A_c) + (r + A_c * 2);
			outMatrix[inpR + inpC * out_r] = chk[idx];
		}
	}
}

template <typename T>
__global__ void GemmRowCopyBack_v3(T *inpMatrix, int64_t inp_r, int64_t inp_c, int64_t num_head, int64_t num_batches,
								   T *A, int64_t A_r, int64_t A_c, int64_t chk_r,
								   T *chk){
	int64_t c = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t r = blockDim.x * blockIdx.x + threadIdx.x;

	if(r < inp_r && c < inp_c){
		// copy to A
		if(c < A_c*num_batches){
			int64_t b = c / A_c;
			int64_t ic = c + b * 2;
			A[r+c*A_r] = inpMatrix[r+ic*inp_r];
		}
		// copy to row chk
		else if(c >= A_c*num_batches){
			int64_t tc = c - A_c*num_batches;
			int64_t batchR = r / chk_r;
			int64_t batchC = tc / 2;
			int64_t chkR = r % chk_r;
			int64_t chkC = tc % 2;
			int64_t b = batchR + batchC*num_head;

			chk[chk_r*2*b + chkR+chkC*chk_r] = inpMatrix[r + A_r*(c-(num_batches-batchC-1)*A_c)];
			// chk[chk_r*2*b + chkR+chkC*chk_r] = 1;
		}
	}
}

template <typename T>
__global__ void BGemmMatrxiColChkMerge_v3(T *A, int64_t A_r, int64_t A_c, int64_t nb,
									   T *chk, T *outMatrix, int64_t out_r, int64_t out_c){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	if(inpR < out_r && inpC < out_c){
		// copy from A
		if(inpR < A_r){
			outMatrix[inpR + inpC * out_r] = A[inpR + inpC * A_r];
		}
		// copy from col chk
		else if(inpR >= A_r){
			int64_t tr = inpR - A_r;
			// int64_t b = tc / 2;
			// printf("%d\n", b);
			// int64_t c = inpC - (nb-b-1)*A_c;
			outMatrix[inpR + inpC * out_r] = chk[tr + 2*inpC];
		}
	}
}


template <typename T>
__global__ void BGemmMatrxiRowChkMerge_v3(T *A, int64_t A_r, int64_t A_c, int64_t nb,
									   T *chk, T *outMatrix, int64_t out_r, int64_t out_c){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(inpR < A_r && inpC < out_c){
		// copy from A
		if(inpC < A_c*nb){
			int64_t b = inpC / A_c;
			int64_t c = inpC + b * 2;
			outMatrix[inpR + c * A_r] = A[inpR + inpC * A_r];
		}
		// copy from row chk
		else if(inpC >= A_c*nb){
			int64_t tc = inpC - nb*A_c;
			int64_t b = tc / 2;
			int64_t c = inpC - (nb-b-1)*A_c;
			outMatrix[inpR + c * A_r] = chk[inpR + tc * A_r];
		}
	}
}

template <typename T>
__global__ void BGemmCopyBack_v3(T *inpMatrix, int64_t inp_r, int64_t inp_c, int64_t num_batches,
								 T *A, int64_t m, int64_t n,
								 T *row_chk, T *col_chk,
								 int64_t inp_N, int64_t N){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;

	if(inpR < inp_r && inpC < inp_N){
		// copy to A
		if(inpR < m && inpC < N){
			int64_t b = inpC / n;
			int64_t c = inpC + b * 2;
			A[inpR + inpC * m] = inpMatrix[inpR + c * inp_r];
		}
		// copy to rowchk
		else if(inpR < m && inpC >= N){
			int64_t tc = inpC - num_batches*n;
			int64_t b = tc / 2;
			int64_t c = inpC - (num_batches-b-1)*n;
			row_chk[inpR + tc * m] = inpMatrix[inpR + c * inp_r];
		}
		// copy to colchk
		else if(inpR >= m && inpC < N){
			int64_t b = inpC / n;
			// int64_t b = 0;
			int64_t c = inpC + b * 2;
			int64_t r = inpR - m;
			col_chk[r + inpC * 2] = inpMatrix[inpR + c * inp_r];
		}
	}
}

template <typename T>
__global__ void BGemmChkMerge_v3(T *inpChk, int64_t R, int64_t C, int64_t num_head,
							  T *outChk, int64_t M, int64_t N,
							  int64_t scaleUnit){
	int64_t outC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t outR = blockDim.x * blockIdx.x + threadIdx.x;

	if(outR < M && outC < N){
		int64_t batchR = outR / R;
		int64_t batchC = outC / C;
		int64_t r = outR % R;
		int64_t c = outC % C;

		int64_t idx = (batchR + batchC * num_head) * (R * C) + (r + c * R); 
		if(c == 0){
			outChk[outR + outC * M] = inpChk[idx];
		}
		else{
			int64_t idx1 = (batchR + batchC * num_head) * (R * C) + (r + 0 * R);
			// T bias = inpChk[idx1] * scaleUnit * batchC;
			outChk[outR + outC * M] = inpChk[idx] + inpChk[idx1] * scaleUnit * batchC;
			// outChk[outR + outC * M] = inpChk[idx1] * scaleUnit * batchC;
		}
		
	}
}

template <typename T>
__global__ void MatrixRowReduceSum(T *input, int64_t R, int64_t C, int64_t nb, 
								T *output, int64_t m, int64_t n){
	int64_t outC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t outR = blockDim.x * blockIdx.x + threadIdx.x;

	if(outC < C && outR < R){
		int64_t batchIdx = outC / n;
		int64_t i = 2 * batchIdx;
		int64_t c = outC % n;

		for(int64_t stride = 1; stride < nb; stride*=2){
			if(batchIdx % stride == 0 && i + stride < nb){
				input[(i*m*n) + outR + c * m] += input[(i+stride)*m*n + outR + c * m];
			}
		}
		__syncthreads();

		if(batchIdx == 0){
			output[outR + c * m] = input[outR + c * m];
		}
	}
}

template <typename T>
__global__ void GemmBiasRowCopyBack(T *inpMatrix, int64_t inp_r, int64_t inp_c,
								T *A, int64_t m, int64_t n, 
								// T *col_chk, bool COL_FT,
								T *row_chk, T *bias){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(inpR < inp_r && inpC < inp_c){
		// copy to A
		if(inpR < m && inpC < n){
			A[inpR + inpC * inp_r] = inpMatrix[inpR + inpC * inp_r];
		}
		// copy to row check
		else if(inpC >= n){
			int64_t c = inpC - n;
			int64_t idx = inpR + c * m;
			if(c == 0){
				T b = bias[inpR] * (T)(n-1);
				row_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
			else{
				T b = bias[inpR] * (T)(((n+1)*n/2)-1);
				row_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
		}
	}
}

template <typename T>
__global__ void GemmBiasCopyBack_v2(T *inpMatrix, int64_t inp_r, int64_t inp_c, int64_t num_batches,
										T *A, int64_t m, int64_t n, 
										T *col_chk, bool COL_FT,
										T *row_chk, bool ROW_FT, T *bias){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(inpR < inp_r && inpC < inp_c){
		int64_t aR = m;
		int64_t aC = n;
		if(COL_FT){
			aR += 2;
		}
		if(ROW_FT){
			aC += 2;
		}
		int64_t batchR = inpR / aR;
		int64_t batchC = inpC / aC;
		int64_t r = inpR % aR;
		int64_t c = inpC % aC;
		// copy to A
		if(r < m && c < n){
			int64_t idx = (m*num_batches) * n * batchC + c * (m*num_batches) + batchR * m + r;
			A[idx] = inpMatrix[inpR + inpC * inp_r];
		}
		// copy to col check
		else if((r >= m && r < aR) && (c < n)){
			r -= m;
			// T b = bias[inpR];
			int64_t idx = (batchR + batchC * num_batches) * (2*n) + (r + c * 2);
			col_chk[idx] = inpMatrix[inpR + inpC * inp_r];
		}
		// copy to row check
		else if((c >= n && c < aC) && (r < m)){
			c -= n;
			if(c == 0){
				T b = bias[inpR] * (T)(n-1);
				int64_t idx = (batchR + batchC * num_batches) * (2*m) + (r + c * m);
				row_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
			else{
				T b = bias[inpR] * (T)(((n+1)*n/2)-1);
				int64_t idx = (batchR + batchC * num_batches) * (2*m) + (r + c * m);
				row_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
			// int64_t idx = (batchR + batchC * num_batches) * (2*m) + (r + c * m);
			// T b = bias[inpR] * (T)(((n+1)*n/2)-1);
			// row_chk[idx] = inpMatrix[inpR + inpC * inp_r];
		}
	}
}

template <typename T>
__global__ void GemmBiasCopyBack_QKV(T *inpMatrix, int64_t inp_r, int64_t inp_c, int64_t num_head, int64_t head_size,
										T *A, int64_t m, int64_t n,
										T *q_chk, T *k_chk, T *v_chk, T *bias){
	int64_t inpC = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t inpR = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(inpR < inp_r && inpC < inp_c){
		int64_t aR = m + 2;
		int64_t aC = n + 2;
		int64_t batchR = inpR / aR;
		int64_t batchC = inpC / aC;
		int64_t r = inpR % aR;
		int64_t c = inpC % aC;
		int64_t headIdx = batchR / head_size;
		// copy to A
		if(r < m && c < n){
			int64_t idx = (m*num_head) * n * batchC + c * (m*num_head) + batchR * m + r;
			A[idx] = inpMatrix[inpR + inpC * inp_r];
		}
		// copy to Q row check
		else if((c >= n && c < aC) && (r < m) && (headIdx == 0)){
			// printf("R: %d\n", batchR);
			// printf("C: %d\n", batchC);
			c -= n;
			int64_t idx = (batchR + batchC * head_size) * (2*m) + (r + c * m);
			if(c == 0){
				T b = bias[inpR] * (T)(n-1);
				q_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
			else{
				T b = bias[inpR] * (T)(((n+1)*n/2)-1);
				q_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
		}
		// copy to K row check
		else if((c >= n && c < aC) && (r < m) && headIdx == 1){
			c -= n;
			int64_t idx = ((batchR-head_size) + batchC * head_size) * (2*m) + (r + c * m);
			if(c == 0){
				T b = bias[inpR] * (T)(n-1);
				k_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
			else{
				T b = bias[inpR] * (T)(((n+1)*n/2)-1);
				k_chk[idx] = inpMatrix[inpR + inpC * inp_r] + b;
			}
		}
		// copy to V col check
		else if((r >= m && r < aR) && (c < n) && headIdx == 2){
			r -= m;
			// T b = bias[inpR];
			int64_t idx = ((batchR-2*head_size) + batchC * head_size) * (2*n) + (r + c * 2);
			v_chk[idx] = inpMatrix[inpR + inpC * inp_r];
		}
	}
}


template <typename T>
__global__ void BiasCopy(T *bias_copy, T *bias, int64_t row, int64_t nb){
	int64_t r = blockDim.x * blockIdx.x + threadIdx.x;

	if(r < row){
		int64_t b = r / nb;
		int64_t idx = r + b * 2;
		bias_copy[idx] = bias[r];
	}
}

template <typename T>
__global__ void checkMatrix(T *A, T *B, int64_t ld, int64_t n, int64_t num_batches){
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if(r < ld && c < n){
		if(A[r+c*ld] != B[r+c*ld]){
			printf("(%d, %d), (%f, %f)\n", r, c, A[r+c*ld], B[r+c*ld]);
		}
	}
}

template <typename T>
__global__ void BGemmMatrxiChkMerge(T *A_copy, T *A, T *chk, 
										int64_t Arow, int64_t Acol,
										int64_t Chkrow, int64_t Chkcol){
	// for A -> copy A
	if(threadIdx.x == 0){
		// printf("%d\n", threadIdx.x);
		int64_t inpIdx = blockIdx.x * (Arow*Acol);
		int64_t outIdx = blockIdx.x * (Acol*(Arow+2));

		for(int c = 0; c < Acol; c++){
			for(int r = 0; r < Arow; r++){
				A_copy[outIdx + (r+c*(Arow+2))] = A[inpIdx + (r+c*Arow)];
			}
		}
	}
	// for chk -> copy A
	else{
		// printf("%d\n", threadIdx.x);
		int64_t inpIdx = blockIdx.x * (Chkcol*Chkrow);
		int64_t outIdx = Arow + blockIdx.x * (Acol*(Arow+2));
		
		for(int c = 0; c < Chkcol; c++){
			for(int r = 0; r < Chkrow; r++){
				A_copy[outIdx + (r+c*(Arow+2))] = chk[inpIdx + (r+c*Chkrow)];
			}
		}
	}
	__syncthreads();
}

template <typename T>
__global__ void BGemmResCopyBack(T *res, T *inp, int64_t row, int64_t col){
	int resIdx = (row*col) * threadIdx.x;
	int inpIdx = (row+2)*(col+2) * threadIdx.x;

	for(int c = 0; c < col; c++){
		for(int r = 0; r < row; r++){
			res[resIdx + c*row+r] = inp[inpIdx + c*(row+2)+r];
		}
	}
}

template <typename T>
__global__ void BGemmChkCopyBack(T *out, T *inp, int64_t row, int64_t col, int64_t strideInp, bool ifColChk){
	int inpIdx = 0;
	int64_t chkRow = row;
	int64_t chkCol = col;
	if(ifColChk){
		inpIdx = row + strideInp * threadIdx.x;
		chkRow = 2;
	}
	else{
		inpIdx = col*(row+2) + strideInp * threadIdx.x;
		chkCol = 2;
	}
	int outIdx = chkRow*chkCol * threadIdx.x;

	for(int c = 0; c < chkCol; c++){
		for(int r = 0; r < chkRow; r++){
			out[outIdx + c*chkRow+r] = inp[inpIdx + c*(row+2)+r];
		}
	}
}

// template <typename T>
// __device__ void run_cublasSetMatrix(T *inp, T *out, int64_t Ild,
// 									int64_t Orow, int64_t Ocol, int64_t Old,
// 									int elemSize){
// 	cublasSetMatrix(Orow, Ocol, elemSize, inp, Ild, out, Old);
// }

// template <typename T>
// __global__ void cublasSetMatrixLancher(T *inp, T *out, 
// 										int64_t Irow, int64_t Icol, int64_t Ild,
// 										int64_t Orow, int64_t Ocol, int64_t Old,
// 										int64_t num_batches){
// 	int64_t inpR = Irow * threadIdx.y;
// 	int64_t inpC = Icol * threadIdx.x;
// 	int64_t outR = Orow * threadIdx.y;
// 	int64_t outC = Ocol * threadIdx.x;

// 	inp = inp + (inpR + Irow * inpC);
// 	out = out + (outR + Orow * outC);
// 	int elemSize = (int) sizeof(T);

// 	run_cublasSetMatrix(inp, out, Ild, Orow, Ocol, Old, elemSize);
// }

template <typename T>
__global__ void memSet(T *input, int64_t startOffset){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	input[idx+startOffset] = (float)idx;
}

// template <typename T>
// __global__ void biasRowChk(T *bias, T *rowchk, int64_t row, int col){
// 	int idx = blockDim.x * blockDim.x + threadIdx.x;
// 	int stride = row*col;
// 	if(idx < stride){
// 		rowchk[idx] = bias[idx] * col;
// 	}
// 	else{
// 		int64_t bid = idx-stride;
// 		rowchk[idx] = bias[bid] * bid * col;
// 	}
// }

// template <typename T>
// __global__ void biasColChk(T *colchk, int64_t col){
// 	SharedMemory<T> smem;
//  	T* bias_tmp = smem.getPointer();

// 	int idx = blockDim.x * blockDim.x + threadIdx.x;
// 	if(idx == 0 || idx == 1){
// 		bias_tmp[idx] = colchk[idx];
// 		return;
// 	}
// 	__syncthreads();
	
// 	if(idx < col){
// 		colchk[idx] = bias_tmp[(idx % 2)];
// 	}
// 	__syncthreads();
// }