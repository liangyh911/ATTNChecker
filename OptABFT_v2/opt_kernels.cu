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
						     T * dA_colchk_r, int64_t ldda_colchk_r,	int64_t stride_colchk_r, 
							 int64_t *BatIdx, int64_t *colIdx){
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

	if(isinf(*dA_colchk_r) || isnan(*dA_colchk_r)){
		*BatIdx = (int64_t)blockIdx.x;
		*colIdx = (int64_t)threadIdx.x;
		printf("INF or NAN loc: %d, %d\n", *BatIdx, *colIdx);
		return;
	}
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
						     T * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r,
							 int64_t *BatIdx, int64_t *rowIdx){
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

	if(isinf(*dA_rowchk_r) || isnan(*dA_rowchk_r)){
		*BatIdx = (int64_t)blockIdx.x;
		*rowIdx = (int64_t)threadIdx.x;
		printf("INF or NAN loc: %d, %d\n", *BatIdx, *rowIdx);
		return;
	}
	if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[row check]error detected (idx = (%d, %d), d1 = %.6f, d2 = %.6f, loc = %d) \n", 
								 (blockIdx.x),(threadIdx.x),  (float)d1, (float)d2, loc);
			
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for (int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i * ldda); 
			}
		}
		*(dA + loc * ldda) = *dA_rowchk - sum;
	}
}

template<typename T>
__global__ void 
detect_correct_col_Gemm(T * dA, int64_t ldda, T E, int64_t num_col,
								T * dA_colchk, 	int64_t ldda_colchk,
								T * dA_colchk_r, int64_t ldda_colchk_r,
								int64_t *BatIdx, int64_t *colIdx){
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

		if(isinf(*dA_colchk_r) || isnan(*dA_colchk_r)){
			*BatIdx = (int64_t)blockIdx.x;
			*colIdx = (int64_t)threadIdx.x;
			printf("INF or NAN loc: %d, %d\n", *BatIdx, *colIdx);
			return;
		}
		
		//error detected
		if(fabs(d1) > E) {
			//locate the error
			int loc = round(d2 / d1) - 1;
			printf("[col check]error detected (d1 = %f, d2 = %f, loc = %d) \n",(float)d1, (float)d2, loc);
				
			//the sum of the rest correct number except the error one
			T sum = 0.0;
			for (int i = 0; i < ldda; i++) {
				if (i != loc) {
					sum +=  *(dA + i); 
				}
			}
			//correct the error
			*(dA + loc) = *dA_colchk - sum;
		}
	}
}

template <typename T>
__global__ void 
detect_correct_row_Gemm(T * dA, int64_t ldda, T E, int64_t num_row, int64_t num_col,
						    	T * dA_rowchk, 	int64_t ldda_rowchk,
						    	T * dA_rowchk_r, int64_t ldda_rowchk_r,
							 	int64_t *BatIdx, int64_t *rowIdx){
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

		if(isinf(*dA_rowchk_r) || isnan(*dA_rowchk_r)){
			*BatIdx = (int64_t)blockIdx.x;
			*rowIdx =(int64_t)threadIdx.x;
			printf("INF or NAN loc: %d, %d\n", *BatIdx, *rowIdx);
			return;
		}
		
		//error detected
		if(fabs(d1) > E) {
			//locate the error
			int loc = round(d2 / d1) - 1;
			printf("[row check]error detected (d1 = %f, d2 = %f, loc = %d) \n",(float)d1, (float)d2, loc);
				
			//the sum of the rest correct number except the error one
			T sum = 0.0;
			for (int i = 0; i < num_col; i++) {
				if (i != loc) {
					sum +=  *(dA + i * ldda); 
				}
			}
			//correct the error
			*(dA + loc * ldda) = *dA_rowchk - sum;
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