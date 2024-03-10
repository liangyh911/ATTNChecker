#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "./abft_checker.h"

void MaxtrixRandom(half *A, int64_t num_batches, int64_t stride, int64_t ld, int64_t row, int64_t col);
void outputChk(half *A, int64_t nb, int64_t ld, int64_t stride, int64_t row, int64_t col);

void abftbgemm(int64_t m, int64_t n, int64_t k, half alpha,
    half *dA, int64_t ldda, int64_t stridea, 
    half *dB, int64_t lddb, int64_t strideb, half beta,
    half *dC, int64_t lddc, int64_t stridec,
    half *dA_colchk, int64_t ldda_colchk, half *dA_rowchk, int64_t ldda_rowchk,
    half *dA_colchk_r, int64_t ldda_colchk_r, half *dA_rowchk_r, int64_t ldda_rowchk_r,
    half *dB_colchk, int64_t lddb_colchk, half *dB_rowchk, int64_t lddb_rowchk,    
    half *dB_colchk_r, int64_t lddb_colchk_r, half *dB_rowchk_r, int64_t lddb_rowchk_r,
    half *dC_colchk, int64_t lddc_colchk, half *dC_rowchk, int64_t lddc_rowchk,
    half *dC_colchk_r, int64_t lddc_colchk_r, half *dC_rowchk_r, int64_t lddc_rowchk_r,
    half *chk_v_a, half *chk_v_b, int64_t ld_chk_v,
    int64_t num_batches,
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER){
    
    std::cout << "Using abftbgemm-at::half function." << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cublasSetStream(handle, stream1);

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;

    float falpha = 1;
    float fbeta = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t, t1, t_Achk, t_Bchk;
    bool DEBUG_GEMM = true;

    if (DEBUG_GEMM) cudaEventRecord(start, stream1);
    if(transA == CUBLAS_OP_N){
        cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, m,
        &alpha, chk_v_a, CUDA_R_16F, ld_chk_v, 0,
        dA, CUDA_R_16F, ldda, stridea, &fbeta,
        dA_colchk, CUDA_R_16F, ldda_colchk, (2*k),
        num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        // std::cout << "  Output dA_colchk: " << std::endl;
        // outputChk(dA_colchk, num_batches, ldda_colchk, (2*k), 2, k);
    }
    else{
        cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, m,
        &falpha, dA, CUDA_R_16F, ldda, stridea,
        chk_v_a, CUDA_R_16F, ld_chk_v, 0, &fbeta,
        dA_rowchk, CUDA_R_16F, ldda_rowchk, (2*k),
        num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        // std::cout << "  Output dA_rowchk: " << std::endl;
        // outputChk(dA_rowchk, num_batches, ldda_rowchk, (2*k), k, 2);
    }
    if (DEBUG_GEMM) {
        cudaEventRecord(stop, stream1);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Achk, start, stop);
        // printf("dA_chk_gemm: %f (%f)(%f)\n", t, (double)num_batches*m*2*k*2/t/1e6, (double)num_batches*(2*k+2*m+k*m)/t/1e6);
    }

    //std::cout << "  Get dB_chk: " << std::endl;
    if (DEBUG_GEMM) cudaEventRecord(start, stream1);
    if (transB == CUBLAS_OP_N){
        cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, n,
        &alpha, dB, CUDA_R_16F, lddb, strideb,
        chk_v_b, CUDA_R_16F, ld_chk_v, 0, &fbeta,
        dB_rowchk, CUDA_R_16F, lddb_rowchk, (2*k),
        num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        // std::cout << " Output dB_rowchk: " << std::endl;
        // outputChk(dB_rowchk, num_batches,lddb_rowchk, (2*k), k, 2);
    }
    else{
        cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, n,
        &alpha, chk_v_b, CUDA_R_16F, ld_chk_v, 0,
        dB, CUDA_R_16F, lddb, strideb, &fbeta,
        dB_colchk, CUDA_R_16F, lddb_colchk, (2*k),
        num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        // std::cout << " Output dB_colchk: " << std::endl;
        // outputMatrixChk(dB_colchk, lddb_colchk, (2*k), num_batches, 2, k);
    }
    if (DEBUG_GEMM) {
        cudaEventRecord(stop, stream1);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Bchk, start, stop);
        // printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*2*n*k*2/t1/1e6, (double)num_batches*(2*k+k*n+2*n)/t1/1e6);
    }

    falpha = alpha;
    fbeta = beta;

    // number of row and col of B stored in memory(no trans operation)
    int64_t mem_row = 0;
    int64_t mem_col = 0;

    // --check before beginning-- //
    std::cout << "-----Check Before Beginning------" << std::endl;
    if (COL_FT && CHECK_BEFORE) {
      // number of row and col of A stored in memory(no trans operation)
      if (transA == CUBLAS_OP_N) {
        mem_row = m;
        mem_col = k;
        if (DEBUG) printf("abftgemm-before-check-A-col\n");
        abft_checker_colchk(handle, transA, transB,
                              dA, ldda, mem_row, mem_col, stridea,
                              dA_colchk,   ldda_colchk,
                              dA_colchk_r, ldda_colchk_r,
                              chk_v_a,       ld_chk_v,
                              DEBUG,
                              stream1,
                              num_batches);
      }
      else if (transA == CUBLAS_OP_T || transA == CUBLAS_OP_C) {
        mem_row = k;
        mem_col = m;
        if (DEBUG) printf("dgemm-before-check-A-row\n");
        abft_checker_rowchk(handle, transA, transB,
                              dA, ldda, mem_row, mem_col, stridea,
                              dA_rowchk,   ldda_rowchk,
                              dA_rowchk_r, ldda_rowchk_r,
                              chk_v_a,       ld_chk_v,
                              DEBUG,
                              stream1,
                              num_batches);
      }
      mem_row = m;
      mem_col = n;
      if (DEBUG) printf("abftgemm-before-check-C-col\n");
      abft_checker_colchk(handle, transA, transB,
                              dC, lddc, mem_row, mem_col, stridec,
                              dC_colchk,   lddc_colchk,
                              dC_colchk_r, lddc_colchk_r,
                              chk_v_a,       ld_chk_v,
                              DEBUG,
                              stream1,
                              num_batches);

    }
    if (ROW_FT && CHECK_BEFORE)	{
      //verify B before use
      if (transB == CUBLAS_OP_N) {
        mem_row = k;
        mem_col = n;
        if (DEBUG) printf("dgemm-before-check-B-row\n");
        abft_checker_rowchk(handle, transA, transB,
                                dB, lddb, mem_row, mem_col, strideb,
                                dB_rowchk,   lddb_rowchk,
                                dB_rowchk_r, lddb_rowchk_r,
                                chk_v_b,       ld_chk_v,
                                DEBUG,
                                stream1,
                                num_batches);

      }
      else if (transB == CUBLAS_OP_T || transB == CUBLAS_OP_C) {
        mem_row = n;
        mem_col = k;
        if (DEBUG) printf("dgemm-before-check-B-col\n");
        abft_checker_colchk(handle, transA, transB,
                                dB, lddb, mem_row, mem_col, strideb,
                                dB_colchk,   lddb_colchk,
                                dB_colchk_r, lddb_colchk_r,
                                chk_v_b,       ld_chk_v,
                                DEBUG,
                                stream1,
                                num_batches);
      }
      mem_row = m;
      mem_col = n;
      if (DEBUG) printf("dgemm-before-check-C-row\n");
      abft_checker_rowchk(handle, transA, transB,
                              dC, lddc, mem_row, mem_col, stridec,
                              dC_rowchk,   lddc_rowchk,
                              dC_rowchk_r, lddc_rowchk_r,
                              chk_v_b,       ld_chk_v,
                              DEBUG,
                              stream1,
                              num_batches);
    }

    std::cout << "-----Begin.------" << std::endl;

    if (DEBUG_GEMM) cudaEventRecord(start, stream1);
    std::cout<<"A*B=C." << std::endl;
    cublasGemmStridedBatchedEx(
        handle, transA, transB, m, n, k,
        &falpha, dA, CUDA_R_16F, ldda, stridea,
        dB, CUDA_R_16F, lddb, strideb, &fbeta,
        dC, CUDA_R_16F, lddc, stridec,
        num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // std::cout << "Output dC: " << std::endl;
    // outputMatrix(dC, lddc, stridec, num_batches, m, n);
    
    if (DEBUG_GEMM) {
      cudaEventRecord(stop, stream1);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t, start, stop);
      printf("  gemm: %f (%f)(%f)\n", t, (double)num_batches*m*n*k*2/t/1e6, (double)num_batches*(m*k+k*n+m*n)/t/1e6);
      printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)num_batches*m*2*k*2/t_Achk/1e6, (double)num_batches*(2*k+2*m+k*m)/t_Achk/1e6);
      printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t_Bchk, t_Bchk/t, (double)num_batches*2*n*k*2/t_Bchk/1e6, (double)num_batches*(2*k+k*n+2*n)/t_Bchk/1e6);
    }

    if (DEBUG_GEMM) cudaEventRecord(start, stream1);
    if(COL_FT){
      //std::cout << "  COL_FT" << std::endl;
      if (transA == CUBLAS_OP_N) {
        std::cout << "dA_colchk * dB = dC_colchk" << std::endl;
        cublasGemmStridedBatchedEx(
            handle, transA, transB, 2, n, k,
            &falpha, dA_colchk, CUDA_R_16F, ldda_colchk, k*2,
            dB, CUDA_R_16F, lddb, strideb, &fbeta,
            dC_colchk, CUDA_R_16F, lddc_colchk, n*2,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      else{
        std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;
        cublasGemmStridedBatchedEx(
            handle, transA, transB, 2, n, k,
            &falpha, dA_rowchk, CUDA_R_16F, ldda_rowchk, k*2,
            dB, CUDA_R_16F, lddb, strideb, &fbeta,
            dC_colchk, CUDA_R_16F, lddc_colchk, n*2,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      // std::cout << "Output dC_colchk: " << std::endl;
      // outputMatrixChk(dC_colchk, ldda_colchk, n*2, num_batches, 2, n);
    }
    if (DEBUG_GEMM) {
        cudaEventRecord(stop, stream1);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t1, start, stop);
        printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*2*n*k*2/t1/1e6, (double)num_batches*(2*k+k*n+2*n)/t1/1e6);
    }

    if (DEBUG_GEMM) cudaEventRecord(start, stream1);
    if (ROW_FT) {
        //std::cout << "  ROW_FT" << std::endl;
        if (transB == CUBLAS_OP_N) {
          std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
          //we can further work on this to support trans A.
          cublasGemmStridedBatchedEx(
            handle, transA, transB, m, 2, k,
            &falpha, dA, CUDA_R_16F, ldda, stridea,
            dB_rowchk, CUDA_R_16F, lddb_rowchk, k*2, &fbeta,
            dC_rowchk, CUDA_R_16F, lddc_rowchk, m*2,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        else{
          std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
          cublasGemmStridedBatchedEx(
            handle, transA, transB, m, 2, k,
            &falpha, dA, CUDA_R_16F, ldda, stridea,
            dB_colchk, CUDA_R_16F, lddb_colchk, k*2, &fbeta,
            dC_rowchk, CUDA_R_16F, lddc_rowchk, m*2,
            num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // std::cout << "Output dC_rowchk: " << std::endl;
        // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
    }
    if (DEBUG_GEMM) {
      cudaEventRecord(stop, stream1);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*m*2*k*2/t1/1e6, (double)num_batches*(m*k+k*2+m*2)/t1/1e6);
    }

    // --- check check-sum of C---//
    std::cout << "------Check check-sum-------" << std::endl;
    if (DEBUG_GEMM) cudaEventRecord(start, stream1);
    if (COL_FT && CHECK_AFTER) {
      mem_row = m;
      mem_col = n;
      if (DEBUG) printf("dgemm-after-check-C-col\n");
      abft_checker_colchk(handle, transA, transB,
                              dC, lddc, mem_row, mem_col, stridec,
                              dC_colchk,   lddc_colchk,
                              dC_colchk_r, lddc_colchk_r,
                              chk_v_a,       ld_chk_v,
                              DEBUG,
                              stream1,
                              num_batches);
    }

    if (DEBUG_GEMM) {
      cudaEventRecord(stop, stream1);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*2*n*m*2/t1/1e6, (double)num_batches*(m*n+2*m+2*n)/t1/1e6);
    }

    if (DEBUG_GEMM) cudaEventRecord(start, stream1);
    if (ROW_FT && CHECK_AFTER) {
      mem_row = m;
      mem_col = n;
      if (DEBUG) printf("dgemm-after-check-C-row\n");
      abft_checker_rowchk(handle, transA, transB,
                              dC, lddc, mem_row, mem_col, stridec,
                              dC_rowchk,   lddc_rowchk,
                              dC_rowchk_r, lddc_rowchk_r,
                              chk_v_b,       ld_chk_v,
                              DEBUG,
                              stream1,
                              num_batches);

    }

    if (DEBUG_GEMM) {
        cudaEventRecord(stop, stream1);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t1, start, stop);
        printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*m*2*n*2/t1/1e6, (double)num_batches*(m*n+2*n+2*m)/t1/1e6);
    }

}


int main(){
    half *A, *B;
    half *dA, *dB, *dC;

    int64_t m = 72;
    int64_t n = 72;
    int64_t k = 64;
    int64_t num_batches = 96;

    size_t size = num_batches * m * k * sizeof(half);
    cudaMalloc((void **)&dA, size);
    A = (half *)malloc(size);
    MaxtrixRandom(A, num_batches, m*k, m, m, k);
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    // printf("dA: \n");
    // outputChk(dA, num_batches, m, m*k, m, k); 

    size = num_batches * k * n * sizeof(half);
    cudaMalloc((void **)&dB, size);
    B = (half *)malloc(size);
    MaxtrixRandom(B, num_batches, k*n, k, k, n);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    // printf("dB: \n");
    // outputChk(dB, num_batches, k, n*k, k, n);
    
    size = num_batches * m * n * sizeof(half);
    cudaMalloc((void **)&dC, size);
    cudaMemset(dC, 0, (num_batches * m * n * sizeof(half)));

    int64_t ldda_colchk = 2;
    int64_t ldda_colchk_r = 2;
    int64_t ldda_rowchk = k;
    int64_t ldda_rowchk_r = k;

    int64_t lddb_rowchk = k;
    int64_t lddb_rowchk_r = k;
    int64_t lddb_colchk = 2;
    int64_t lddb_colchk_r = 2;

    int64_t lddc_colchk = 2;
    int64_t lddc_colchk_r = 2;
    int64_t lddc_rowchk = m;
    int64_t lddc_rowchk_r = m;
    int64_t ld_chk_v = 2;

    half *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
    half *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
    half *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
    half *chk_v_a;
    half *chk_v_b;

    size = (2*num_batches) * k * sizeof(half);
    cudaMalloc((void**)&dA_colchk, size);
    cudaMemset(dA_colchk, 0, size);
    cudaMalloc((void**)&dA_colchk_r, size);
    cudaMemset(dA_colchk_r, 0, size);

    cudaMalloc((void**)&dA_rowchk, size);
    cudaMemset(dA_rowchk, 0, size);
    cudaMalloc((void**)&dA_rowchk_r, size);
    cudaMemset(dA_rowchk_r, 0, size);
    //std::cout << "  finish dA." << std::endl;
    
    cudaMalloc((void**)&dB_colchk, size);
    cudaMemset(dB_colchk, 0, size);
    cudaMalloc((void**)&dB_colchk_r, size);
    cudaMemset(dB_colchk_r, 0, size);
    
    cudaMalloc((void**)&dB_rowchk, size);
    cudaMemset(dB_rowchk, 0, size);
    cudaMalloc((void**)&dB_rowchk_r, size);
    cudaMemset(dB_rowchk_r, 0, size);
    //std::cout << "  finish dB." << std::endl;

    size = (2*num_batches) * n * sizeof(half);
    cudaMalloc((void**)&dC_colchk, size);
    cudaMemset(dC_colchk, 0, size);
    cudaMalloc((void**)&dC_colchk_r, size);
    cudaMemset(dC_colchk_r, 0, size);
    
    size = (2*num_batches) * m * sizeof(half);
    cudaMalloc((void**)&dC_rowchk, size);
    cudaMemset(dC_rowchk, 0, size);
    cudaMalloc((void**)&dC_rowchk_r, size);
    cudaMemset(dC_rowchk_r, 0, size);

    int64_t len = m;
    size = 2 * len * sizeof(half);
    cudaMalloc((void**)&chk_v_a, size);
    // std::cout << "  assign values to chk_v_a." << std::endl;
    half *h_matrix;
    h_matrix = (half *)malloc(size);
    int idx = 0;
    for(int i = 0; i < len; i++){
        idx = i*ld_chk_v;
        h_matrix[idx] = half(1);
        h_matrix[idx+1] = half(i+1);
    }
    cudaMemcpy(chk_v_a, h_matrix, size, cudaMemcpyHostToDevice);
    // std::cout << "chk_v_a: " << std::endl;
    // outputChk(chk_v_a, 1, ld_chk_v, 0, 2, m);
    free(h_matrix);

    len = n;
    size = 2 * len * sizeof(half);
    cudaMalloc((void**)&chk_v_b, size);
    // std::cout << "  assign values to chk_v_b." << std::endl;
    h_matrix = (half *)malloc(size);
    idx = 0;
    for(int i = 0; i < len; i++){
        idx = i*ld_chk_v;
        h_matrix[idx] = half(1);
        h_matrix[idx+1] = half(i+1);
    }
    cudaMemcpy(chk_v_b, h_matrix, size, cudaMemcpyHostToDevice);
    // std::cout << "chk_v_b: " << std::endl;
    // outputChk(chk_v_a, 1, ld_chk_v, 0, 2, len);
    free(h_matrix);
    //std::cout << "  finish chk_v." << std::endl;

    bool COL_FT = true;
    bool ROW_FT = true;
    bool DEBUG = true;
    bool CHECK_BEFORE = true;
    bool CHECK_AFTER = true;

    float alpha = 1;
    float beta = 0;
    int64_t stridea = m*k;
    int64_t strideb = n*k;
    int64_t stridec = m*n;
    int64_t ldda = m;
    int64_t lddb = k;
    int64_t lddc = m;


    abftbgemm(m, n, k,
        alpha, dA, ldda, stridea,
        dB, lddb, strideb, beta,
        dC, lddc, stridec,
        dA_colchk, ldda_colchk,
        dA_rowchk, ldda_rowchk,
        dA_colchk_r, ldda_colchk_r,
        dA_rowchk_r, ldda_rowchk_r,
        dB_colchk, lddb_colchk,
        dB_rowchk, lddb_rowchk,
        dB_colchk_r, lddb_colchk_r,
        dB_rowchk_r, lddb_rowchk_r,
        dC_colchk, lddc_colchk,
        dC_rowchk, lddc_rowchk,
        dC_colchk_r, lddc_colchk_r,
        dC_rowchk_r, lddc_rowchk_r,
        chk_v_a, chk_v_b, ld_chk_v,
        num_batches,
        COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER);

    cudaFree(dA_colchk);
    cudaFree(dA_rowchk);
    cudaFree(dA_colchk_r);
    cudaFree(dA_rowchk_r);
    cudaFree(dB_colchk);
    cudaFree(dB_rowchk);
    cudaFree(dB_colchk_r);
    cudaFree(dB_rowchk_r);
    cudaFree(dC_colchk);
    cudaFree(dC_rowchk);
    cudaFree(dC_colchk_r);
    cudaFree(dC_rowchk_r);
    cudaFree(chk_v_a);
    cudaFree(chk_v_b);

    return 0;
}


void MaxtrixRandom(half *A, int64_t num_batches, int64_t stride, int64_t ld, int64_t row, int64_t col){
  for(int num = 0; num < num_batches; num++){
    for (int r = 0; r < row; r++){
      for (int c = 0; c < col; c++){
        A[num*stride + c*ld + r] = __float2half ((float)rand() / RAND_MAX);
        // (half)((float)(rand()) / (float)(rand()));
        // A[num*stride + c*ld + r] = 1;
      }
    }
  }
}

void outputChk(half *A, int64_t nb, int64_t ld, int64_t stride, int64_t row, int64_t col){
  size_t size = nb * (row * col) * sizeof(float);
  half *tensor;
  tensor = (half *)malloc(size);
  cudaMemcpy(tensor, A, size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < nb; i++){
    printf("[ \n");
    for(int r = 0; r < row; r++){
      for(int c = 0; c < col; c++){
        printf("%.6f", __half2float((tensor[i*stride + c*ld + r])));
        printf(", ");
      }
      printf("\n");
    }
    printf("]\n");
  }
  free(tensor);
}