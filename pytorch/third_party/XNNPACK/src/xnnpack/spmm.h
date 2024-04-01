// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
    size_t mc,                                            \
    size_t nc,                                            \
    const float* input,                                   \
    const float* weights,                                 \
    const int32_t* widx_dmap,                             \
    const uint32_t* nidx_nnzmap,                          \
    float* output,                                        \
    size_t output_stride,                                 \
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_1x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_1x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_2x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_2x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neon)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neon_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neon_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neonfma_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neonfma_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__sse)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x2__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x4__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neon)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neon_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neon_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neonfma_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neonfma_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__sse)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x2__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x2__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x4__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x4__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_12x1__neon)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_12x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_12x2__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_12x4__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neon)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neon_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neon_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neonfma_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neonfma_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__sse)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x2__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x4__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__neon)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__neon_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__neon_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__neonfma_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__neonfma_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__sse)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x4)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x2__aarch64_neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_32x4__aarch64_neonfma)

#define DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
    size_t mc,                                            \
    size_t nc,                                            \
    const void* input,                                    \
    const void* weights,                                  \
    const int32_t* widx_dmap,                             \
    const uint32_t* nidx_nnzmap,                          \
    void* output,                                         \
    size_t output_stride,                                 \
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2)

#ifdef __cplusplus
}  // extern "C"
#endif
