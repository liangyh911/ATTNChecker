#################################################################################################
#
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Utilities for emitting Trmm kernels
"""

import enum
import os.path
import shutil
import functools
import operator

from cutlass_library.library import *


###################################################################################################
#
# Data structure modeling a TRMM operation
#
###################################################################################################

#
class TrmmOperation:
  #
  def __init__(self, trmm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8):

    self.operation_kind = OperationKind.Trmm
    self.arch = arch
    self.tile_description = tile_description
    self.trmm_kind = trmm_kind
    self.A = A
    self.B = B
    self.C = C
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor
    self.swizzling_functor = swizzling_functor

  #
  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators
    return False

  #
  def is_planar_complex(self):
#   return self.trmm_kind in (TrmmKind.PlanarComplex, TrmmKind.PlanarComplexArray)
    return False

  #
  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator

    if self.is_complex():
      return get_complex_from_real(accum)

    return accum

  #
  def short_math_name(self):
    if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
      return "g%s" % ShortDataTypeNames[self.accumulator_type()]
    return ShortDataTypeNames[self.accumulator_type()]


  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    inst_shape = ''
    inst_operation = ''
    intermediate_type = ''

    math_operations_map = {
      MathOperation.xor_popc: 'xor',
      MathOperation.and_popc: 'and'
    }

    if self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp or \
      self.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp:

      math_op = self.tile_description.math_instruction.math_operation
      math_op_string = math_operations_map[math_op] if math_op in math_operations_map.keys() else ''

      inst_shape = "%d%d%d" % tuple(self.tile_description.math_instruction.instruction_shape)
      inst_shape += math_op_string

      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator:
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

    return "%s%s%s%s" % (self.short_math_name(), inst_shape, intermediate_type, TrmmKindNames[self.trmm_kind])

  #
  def extended_name(self):
    ''' Append data types if they differ from compute type. '''
    if self.is_complex():
      extended_name = "${core_name}"
    else:
      if self.C.element != self.tile_description.math_instruction.element_accumulator and \
        self.A.element != self.tile_description.math_instruction.element_accumulator:
        extended_name = "${element_c}_${core_name}_${element_a}"
      elif self.C.element == self.tile_description.math_instruction.element_accumulator and  \
        self.A.element != self.tile_description.math_instruction.element_accumulator:
        extended_name = "${core_name}_${element_a}"
      else:
        extended_name = "${core_name}"

    extended_name = SubstituteTemplate(extended_name, {
      'element_a': DataTypeNames[self.A.element],
      'element_c': DataTypeNames[self.C.element],
      'core_name': self.core_name()
      })

    return extended_name

  #
  def layout_name(self):
    if self.is_complex() or self.is_planar_complex():
      return "%s%s" % (
        ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
        ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)]
      )
    return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

  #
  def side_mode_name(self):
    return "%s" % (ShortSideModeNames[self.A.side_mode])

  #
  def fill_mode_name(self):
    return "%s" % (ShortFillModeNames[self.A.fill_mode])

  #
  def diag_type_name(self):
    return "%s" % (ShortDiagTypeNames[self.A.diag_type])

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    threadblock = self.tile_description.procedural_name()

    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

    alignment = max([self.C.alignment])

    return SubstituteTemplate(
      "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_${side_mode}_${fill_mode}_${diag_type}_align${alignment}",
      {
        'opcode_class': opcode_class_name,
        'extended_name': self.extended_name(),
        'threadblock': threadblock,
        'layout': self.layout_name(),
        'side_mode': self.side_mode_name(),
        'fill_mode': self.fill_mode_name(),
        'diag_type': self.diag_type_name(),
        'alignment': "%d" % self.C.alignment,
      }
    )

  #
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.procedural_name()

###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitTrmmUniversalInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self):
    self.trmm_template = """
// Trmm operator ${operation_name}
using Operation_${operation_name} =
  typename cutlass::gemm::device::Trmm<
    ${element_a}, ${layout_a},
    ${side_mode}, ${fill_mode}, ${diag_type},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue},
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    ${split_k_serial},
    ${math_operation}
>;
"""
    self.trmm_complex_template = """
// Trmm operator ${operation_name}
using Operation_${operation_name} =
  typename cutlass::gemm::device::Trmm<
    ${element_a}, ${layout_a},
    ${side_mode}, ${fill_mode}, ${diag_type},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue},
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    ${split_k_serial},
    ${math_operation},
    ${transform_a}
>;
"""

  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'side_mode' : SideModeTag[operation.A.side_mode],
      'fill_mode': FillModeTag[operation.A.fill_mode],
      'diag_type' : DiagTypeTag[operation.A.diag_type],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(1),  # TRMM A's alignment is always 1 for no padding to work until we make zfill work with variable bytes
      'align_b': str(operation.B.alignment),
      'split_k_serial': 'false',
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'transform_a': ComplexTransformTag[operation.A.complex_transform]
    }

    trmm_template = self.trmm_complex_template if operation.is_complex() else self.trmm_template

    return SubstituteTemplate(trmm_template, values)

###################################################################################################


###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################

class EmitTrmmConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.cu" % configuration_name).replace('\\', '/')

    self.instance_emitter = {
      TrmmKind.Universal: EmitTrmmUniversalInstance,
    }

    self.trmm_kind_wrappers = {
      TrmmKind.Universal: 'TrmmOperation',
    }

    self.instance_template = {
      TrmmKind.Universal: """
${compile_guard_start}
  manifest.append(new ${trmm_kind}<
    Operation_${operation_name}
  >("${operation_name}"));
${compile_guard_end}
"""
    }

    self.header_template = """
/*
  Generated by trmm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "trmm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    self.initialize_function_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_${configuration_name}(Manifest &manifest) {

"""
    self.epilogue_template = """

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  def __enter__(self):
    self.configuration_file = open(self.configuration_path, "w")
    self.configuration_file.write(self.header_template)

    self.instance_definitions = []
    self.instance_wrappers = []

    self.operations = []
    return self

  def emit(self, operation):
    emitter = self.instance_emitter[operation.trmm_kind]()

    self.operations.append(operation)

    self.instance_definitions.append(emitter.emit(operation))

    self.instance_wrappers.append(SubstituteTemplate(self.instance_template[operation.trmm_kind], {
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'trmm_kind': self.trmm_kind_wrappers[operation.trmm_kind],
      'compile_guard_start': SubstituteTemplate(self.wmma_guard_start, {'sm_number': str(operation.arch)}) \
        if operation.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp else "",
      'compile_guard_end': "#endif" \
        if operation.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp else ""
      }))

  def __exit__(self, exception_type, exception_value, traceback):

    # Write instance definitions in top-level namespace
    for instance_definition in self.instance_definitions:
      self.configuration_file.write(instance_definition)

    # Add wrapper objects within initialize() function
    self.configuration_file.write(SubstituteTemplate(self.initialize_function_template, {
      'configuration_name': self.configuration_name
      }))

    for instance_wrapper in self.instance_wrappers:
      self.configuration_file.write(instance_wrapper)

    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()

###################################################################################################
