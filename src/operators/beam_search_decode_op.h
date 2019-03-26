/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef BEAM_SEARCH_DECODE_OP

#pragma once

#include <string>
#include "framework/operator.h"
#include "operators/kernel/beam_search_decode_kernel.h"

namespace paddle_mobile {
namespace operators {
/*template <typename T>
class BeamSearchDecodeOp : public framework::OperatorWithKernels<T,
BeamSearchDecodeParam> { public: BeamSearchDecodeOp(const std::string &type,
const VariableNameMap &inputs, const VariableNameMap &outputs, const
framework::AttributeMap &attrs, framework::Scope *scope) :
framework::OperatorWithKernels<T, BeamSearchDecodeParam>(type, inputs, outputs,
                                                     attrs, scope) {
    framework::OperatorWithKernels<T,
BeamSearchDecodeParam>::kernels.insert(TYPE_GPU, kernelGpu_);
    framework::OperatorWithKernels<T,
BeamSearchDecodeParam>::kernels.insert(TYPE_CPU, kernelCpu_);
  }
  void InferShape() const override;
  BeamSearchDecodeKernelCpu<T> kernelCpu_;
  BeamSearchDecodeKernelGpu<T> kernelGpu_;
};*/

DECLARE_OPERATOR(BeamSearchDecode);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // BEAM_SEARCH_DECODE_OP
