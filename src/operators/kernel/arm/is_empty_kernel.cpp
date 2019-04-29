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

#ifdef INCREMENT_OP

#include "operators/kernel/is_empty_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool IsEmptyKernelCpu<float>::Init(IsEmptyParam *param) {
  return true;
}

template <>
void IsEmptyKernelCpu<float>::Compute(const IsEmptyParam &param) {
  const framework::Tensor *input = param.InputX()->LodTensor();
  framework::Tensor *out = param.Out()->LodTensor();
  out->mutable_data<bool>()[0] = input->numel() == 0;
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
