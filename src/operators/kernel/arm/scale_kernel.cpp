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

#ifdef SCALE_OP

#include "operators/kernel/scale_kernel.h"

namespace paddle_mobile {
namespace operators {

/*
 * @b 特化到具体平台的实现, param 从 op 层传入
 * */

template <>
void ScaleKernel<CPU, float>::Compute(const ScaleParam<CPU> &param) {
  const auto input = param.InputX();
  auto output = param.Out();
  if (input->dims() != output->dims()) {
    output->Resize(input->dims());
  }
  const float scale = param.Scale();
  const float bias = param.Bias();
  if (input->type() == typeid(int64_t)) {
    const int64_t *input_data = input->data<int64_t>();
    int64_t *output_data = output->mutable_data<int64_t>();

    int i = 0;
    for (; i < output->numel(); ++i, ++output_data, ++input_data) {
      *output_data = scale * (*input_data) + bias;
    }
  } else {
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

    int i = 0;
    for (; i < output->numel(); ++i, ++output_data, ++input_data) {
      *output_data = scale * (*input_data) + bias;
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
