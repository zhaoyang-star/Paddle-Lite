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

#pragma once

#include <string>
#include "framework/tensor.h"

namespace paddle_mobile {
namespace operators {
namespace math {

void SetConstant(framework::Tensor *tensor, float value);

template <typename Itype, typename Otype>
void MatMul(const framework::Tensor &matrix_a, bool trans_a,
            const framework::Tensor &matrix_b, bool trans_b, float alpha,
            framework::Tensor *matrix_out, float beta, bool relu = false,
            Otype *bias = nullptr);

template <typename Itype, typename Otype>
void MatMul(const framework::Tensor &matrix_a, bool trans_a,
            const framework::Tensor &matrix_b, bool trans_b, float alpha,
            framework::Tensor *matrix_out, float beta, bool relu, Otype *bias,
            bool addOnRow);

void MatMulWithBn(const framework::Tensor &matrix_a, bool trans_a,
                  const framework::Tensor &matrix_b, bool trans_b, float alpha,
                  framework::Tensor *matrix_out, float beta, bool relu,
                  framework::Tensor *new_scale, framework::Tensor *new_bias,
                  int group, float *bias = nullptr);

void MatMulWithPRelu(const framework::Tensor &matrix_a, bool trans_a,
                     const framework::Tensor &matrix_b, bool trans_b,
                     framework::Tensor *matrix_out, float *p, std::string mode,
                     float *bias, float *bias1);

template <typename T>
struct ClearTensor {
  void operator()(framework::Tensor *tensor) {
    auto size = tensor->numel();
    auto *tensor_data = tensor->data<T>();
    memset((void *)tensor_data, 0, sizeof(T) * size);  // NOLINT
  }
};

template <typename T>
struct RowwiseAdd {
  void operator()(const framework::Tensor &input,
                  const framework::Tensor &vector, framework::Tensor *output) {
    auto in_dims = input.dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_MOBILE_ENFORCE((vector.numel() == size),
                          "vector.numel() must be equal to size.");
    PADDLE_MOBILE_ENFORCE((output->dims() == in_dims),
                          "output->dims() must be equal to in_dims.");

    auto *input_data = input.data<T>();
    auto *out_data = output->data<T>();
    auto *vec_data = vector.data<T>();
    for (int64_t i = 0; i < in_dims[0]; ++i) {
      for (int64_t j = 0; j < size; ++j) {
        out_data[i * size + j] = input_data[i * size + j] + vec_data[j];
      }
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
