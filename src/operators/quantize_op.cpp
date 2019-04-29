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

#ifdef QUANT_OP

#include "operators/quantize_op.h"
#include <vector>

namespace paddle_mobile {
namespace operators {

template <typename T>
void QuantizeOp<T>::InferShape() const {
  const auto &input_dims = this->param_.input_->LodTensor()->dims();
  this->param_.output_->LodTensor()->Resize(input_dims);
  auto scale_dims = framework::make_ddim(std::vector<int>{1});
  this->param_.online_scale_->LodTensor()->Resize(scale_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

REGISTER_OPERATOR(quantize, ops::QuantizeOp);

#endif  // QUANT_OP
