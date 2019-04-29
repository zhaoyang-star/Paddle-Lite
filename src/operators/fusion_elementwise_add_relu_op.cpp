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

#ifdef FUSION_ELEMENTWISEADDRELU_OP

#include "operators/fusion_elementwise_add_relu_op.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
void FusionElementwiseAddReluOp<T>::InferShape() const {
  auto x_dim = this->param_.InputX()->InnerLoDTensor()->dims();
  this->param_.Out()->InnerLoDTensor()->Resize(x_dim);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
REGISTER_FUSION_MATCHER(fusion_elementwise_add_relu,
                        ops::FusioneElementwiseAddReluMatcher);

// REGISTER_OPERATOR(fusion_elementwise_add_relu,
//                      ops::FusionElementwiseAddReluOp);
REGISTER_OPERATOR(fusion_elementwise_add_relu, ops::FusionElementwiseAddReluOp);
// fixme use #if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
#endif
