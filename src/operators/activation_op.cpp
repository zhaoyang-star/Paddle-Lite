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

#include "operators/activation_op.h"

namespace paddle_mobile {
namespace operators {

#define DEFINE_ACTIVATION_INFERSHAPE(OpName)                             \
  template <typename T>                                                  \
  void OpName##Op<T>::InferShape() const {                               \
    const auto &input_dims = this->param_.InputX()->LodTensor()->dims(); \
    this->param_.Out()->LodTensor()->Resize(input_dims);                 \
  }

#ifdef RELU_OP
DEFINE_ACTIVATION_INFERSHAPE(Relu);
DEFINE_ACTIVATION_INFERSHAPE(Relu6);
#endif  // RELU_OP

#ifdef SIGMOID_OP
DEFINE_ACTIVATION_INFERSHAPE(Sigmoid);
namespace ops = paddle_mobile::operators;
#endif  // SIGMOID_OP

#ifdef TANH_OP
DEFINE_ACTIVATION_INFERSHAPE(Tanh);
#endif  // TANH_OP

#ifdef LOG_OP
DEFINE_ACTIVATION_INFERSHAPE(Log);
#endif  // LOG_OP

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef RELU_OP
REGISTER_OPERATOR(relu, ops::ReluOp);
REGISTER_OPERATOR(relu6, ops::Relu6Op);
#endif  // RELU_OP

#ifdef SIGMOID_OP
REGISTER_OPERATOR(sigmoid, ops::SigmoidOp);
#endif  // SIGMOID_OP

#ifdef TANH_OP
REGISTER_OPERATOR(tanh, ops::TanhOp);
#endif  // TANH_OP

#ifdef LOG_OP
REGISTER_OPERATOR(log, ops::LogOp);
#endif  // LOG_OP
