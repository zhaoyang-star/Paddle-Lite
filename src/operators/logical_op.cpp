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

#include "operators/logical_op.h"

namespace paddle_mobile {
namespace operators {

#define DEFINE_LOGICAL_INFERSHAPE(OpName)                   \
  template <typename T>                                     \
  void OpName##Op<T>::InferShape() const {                  \
    const auto &input_dims = this->param_.InputX()->dims(); \
    this->param_.Out()->Resize(input_dims);                 \
  }

#ifdef LOGICAL_AND_OP
DEFINE_LOGICAL_INFERSHAPE(LogicalAnd);
#endif  // TLOGICAL_AND_OP

#ifdef LOGICAL_OR_OP
DEFINE_LOGICAL_INFERSHAPE(LogicalOr);
#endif  // TLOGICAL_OR_OP

#ifdef LOGICAL_NOT_OP
DEFINE_LOGICAL_INFERSHAPE(LogicalNot);
#endif  // LOGICAL_NOT_OP

#ifdef LOGICAL_XOR_OP
DEFINE_LOGICAL_INFERSHAPE(LogicalXor);
#endif  // TLOGICAL_XOR_OP

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef LOGICAL_AND_OP

REGISTER_OPERATOR(logical_and, ops::LogicalAndOp);
#endif  // LOGICAL_AND_OP

#ifdef LOGICAL_OR_OP

REGISTER_OPERATOR(logical_or, ops::LogicalOrOp);
#endif  // LOGICAL_OR_OP

#ifdef LOGICAL_NOT_OP

REGISTER_OPERATOR(logical_not, ops::LogicalNotOp);
#endif  // LOGICAL_NOT_OP

#ifdef LOGICAL_XOR_OP

REGISTER_OPERATOR(logical_xor, ops::LogicalXorOp);
#endif  // LOGICAL_XOR_OP
