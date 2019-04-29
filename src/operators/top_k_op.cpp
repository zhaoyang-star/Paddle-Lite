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

#ifdef TOP_K_OP

#include "operators/top_k_op.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
void TopKOp<T>::InferShape() const {
  const int k = this->param_.k_;
  auto dims = this->param_.input_->LodTensor()->dims();
  // should check k <= dims[-1] && k >= 1
  dims[dims.size() - 1] = k;
  this->param_.output_->LodTensor()->Resize(dims);
  this->param_.indices_->LodTensor()->Resize(dims);
  this->param_.output_->LodTensor()->set_lod(
      this->param_.input_->LodTensor()->lod());
  this->param_.indices_->LodTensor()->set_lod(
      this->param_.input_->LodTensor()->lod());
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

REGISTER_OPERATOR(top_k, ops::TopKOp);

#endif  // TOP_K_OP
