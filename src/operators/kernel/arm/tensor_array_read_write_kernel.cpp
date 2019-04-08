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

#include "operators/kernel/tensor_array_read_write_kernel.h"

namespace paddle_mobile {
namespace operators {

#ifdef WRITE_TO_ARRAY_OP
template <>
bool WriteToArrayKernelCpu<float>::Init(WriteToArrayParam *param) {
  return true;
}

template <>
void WriteToArrayKernelCpu<float>::Compute(const WriteToArrayParam &param) {
  int64_t offset = param.index_->InnerLoDTensor()->data<int64_t>()[0];
  if (offset >= param.output_->size()) {
    param.output_->resize(offset + 1);
  }

  framework::LoDTensor *out_tensor = param.output_->at(offset).InnerLoDTensor();
  out_tensor->set_lod(param.input_->InnerLoDTensor()->lod());
  if (param.input_->InnerLoDTensor()->memory_size() > 0) {
    TensorCopy(*(param.input_->InnerLoDTensor()), out_tensor);
  }
}
#endif  // WRITE_TO_ARRAY_OP

#ifdef READ_FROM_ARRAY_OP
template <>
bool ReadFromArrayKernelCpu<float>::Init(ReadFromArrayParam *param) {
  return true;
}

template <>
void ReadFromArrayKernelCpu<float>::Compute(const ReadFromArrayParam &param) {
  int64_t offset = param.index_->InnerLoDTensor()->data<int64_t>()[0];
  if (offset < param.input_->size()) {
    TensorCopy(*param.input_->at(offset).InnerLoDTensor(),
               param.output_->InnerLoDTensor());
    param.output_->InnerLoDTensor()->set_lod(
        param.input_->at(offset).InnerLoDTensor()->lod());
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(
        "Can not read tensor which index is `%d` since it only has `%d` inputs",
        offset, param.input_->size());
  }
}
#endif  // READ_FROM_ARRAY_OP

}  // namespace operators
}  // namespace paddle_mobile
