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

#include <memory>
#include <string>
#include <vector>

#include "CL/cl.h"
#include "framework/cl/cl_tensor.h"

namespace paddle_mobile {
namespace framework {

#ifdef PADDLE_MOBILE_DEBUG
Print &operator<<(Print &printer, const CLTensor &tensor) {
  auto pTensor = const_cast<CLTensor *>(&tensor);

  printer << " dims: " << tensor.dims() << "\n";
  int stride = tensor.numel() / 20;
  stride = stride > 0 ? stride : 1;
  for (int i = 0; i < tensor.numel(); i += stride) {
    if (tensor.type() == type_id<float>()) {
      printer << pTensor->Data<float>()[i] << " ";
    } else if (tensor.type() == type_id<int32_t>()) {
      printer << pTensor->Data<int32_t>()[i] << " ";
    } else if (tensor.type() == type_id<int64_t>()) {
      printer << pTensor->Data<int64_t>()[i] << " ";
    } else if (tensor.type() == type_id<int8_t>()) {
      printer << static_cast<int>(pTensor->Data<int8_t>()[i]) << " ";
    } else if (tensor.type() == type_id<int32_t>()) {
      printer << pTensor->Data<int32_t>()[i] << " ";
    } else if (tensor.type() == type_id<bool>()) {
      printer << pTensor->Data<bool>()[i] << " ";
    }
  }
  return printer;
}
#endif  // PADDLE_MOBILE_DEBUG}
}  // namespace framework
}  // namespace paddle_mobile
