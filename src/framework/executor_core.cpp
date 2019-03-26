///* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */
//
//#pragma once
//
//#include "framework/executor_core.h"
//#include <map>
//#include <memory>
//#include <string>
//#include <utility>
//#include <vector>
//#include "common/types.h"
//#include "common/util.h"
//#include "framework/lod_tensor.h"
//#include "framework/operator.h"
//#include "framework/program/program.h"
//#include "framework/tensor.h"
//
// namespace paddle_mobile {
// namespace framework {
// template <typename Device, typename T>
// void ExecutorCore<Device, T>::LoadMemInternal(void **data, LoDTensor *tensor,
//                                              bool quant_uint8) {
//  char **data_buf = reinterpret_cast<char **>(data);
//  int64_t size = tensor->numel();
//  T *tensor_data = tensor->mutable_data<T>();
//  if (quant_uint8) {
//    // should be moved into operator init function
//    float min_value;
//    float max_value;
//    memory::Copy(&min_value, *data_buf, sizeof(float));
//    memory::Copy(&max_value, *data_buf + sizeof(float), sizeof(float));
//    *data_buf += 2 * sizeof(float);
//    const float factor = (max_value - min_value) / 255.0;
//    const uint8_t *uint8_data = reinterpret_cast<uint8_t *>(*data_buf);
//    for (int k = 0; k < size; ++k) {
//      tensor_data[k] = uint8_data[k] * factor + min_value;
//    }
//    *data_buf += size * sizeof(uint8_t);
//  } else {
//    memory::Copy(tensor_data, *data_buf, size * sizeof(T));
//    *data_buf += size * sizeof(T);
//  }
//}
//
// template <typename Device, typename T>
// void ExecutorCore<Device, T>::LoadMemory(
//    void **data, const std::shared_ptr<VarDesc> var_desc, LoDTensor *tensor) {
//  char **data_buf = reinterpret_cast<char **>(data);
//  // version
//  uint32_t version = *(reinterpret_cast<uint32_t *>(*data_buf));
//  *data_buf += sizeof(uint32_t);
//  // lod information
//  // uint64_t lod_level = *(reinterpret_cast<uint64_t *>(*data_buf));
//  uint64_t lod_level = 0;
//  memory::Copy(&lod_level, *data_buf, sizeof(uint64_t));
//  *data_buf += sizeof(uint64_t);
//
//  auto *lod = tensor->mutable_lod();
//  lod->resize(lod_level);
//  for (uint64_t i = 0; i < lod_level; ++i) {
//    uint64_t size = *(reinterpret_cast<uint64_t *>(*data_buf));
//    *data_buf += sizeof(uint64_t);
//    std::vector<size_t> tmp_dim(size / sizeof(size_t));
//    memory::Copy(tmp_dim.data(), *data_buf, size);
//    (*lod)[i] = std::move(tmp_dim);
//    *data_buf += size;
//  }
//  // tensor version
//  uint32_t tensor_version = *(reinterpret_cast<uint32_t *>(*data_buf));
//  *data_buf += sizeof(uint32_t);
//  // tensor desc size
//  int32_t tensor_desc_size = *(reinterpret_cast<int32_t *>(*data_buf));
//  *data_buf += sizeof(int32_t);
//  // skip tensor desc
//  *data_buf += tensor_desc_size;
//
//  const TensorDesc &tensor_desc = var_desc->Tensor_desc();
//  tensor->Resize(make_ddim(tensor_desc.Dims()));
//  // parse tensor from stream
//  switch (tensor_desc.DataType()) {
//    case VARTYPE_TYPE_FP32:
//      LoadMemInternal(reinterpret_cast<void **>(data_buf), tensor,
//                      program_.quantification);
//      break;
//    case VARTYPE_TYPE_INT8:
//      LoadMemInternal(reinterpret_cast<void **>(data_buf), tensor);
//      break;
//    case VARTYPE_TYPE_INT32:
//      LoadMemInternal(reinterpret_cast<void **>(data_buf), tensor);
//      break;
//    default:
//      LOG(kLOG_ERROR) << "data type is not supported";
//  }
//}
///*template class ExecutorCore< float>;
// template class ExecutorCore<GPU_CL, float>;
// template class ExecutorCore<FPGA, float>;*/
//}  // namespace framework
//}  // namespace paddle_mobile
