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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/types.h"
#include "common/util.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace framework {

template <typename Device, typename T = float>
class ExecutorCore {
 public:
  void LoadMemInternal(void **data, LoDTensor *tensor,bool quant_uint8 = false);

  void LoadMemory(
      void **data, const std::shared_ptr<VarDesc> var_desc, LoDTensor *tensor) ;
  int batch_size_;
  bool use_optimize_;
  bool lod_mode_;
  PaddleMobileConfigInternal config_;
  Program<float> program_;
  std::shared_ptr<ProgramDesc> program_desc_;
  std::unordered_map<std::string, int> feed_indices_;
  std::unordered_map<std::string, int> fetch_indices_;


  // for super resoltion
  DDim input_dim_last_;
};

}  // namespace framework
}  // namespace paddle_mobile
