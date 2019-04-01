/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include "framework/cl/cl_helper.h"

namespace paddle_mobile {
namespace framework {
class ImageConverterHelper {
 public:
  void Init(CLScope * scope) {
    helper_global = CLHelper(scope);
    helper_global.AddKernel("feed", "feed_kernel.cl");
    helper_global.AddKernel("fetch", "fetch_kernel.cl");
  }
  static ImageConverterHelper *Instance() {
    static ImageConverterHelper converter_;
    return &converter_;
  }

  CLHelper *GetClHelper(){
    return &helper_global;
  }
private:
  CLHelper helper_global;
};
}  // namespace framework
}  // namespace paddle_mobile
