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

#include "io/paddle_test_inference_api.h"
#include "io/paddle_mobile.h"

namespace paddle_mobile {

template <typename T>
double PaddleTester<T>::CaculateCpuPredictTime() {
  PaddleMobile<T> paddle_mobile;
  return paddle_mobile.GetCpuPredictTime();
}

template <typename T>
double PaddleTester<T>::CaculateGpuPredictTime(std::string *cl_path) {
#ifdef PADDLE_MOBILE_CL

  PaddleMobile<T> paddle_mobile;
  if (cl_path) {
    paddle_mobile.SetCLPath(*cl_path);
  }

  return paddle_mobile.GetGpuPredictTime();
#else
  return -1;
#endif
}
template class PaddleTester<float>;

}  // namespace paddle_mobile
