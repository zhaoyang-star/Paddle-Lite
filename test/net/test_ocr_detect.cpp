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

#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"

int main() {
  PaddleMobileConfigInternal configInternalGpu = PaddleMobileConfigInternal();
  configInternalGpu.running_expected_map_.insert(
      std::make_pair("conv", TYPE_GPU));

  configInternalGpu.running_expected_map_.insert(
      std::make_pair("batch_norm", TYPE_GPU));

  configInternalGpu.running_expected_map_.insert(
      std::make_pair("sigmoid", TYPE_GPU));

  configInternalGpu.running_expected_map_.insert(
      std::make_pair("relu", TYPE_GPU));

  configInternalGpu.running_expected_map_.insert(
      std::make_pair("concat", TYPE_CPU));

  paddle_mobile::PaddleMobile<float> paddle_mobile(configInternalGpu);
  //    paddle_mobile.SetThreadNum(4);
  auto time1 = paddle_mobile::time();
#ifdef PADDLE_MOBILE_CL
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#endif

  if (paddle_mobile.Load(std::string(g_ocr_detect) + "/model",
                         std::string(g_ocr_detect) + "/params", true)) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time1) << "ms"
              << std::endl;

    paddle_mobile::framework::Tensor input;
    std::vector<int64_t> dims{1, 3, 512, 512};
    paddle_mobile::framework::DDim in_shape =
        paddle_mobile::framework::make_ddim(dims);
    SetupTensor<float>(&input, in_shape, 0.f, 255.f);

    //    paddle_mobile.Predict(input);

    // 预热十次
    //    for (int i = 0; i < 10; ++i) {
    //        auto vec_result = paddle_mobile.Predict(input);
    //    }
    auto time3 = time();
    //    for (int i = 0; i < 10; ++i) {
    auto vec_result = paddle_mobile.Predict(input);
    //    }
    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) << "ms"
              << std::endl;
  }

  return 0;
}
