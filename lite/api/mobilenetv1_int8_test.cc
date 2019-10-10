// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"

#define MV6S
#ifdef MV6S
#include <unistd.h>
#endif

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir, "", "", preferred_place, valid_places);

  auto* input_tensor = predictor.GetInput(0);
#ifdef MV6S
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 64, 64})));
#else
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
#endif
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }





//#define SLEEP_ENABLE
#ifdef SLEEP_ENABLE
  for (auto sleep_time : {0,
                          100,
                          200,
                          500,
                          1000,
                          1500,
                          2000,
                          5000,
                          6000,
                          8000,
                          10000,
                          12000,
                          15000,
                          20000,
                          25000,
                          30000}) {
#else
  for (auto sleep_time : {0}) {
#endif
    size_t sum = 0;
    for (int i = 0; i < FLAGS_repeats; ++i) {
      auto start = GetCurrentUS();
      predictor.Run();
      auto end = GetCurrentUS();
      sum += end - start;
      usleep(sleep_time);
    }

    LOG(INFO) << "================== Speed Report ===================";
    LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num "
              << FLAGS_threads << ", warmup: " << FLAGS_warmup
              << ", repeats: " << FLAGS_repeats
              << ", sleep_time: " << sleep_time << ", spend "
              << (/*end - start*/ sum) / FLAGS_repeats / 1000.0
              << " ms in average.";

    std::cout << "================== Speed Report ==================="
              << std::endl;
    std::cout << "Model: " << FLAGS_model_dir << ", threads num "
              << FLAGS_threads << ", warmup: " << FLAGS_warmup
              << ", repeats: " << FLAGS_repeats
              << ", sleep_time: " << sleep_time << ", spend "
              << (/*end - start*/ sum) / FLAGS_repeats / 1000.0
              << " ms in average." << std::endl;
  }
}

TEST(MobileNetV1, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kInt8)},
  });

  TestModel(valid_places, Place({TARGET(kARM), PRECISION(kInt8)}));
}

}  // namespace lite
}  // namespace paddle
