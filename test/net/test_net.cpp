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

#include <fstream>
#include <iostream>
#include <string>
#include "../test_helper.h"
#include "../test_include.h"

void test(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  test(argc, argv);
  return 0;
}

void test(int argc, char *argv[]) {
  int arg_index = 1;
  bool fuse = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  bool enable_memory_optimization = std::stoi(argv[arg_index]) == 1;
  arg_index++;

  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  std::cout << "testing cpu yyz " << std::endl;

  int dim_count = std::stoi(argv[arg_index]);
  arg_index++;
  int size = 1;
  std::vector<int64_t> dims;
  for (int i = 0; i < dim_count; i++) {
    int64_t dim = std::stoi(argv[arg_index + i]);
    size *= dim;
    dims.push_back(dim);
  }
  arg_index += dim_count;

  bool is_lod = std::stoi(argv[arg_index]) == 1;
  arg_index++;

  int var_count = std::stoi(argv[arg_index]);
  arg_index++;
  bool is_sample_step = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  int sample_arg = std::stoi(argv[arg_index]);
  int sample_step = sample_arg;
  int sample_num = sample_arg;
  arg_index++;
  std::vector<std::string> var_names;
  for (int i = 0; i < var_count; i++) {
    std::string var_name = argv[arg_index + i];
    var_names.push_back(var_name);
  }
  arg_index += var_count;
  bool check_shape = std::stoi(argv[arg_index]) == 1;
  arg_index++;

  auto time1 = time();
  if (paddle_mobile.Load("./checked_model/model", "./checked_model/params",
                         fuse, false, 1, true)) {
    auto time2 = time();
    std::cout << "auto-test"
              << " load-time-cost :" << time_diff(time1, time2) << "ms"
              << std::endl;

    float *input_data_array = new float[size];
    std::ifstream in("input.txt", std::ios::in);
    for (int i = 0; i < size; i++) {
      float num;
      in >> num;
      input_data_array[i] = num;
    }
    in.close();

    auto time3 = time();
    std::vector<float> input_data;
    for (int i = 0; i < size; i++) {
      float num = input_data_array[i];
      input_data.push_back(num);
    }
    paddle_mobile::framework::Tensor input_tensor(
        input_data, paddle_mobile::framework::make_ddim(dims));
    auto time4 = time();
    std::cout << "auto-test"
              << " preprocess-time-cost :" << time_diff(time3, time4) << "ms"
              << std::endl;

    paddle_mobile.Predict(input_tensor);
    for (auto var_name : var_names) {
      auto out = paddle_mobile.Fetch(var_name);
      auto len = out->numel();
      if (len == 0) {
        continue;
      }
      if (out->memory_size() == 0) {
        continue;
      }
      auto data = out->data<float>();
      std::string sample = "";
      if (check_shape) {
        for (int i = 0; i < out->dims().size(); i++) {
          sample += " " + std::to_string(out->dims()[i]);
        }
      }
      if (!is_sample_step) {
        sample_step = len / sample_num;
      }
      if (sample_step <= 0) {
        sample_step = 1;
      }
      for (int i = 0; i < len; i += sample_step) {
        sample += " " + std::to_string(data[i]);
      }
      std::cout << "auto-test"
                << " var " << var_name << sample << std::endl;
    }
    std::cout << std::endl;
  }
}
