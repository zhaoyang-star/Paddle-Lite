/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef GRU_OP
#pragma once

#include "common/types.h"
#include "operators/math/activation.h"
#include "operators/math/gemm/cblas.h"
#include "operators/math/gru_compute.h"
#include "operators/math/gru_cpu_kernel.h"
namespace paddle_mobile {
namespace operators {
namespace math {

template <typename T>
struct GRUMetaValue {
  T *gate_weight;
  T *state_weight;
  T *gate_value;
  T *reset_output_value;
  T *output_value;
  T *prev_out_value;
};

template <typename T>
inline void forward_reset_output(GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_node) {
  for (int b = 0; b < batch_size; ++b) {
    switch (active_node) {
      case RELU:
        FORWARD_RESET_OUTPUT(RELU, value, frame_size);
        break;
      case SIGMOID:
        FORWARD_RESET_OUTPUT(SIGMOID, value, frame_size);
        break;
      case TANH:
        FORWARD_RESET_OUTPUT(TANH, value, frame_size);
        break;
      default:
        FORWARD_RESET_OUTPUT(IDENTITY, value, frame_size);
    }
    value.gate_value += frame_size * 3;
    value.reset_output_value += frame_size;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }
  }
}

template <typename T>
inline void forward_final_output(GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_node) {
  for (int b = 0; b < batch_size; ++b) {
    switch (active_node) {
      case RELU:
        FORWARD_FINAL_OUTPUT(RELU, value, frame_size);
        break;
      case SIGMOID:
        FORWARD_FINAL_OUTPUT(SIGMOID, value, frame_size);
        break;
      case TANH:
        FORWARD_FINAL_OUTPUT(TANH, value, frame_size);
        break;
      default:
        FORWARD_FINAL_OUTPUT(IDENTITY, value, frame_size);
    }
    value.gate_value += frame_size * 3;
    value.output_value += frame_size;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }
  }
}

template <typename T>
struct GRUUnitFunctor {
  static void compute(GRUMetaValue<float> value, int frame_size, int batch_size,
                      const ActivationType active_node,
                      const ActivationType active_gate) {
    if (value.prev_out_value) {
      cblas_sgemm(false, false, batch_size, frame_size * 2, frame_size, 1.f,
                  value.prev_out_value, frame_size, value.gate_weight,
                  frame_size * 2, 1.f, value.gate_value, frame_size * 3);
    }

    forward_reset_output(value, frame_size, batch_size, active_gate);

    if (value.prev_out_value) {
      cblas_sgemm(false, false, batch_size, frame_size, frame_size, 1.f,
                  value.reset_output_value, frame_size, value.state_weight,
                  frame_size, 1.f, value.gate_value + frame_size * 2,
                  frame_size * 3);
    }

    forward_final_output(value, frame_size, batch_size, active_node);
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
#endif
