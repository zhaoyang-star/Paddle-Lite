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

#include <string>
#include "framework/operator.h"
#include "operators/kernel/activation_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#ifdef RELU_OP
DECLARE_OPERATOR_MIXED_WITH_PARAMS(Relu, ReluParam, ReluKernel);
DECLARE_OPERATOR_WITH_PARAMS(Relu6, ReluParam, Relu6Kernel);
#endif

#ifdef SIGMOID_OP
DECLARE_OPERATOR_MIXED_WITH_PARAMS(Sigmoid, SigmoidParam, SigmoidKernel);
#endif

#ifdef TANH_OP
DECLARE_OPERATOR_WITH_PARAMS(Tanh, TanhParam, TanhKernel);
#endif

#ifdef LOG_OP
DECLARE_OPERATOR_WITH_PARAMS(Log, ReluParam, LogKernel);
#endif

}  // namespace operators
}  // namespace paddle_mobile
