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

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#ifdef FUSION_DEQUANT_BN_OP
DECLARE_KERNEL_ALL_WITH_PARAMS(FusionDequantBN, FusionDequantBNParam);
#endif

#ifdef FUSION_DEQUANT_BN_RELU_OP
DECLARE_KERNEL_ALL_WITH_PARAMS(FusionDequantBNRelu, FusionDequantBNParam);
#endif

#ifdef FUSION_DEQUANT_ADD_BN_OP
DECLARE_KERNEL_ALL_WITH_PARAMS(FusionDequantAddBN, FusionDequantAddBNParam);
#endif

#ifdef FUSION_DEQUANT_ADD_BN_RELU_OP
DECLARE_KERNEL_ALL_WITH_PARAMS(FusionDequantAddBNRelu, FusionDequantAddBNParam);
#endif

#ifdef FUSION_DEQUANT_ADD_BN_QUANT_OP
DECLARE_KERNEL_ALL_WITH_PARAMS(FusionDequantAddBNQuant,
                               FusionDequantAddBNQuantParam);
#endif

#ifdef FUSION_DEQUANT_ADD_BN_RELU_QUANT_OP
DECLARE_KERNEL_ALL_WITH_PARAMS(FusionDequantAddBNReluQuant,
                               FusionDequantAddBNQuantParam);
#endif

}  // namespace operators
}  // namespace paddle_mobile
