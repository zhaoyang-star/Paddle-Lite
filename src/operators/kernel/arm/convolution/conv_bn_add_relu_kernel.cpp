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

#ifdef FUSION_CONVBNADDRELU_OP

#include "operators/kernel/conv_bn_add_relu_kernel.h"
#include <cmath>
#include "operators/kernel/arm/convolution/conv_common.h"
#include "operators/kernel/central-arm-func/conv_arm_func.h"
#include "operators/math/element_wise.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNAddReluKernelCpu<float>::Init(FusionConvBNAddReluParam *param) {
  const Tensor *mean = param->InputMean()->LodTensor();
  const Tensor *variance = param->InputVariance()->LodTensor();
  const Tensor *scale = param->InputScale()->LodTensor();
  const Tensor *bias = param->InputBias()->LodTensor();
  const float epsilon = param->Epsilon();

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = const_cast<float *>(scale->data<float>());
  auto bias_ptr = const_cast<float *>(bias->data<float>());

  for (int c = 0; c < scale->numel(); ++c) {
    float inv_scale = 1.f / (pow(variance_ptr[c] + epsilon, 0.5));
    bias_ptr[c] -= inv_scale * scale_ptr[c] * mean_ptr[c];
    scale_ptr[c] *= inv_scale;
  }

  InitBaseConvKernel(param);
  return true;
}

template <>
void ConvBNAddReluKernelCpu<float>::Compute(
    const FusionConvBNAddReluParam &param) {
  switch (param.ExecMode()) {
    case ConvParam::EXEC_DEPTHWISE3x3S1_FLOAT:
    case ConvParam::EXEC_DEPTHWISE3x3S2_FLOAT:
      DepthwiseConv3x3<float, float>(param);
      break;
    case ConvParam::EXEC_DEPTHWISE5x5_FLOAT:
      DepthwiseConv5x5<float, float>(param);
      break;
    case ConvParam::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<8, 3>(param);
      break;
    case ConvParam::EXEC_GEMM_FLOAT:
      GemmConv<float, float>(param);
      break;
    case ConvParam::EXEC_SLIDINGWINDOW3x3S1_FLOAT:
    case ConvParam::EXEC_SLIDINGWINDOW3x3S2_FLOAT:
      SlidingwindowConv3x3<float, float>(param);
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }

  if (param.Bias()->LodTensor()->dims() == param.Output()->LodTensor()->dims()) {
    math::ScaleAddChannelWise<RELU>(param.Output()->LodTensor(), param.InputScale()->LodTensor(),
                                    param.InputBias()->LodTensor(), param.Bias()->LodTensor(),
                                    param.Output()->LodTensor());
  } else {
    math::ScaleAddChannelWise<IDENTITY>(param.Output()->LodTensor(), param.InputScale()->LodTensor(),
                                        param.InputBias()->LodTensor(), param.Output()->LodTensor());
    math::AddElememtWise<RELU>(param.Output()->LodTensor(),
                               param.Bias()->LodTensor(), param.Axis(),
                               param.Output()->LodTensor());
  }
}

template class ConvBNAddReluKernelCpu<float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
