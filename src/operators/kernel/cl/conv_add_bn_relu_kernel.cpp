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

#ifdef FUSION_CONVADDBNRELU_OP

#include "operators/kernel/conv_add_bn_relu_kernel.h"
#include <cmath>
#include "framework/cl/cl_image.h"
#include "framework/cl/cl_tool.h"
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

namespace paddle_mobile {
namespace operators {
template <>
bool ConvAddBNReluKernelGpu<float>::Init(FusionConvAddBNReluParam *param) {
  PADDLE_MOBILE_ENFORCE(param->Filter()->InnerCLImage()->dims()[2] ==
                                param->Filter()->InnerCLImage()->dims()[3] &&
                            param->Paddings()[0] == param->Paddings()[1],
                        "need equal");
  param->Bias()->InnerCLImage()->InitCLImage(cl_helper_.CLContext(),
                                             cl_helper_.CLCommandQueue());
  const framework::CLImage *mean = param->InputMean()->InnerCLImage();
  const framework::CLImage *variance = param->InputVariance()->InnerCLImage();
  const framework::CLImage *scale = param->InputScale()->InnerCLImage();
  const framework::CLImage *bias = param->InputBias()->InnerCLImage();
  const float epsilon = param->Epsilon();
  const int C = mean->numel();
  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  float inv_std_ptr[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }
  float *new_scale_ptr = new float[C];
  float *new_bias_ptr = new float[C];

  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
  }

  auto *new_scale_w = param->CreateNewScale<framework::TensorWrapper>();
  auto *new_bias_w = param->CreateNewBiase<framework::TensorWrapper>();
  auto *new_scale = new_scale_w->MuteClImage();
  auto *new_bias = new_bias_w->MuteClImage();

  new_scale->SetTensorData(new_scale_ptr, variance->dims());
  new_scale->InitCLImage(this->cl_helper_.CLContext(),
                         cl_helper_.CLCommandQueue());

  new_bias->SetTensorData(new_bias_ptr, variance->dims());
  new_bias->InitCLImage(this->cl_helper_.CLContext(),
                        cl_helper_.CLCommandQueue());

  param->SetNewScale(new_scale_w);
  param->SetNewBias(new_bias_w);

  delete[](new_scale_ptr);
  delete[](new_bias_ptr);

  PADDLE_MOBILE_ENFORCE(param->Filter()->InnerCLImage()->dims()[2] ==
                                param->Filter()->InnerCLImage()->dims()[3] &&
                            param->Paddings()[0] == param->Paddings()[1],
                        "need equal");

  int offset =
      static_cast<int>(param->Filter()->InnerCLImage()->dims()[2]) / 2 -
      static_cast<int>(param->Paddings()[1]);

  param->SetOffset(offset);

  /*
  if (param->Filter()->dims()[2] == 1 &&
      param->Filter()->dims()[3] == 1 &&
      (param->Filter()->dims()[0] % 16) == 0) {
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_1x1_4", "conv_add_bn_relu_kernel.cl");
    DLOG << " conv add bn relu conv 1x1 4";
  }
  */
  if (param->Filter()->dims()[2] == 1 && param->Filter()->dims()[3] == 1) {
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_1x1_spl", "conv_add_bn_relu_kernel.cl");

    DLOG << " conv add bn relu conv 1x1";
  } else if (param->Filter()->InnerCLImage()->dims()[1] == 1 &&
             param->Input()->InnerCLImage()->dims()[1] ==
                 param->Output()->InnerCLImage()->dims()[1] &&
             param->Filter()->InnerCLImage()->dims()[2] == 3) {
    param->Filter()->InnerCLImage()->InitDWImage(cl_helper_.CLContext(),
                                                 cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("depth_conv_3x3", "conv_add_bn_relu_kernel.cl");
    DLOG << " conv add bn relu depth_conv_3x3";

  } else if (param->Filter()->InnerCLImage()->dims()[2] == 3 &&
             param->Filter()->InnerCLImage()->dims()[3] == 3) {
    param->Filter()->InnerCLImage()->InitCLImage(cl_helper_.CLContext(),
                                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_3x3", "conv_add_bn_relu_kernel.cl");
    DLOG << " conv add bn relu conv_3x3";
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }
  DLOG << "ConvAddBNReluKernelGpu end: ";
  return true;
}

template <>
void ConvAddBNReluKernelGpu< float>::Compute(
    const FusionConvAddBNReluParam &param) {
  ConvAddBnRelu(this->cl_helper_, param, true, param.Bias()->InnerCLImage(), param.NewScale()->InnerCLImage(),
                param.NewBias()->InnerCLImage());
}

template class ConvAddBNReluKernelGpu< float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
