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

#ifdef FUSION_CONVBNRELU_OP

#include "operators/kernel/conv_bn_relu_kernel.h"
#include <cmath>
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNReluKernelGpu<float>::Init(
    FusionConvBNReluParam*param) {
  PADDLE_MOBILE_ENFORCE(
      param->Filter()->ClImage()->dims()[2] == param->Filter()->ClImage()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");
  const framework::CLImage *mean = param->InputMean()->ClImage();
  const framework::CLImage *variance = param->InputVariance()->ClImage();
  const framework::CLImage *scale = param->InputScale()->ClImage();
  const framework::CLImage *bias = param->InputBias()->ClImage();
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

  Variable *scale_var = param->GetScope()->Var();
  Variable *bias_var = param->GetScope()->Var();
  framework::MobileTensor *new_scale_w = scale_var->GetMutable<framework::MobileTensor>();
  framework::MobileTensor *new_bias_w = bias_var->GetMutable<framework::MobileTensor>();

  auto *new_scale = new_scale_w->MuteClImage();
  auto *new_bias = new_bias_w->MuteClImage();

  //  framework::CLImage *new_scale = new framework::ClImage();
  new_scale->SetTensorData(new_scale_ptr, variance->dims());
  new_scale->InitCLImage(this->cl_helper_.CLContext(),
                         this->cl_helper_.CLCommandQueue());

  //  framework::CLImage *new_bias = new framework::ClImage();
  new_bias->SetTensorData(new_bias_ptr, variance->dims());
  new_bias->InitCLImage(this->cl_helper_.CLContext(),
                        this->cl_helper_.CLCommandQueue());

  param->SetNewScale(new_scale_w);
  param->SetNewBias(new_bias_w);


  delete[](new_scale_ptr);
  delete[](new_bias_ptr);

  PADDLE_MOBILE_ENFORCE(
      param->Filter()->ClImage()->dims()[2] == param->Filter()->ClImage()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");

  int offset = static_cast<int>(param->Filter()->ClImage()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);

  param->SetOffset(offset);

  if (param->Filter()->ClImage()->dims()[2] == 1 && param->Filter()->ClImage()->dims()[3] == 1) {
    param->Filter()->ClImage()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_1x1_spl", "conv_bn_relu_kernel.cl");
    DLOG << " conv bn relu conv 1x1";
  } else if (param->Filter()->ClImage()->dims()[1] == 1 &&
      param->Input()->ClImage()->dims()[1] == param->Output()->ClImage()->dims()[1] &&
      param->Filter()->ClImage()->dims()[2] == 3) {
    param->Filter()->ClImage()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("depth_conv_3x3", "conv_bn_relu_kernel.cl");
    DLOG << " conv bn relu depth_conv_3x3";

  } else if (param->Filter()->ClImage()->dims()[2] == 3 &&
      param->Filter()->ClImage()->dims()[3] == 3) {
    param->Filter()->ClImage()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_3x3", "conv_bn_relu_kernel.cl");
    DLOG << " conv bn relu conv_3x3";
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }
  return true;
}

template <>
void ConvBNReluKernelGpu<float>::Compute(
    const FusionConvBNReluParam&param) {
  ConvAddBnRelu(this->cl_helper_, param, true, nullptr, param.NewScale()->ClImage(),
                param.NewBias()->ClImage());
}
template class ConvBNReluKernelGpu<float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
