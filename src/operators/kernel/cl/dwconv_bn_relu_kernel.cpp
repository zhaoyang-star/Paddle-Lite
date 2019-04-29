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

#ifdef FUSION_DWCONVBNRELU_OP

#include "operators/kernel/dwconv_bn_relu_kernel.h"
#include <cmath>

namespace paddle_mobile {
namespace operators {

template <>
bool DWConvBNReluKernelGpu<float>::Init(FusionDWConvBNReluParam *param) {
  PADDLE_MOBILE_ENFORCE(param->Filter()->ClImage()->dims()[2] ==
                                param->Filter()->ClImage()->dims()[3] &&
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
  framework::MobileTensor *new_scale_w =
      scale_var->GetMutable<framework::MobileTensor>();
  framework::MobileTensor *new_bias_w =
      bias_var->GetMutable<framework::MobileTensor>();

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

  PADDLE_MOBILE_ENFORCE(param->Filter()->ClImage()->dims()[2] ==
                                param->Filter()->ClImage()->dims()[3] &&
                            param->Paddings()[0] == param->Paddings()[1],
                        "need equal");

  int offset = static_cast<int>(param->Filter()->ClImage()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);

  param->SetOffset(offset);

  param->Filter()->ClImage()->InitDWImage(cl_helper_.CLContext(),
                                          cl_helper_.CLCommandQueue());
  this->cl_helper_.AddKernel("depth_conv_3x3", "conv_bn_relu_kernel.cl");
  DLOG << " conv bn relu depth_conv_3x3";

  return true;
}

template <>
void DWConvBNReluKernelGpu<float>::Compute(
    const FusionDWConvBNReluParam &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size =
      this->cl_helper_.DefaultWorkSize(*param.Output()->ClImage());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];
  auto input = param.Input()->ClImage()->GetCLImage();
  auto filter = param.Filter()->ClImage()->GetCLImage();
  auto new_scale = param.NewScale()->ClImage()->GetCLImage();
  auto new_bias = param.NewBias()->ClImage()->GetCLImage();
  auto output = param.Output()->ClImage()->GetCLImage();
  int stride = param.Strides()[0];
  int offset = param.Offset();
  int input_c = reinterpret_cast<framework::CLImageConverterFolder *>(
                    param.Input()->ClImage()->Converter())
                    ->GetCBlock();
  int dilation = param.Dilations()[0];
  int input_width = param.Input()->ClImage()->dims()[3];
  int input_height = param.Input()->ClImage()->dims()[2];
  int output_width = param.Output()->ClImage()->dims()[3];
  int output_height = param.Output()->ClImage()->dims()[2];

  cl_int status;

  status = clSetKernelArg(kernel, 0, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 1, sizeof(int), &w);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 2, sizeof(int), &nh);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &filter);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &new_scale);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &new_bias);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 8, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 9, sizeof(int), &offset);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 10, sizeof(int), &input_c);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 11, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 12, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 13, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 14, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 15, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}
template class DWConvBNReluKernelGpu<float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
