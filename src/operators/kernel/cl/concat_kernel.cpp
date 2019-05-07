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

#ifdef CONCAT_OP

#include "operators/kernel/concat_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConcatKernelGpu<float>::Init(ConcatParam *param) {
  CLImage *const out_cl_image = param->Out()->ClImage();

  if (!out_cl_image->isInit()) {
    out_cl_image->InitEmptyImage(this->cl_helper_.CLContext(),
                                 this->cl_helper_.CLCommandQueue(),
                                 out_cl_image->dims());
  }
  if (out_cl_image->dims().size() < 4) {
    this->cl_helper_.AddKernel("concatByH", "concat_kernel.cl");
  } else if (out_cl_image->dims().size() == 4) {
    this->cl_helper_.AddKernel("concatByC0", "concat_kernel.cl");
    this->cl_helper_.AddKernel("concatByC", "concat_kernel.cl");
  }
  return true;
}

template <>
void ConcatKernelGpu<float>::Compute(const ConcatParam &param) {
  DLOG << "yangfei50";
  DLOG << param.Out()->ClImage()->dims();
  if (param.Out()->ClImage()->dims().size() < 4) {
    auto kernel = this->cl_helper_.KernelAt(0);
    auto inputs = param.Inputs();
    auto *output_image = param.Out()->ClImage()->GetCLImage();
    int out_W = 0;
    if (param.Out()->ClImage()->dims().size() == 3) {
      out_W = param.Out()->ClImage()->dims()[2];
    } else if (param.Out()->ClImage()->dims().size() == 2) {
      out_W = param.Out()->ClImage()->dims()[1];
    }
    int out_H_Start = 0;
    for (int i = 0; i < inputs.size(); i++) {
      auto input_image = inputs[i]->ClImage()->GetCLImage();
      auto default_work_size =
          this->cl_helper_.DefaultWorkSize(*inputs[i]->ClImage());
      cl_int status;
      status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 2, sizeof(int), &out_W);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 3, sizeof(int), &out_H_Start);
      CL_CHECK_ERRORS(status);
      status = clEnqueueNDRangeKernel(
          this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
          NULL, default_work_size.data(), NULL, 0, NULL, NULL);
      CL_CHECK_ERRORS(status);
      if (param.Out()->ClImage()->dims().size() == 3) {
        out_H_Start += inputs[i]->ClImage()->dims()[1];
      } else if (param.Out()->ClImage()->dims().size() == 2) {
        out_H_Start += inputs[i]->ClImage()->dims()[0];
      }
    }
  } else {
    auto kernel0 = this->cl_helper_.KernelAt(0);
    auto kernel1 = this->cl_helper_.KernelAt(1);
    auto inputs = param.Inputs();
    auto *output_image = param.Out()->ClImage()->GetCLImage();

    int out_C_Start = 0;
    auto input_image = inputs[0]->ClImage()->GetCLImage();
    auto default_work_size =
        this->cl_helper_.DefaultWorkSize(*inputs[0]->ClImage());
    int out_W = param.Out()->ClImage()->dims()[3];
    cl_int status;
    status = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &input_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel0, 1, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel0, 2, sizeof(int), &out_W);
    CL_CHECK_ERRORS(status);
    status = clEnqueueNDRangeKernel(
        this->cl_helper_.CLCommandQueue(), kernel0, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
    out_C_Start += inputs[0]->ClImage()->dims()[1];
    for (int i = 1; i < inputs.size(); i++) {
      auto input_image1 = inputs[i - 1]->ClImage()->GetCLImage();
      auto input_image2 = inputs[i]->ClImage()->GetCLImage();
      default_work_size =
          this->cl_helper_.DefaultWorkSize(*inputs[i]->ClImage());
      int out_C = param.Out()->ClImage()->dims()[1];
      int out_H = param.Out()->ClImage()->dims()[2];
      int in_W = inputs[i]->ClImage()->dims()[3];
      int in_H = inputs[i]->ClImage()->dims()[2];
      int in_C1 = inputs[i - 1]->ClImage()->dims()[1];
      int in_C2 = inputs[i]->ClImage()->dims()[1];
      DLOG << "第" << i << "个";
      DLOG << "out_C=" << out_C;
      DLOG << "out_H=" << out_H;
      DLOG << "in_W=" << in_W;
      DLOG << "in_H=" << in_H;
      DLOG << "in_C1=" << in_C1;
      DLOG << "in_C2=" << in_C2;
      DLOG << "out_C_Start = " << out_C_Start;
      status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &input_image1);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 1, sizeof(cl_mem), &input_image2);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 2, sizeof(cl_mem), &output_image);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 3, sizeof(int), &out_C);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 4, sizeof(int), &out_H);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 5, sizeof(int), &out_W);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 6, sizeof(int), &out_C_Start);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 7, sizeof(int), &in_W);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 8, sizeof(int), &in_H);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 9, sizeof(int), &in_C1);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel1, 10, sizeof(int), &in_C2);
      CL_CHECK_ERRORS(status);

      status = clEnqueueNDRangeKernel(
          this->cl_helper_.CLCommandQueue(), kernel1, default_work_size.size(),
          NULL, default_work_size.data(), NULL, 0, NULL, NULL);
      CL_CHECK_ERRORS(status);

      out_C_Start += inputs[i]->ClImage()->dims()[1];
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
