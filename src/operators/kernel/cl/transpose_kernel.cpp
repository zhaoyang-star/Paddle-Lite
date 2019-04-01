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
#ifdef TRANSPOSE_OP

#include "operators/kernel/transpose_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool TransposeKernelGpu<float>::Init(TransposeParam *param) {
  if (param->Out()->InnerCLImage()->dims().size() == 4) {
    this->cl_helper_.AddKernel("transpose_4d", "transpose_kernel.cl");
  } else if (param->Out()->InnerCLImage()->dims().size() < 4) {
    this->cl_helper_.AddKernel("transpose", "transpose_kernel.cl");
  }
  return true;
}

template <>
void TransposeKernelGpu<float>::Compute(const TransposeParam &param) {
  if (param.Out()->InnerCLImage()->dims().size() == 4) {
    auto kernel = this->cl_helper_.KernelAt(0);
    auto default_work_size =
        this->cl_helper_.DefaultWorkSize(*param.Out()->InnerCLImage());
    int out_C = param.Out()->InnerCLImage()->dims()[1];
    int out_H = param.Out()->InnerCLImage()->dims()[2];
    int out_W = param.Out()->InnerCLImage()->dims()[3];
    int in_W = param.InputX()->InnerCLImage()->dims()[3];
    auto output_image = param.Out()->InnerCLImage()->GetCLImage();
    auto input_image = param.InputX()->InnerCLImage()->GetCLImage();
    DLOG << "out_C=" << out_C;
    DLOG << "out_H=" << out_H;
    DLOG << "out_W=" << out_W;
    DLOG << "in_C=" << in_W;
    DLOG << "default_work_size=" << default_work_size;
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(int), &out_C);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 3, sizeof(int), &out_H);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 4, sizeof(int), &out_W);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 5, sizeof(int), &in_W);
    CL_CHECK_ERRORS(status);
    status = clEnqueueNDRangeKernel(
        this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  } else if (param.Out()->InnerCLImage()->dims().size() == 3) {
    auto kernel = this->cl_helper_.KernelAt(0);
    auto default_work_size =
        this->cl_helper_.DefaultWorkSize(*param.Out()->InnerCLImage());
    int out_C = param.Out()->InnerCLImage()->dims()[0];
    int out_H = param.Out()->InnerCLImage()->dims()[1];
    int out_W = param.Out()->InnerCLImage()->dims()[2];
    int in_W = param.InputX()->InnerCLImage()->dims()[2];
    auto output_image = param.Out()->InnerCLImage()->GetCLImage();
    auto input_image = param.InputX()->InnerCLImage()->GetCLImage();
    DLOG << "out_C=" << out_C;
    DLOG << "out_H=" << out_H;
    DLOG << "out_W=" << out_W;
    DLOG << "in_C=" << in_W;
    DLOG << "default_work_size=" << default_work_size;
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(int), &out_C);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 3, sizeof(int), &out_H);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 4, sizeof(int), &out_W);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 5, sizeof(int), &in_W);
    CL_CHECK_ERRORS(status);
    status = clEnqueueNDRangeKernel(
        this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);

  } else if (param.Out()->InnerCLImage()->dims().size() == 2) {
    auto kernel = this->cl_helper_.KernelAt(0);
    auto default_work_size =
        this->cl_helper_.DefaultWorkSize(*param.Out()->InnerCLImage());
    int out_C = 1;
    int out_H = param.Out()->InnerCLImage()->dims()[0];
    int out_W = param.Out()->InnerCLImage()->dims()[1];
    int in_W = param.InputX()->InnerCLImage()->dims()[1];
    auto output_image = param.Out()->InnerCLImage()->GetCLImage();
    auto input_image = param.InputX()->InnerCLImage()->GetCLImage();
    DLOG << "out_C=" << out_C;
    DLOG << "out_H=" << out_H;
    DLOG << "out_W=" << out_W;
    DLOG << "in_C=" << in_W;
    DLOG << "default_work_size=" << default_work_size;
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(int), &out_C);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 3, sizeof(int), &out_H);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 4, sizeof(int), &out_W);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 5, sizeof(int), &in_W);
    CL_CHECK_ERRORS(status);
    status = clEnqueueNDRangeKernel(
        this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
