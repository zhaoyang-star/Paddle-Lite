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

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <vector>
#ifdef PADDLE_MOBILE_CL
#include "cl/cl_image.h"
#include "framework/cl/cl_helper.h"
#include "framework/cl/cl_tensor.h"
#endif
#include "common/enforce.h"
#include "common/types.h"

#include "framework/data_layout.h"
#include "framework/lod_tensor.h"
#include "framework/tensor.h"
#include "memory/t_malloc.h"

namespace paddle_mobile {
namespace framework {

class TensorWrapper {
 public:
#ifdef PADDLE_MOBILE_CL
  TensorWrapper() : holder_cpu_(new LoDTensor()), holder_gpu_(new CLImage()) {}
#else
  TensorWrapper() : holder_cpu_(new LoDTensor()) {}
#endif

  int GetMemType() const {
    if (holder_cpu_) {
      return MEM_CPU;
    }
#ifdef PADDLE_MOBILE_CL

    else if (holder_gpu_) {
      return MEM_GPU;
    }
#endif
    else {

      // todo ---> 这里应该知道是什么op了.  内存应该预先处理一次
      //  PADDLE_MOBILE_ENFORCE(false, "Mem get but not init!");
      return MEM_UNKNOWN;
    }
  }

#ifdef PADDLE_MOBILE_CL

  framework::CLImage *MuteClImage() const { return this->GetGpu(); }
#endif
  framework::LoDTensor *MuteLodTensor() const { return this->GetCpu(); }

#ifdef PADDLE_MOBILE_CL
  CLImage *InnerCLImage() {
    if (this->GetMemType() == MEM_GPU) {
      return this->GetGpu();
    } else {
      // conver cpu mem to gpu
      // cast gpu to cpu
      const LoDTensor *input = this->GetCpu();
      const float *input_data = input->data<float>();
      CLImage *output = this->GetGpu();
      cl_context context = CLEngine::Instance()->getContext();
      cl_command_queue command_queue =
          CLEngine::Instance()->getClCommandQueue();
      output->InitEmptyImage(context, command_queue, output->dims());
      this->cl_helper_.AddKernel("feed", "feed_kernel.cl");
      cl_mem output_image = output->GetCLImage();
      const int out_C = output->dims()[1];
      const int out_H = output->dims()[2];
      const int out_W = output->dims()[3];
      const int Stride2 = out_C * out_H * out_W;
      const int Stride1 = out_H * out_W;
      const int Stride0 = out_W;
      framework::CLTensor input_cl_tensor(this->cl_helper_.CLContext(),
                                          this->cl_helper_.CLCommandQueue());
      input_cl_tensor.Resize(input->dims());
      cl_mem inputBuffer = input_cl_tensor.mutable_with_data<float>(input_data);
      auto kernel = this->cl_helper_.KernelAt(0);
      auto default_work_size = this->cl_helper_.DefaultWorkSize(*(output));
      cl_int status;
      status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 4, sizeof(cl_int), &out_C);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 5, sizeof(cl_int), &Stride0);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 6, sizeof(cl_int), &Stride1);
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 7, sizeof(cl_int), &Stride2);
      CL_CHECK_ERRORS(status);
      status = clEnqueueNDRangeKernel(
          this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
          NULL, default_work_size.data(), NULL, 0, NULL, NULL);
      CL_CHECK_ERRORS(status);
      this->holder_gpu_.reset();
      return output;
    }
  }
#endif
  framework::LoDTensor *InnerLoDTensor() const {
    if (this->GetMemType() == MEM_UNKNOWN) {
      DLOG << "tensor wrapper got MEM_UNKNOWN";
      return this->MuteLodTensor();
    } else if (this->GetMemType() == MEM_CPU) {
      return this->GetCpu();
    }
#ifdef PADDLE_MOBILE_CL
    else {
      const CLImage *pClImage = this->GetGpu();
      // cast gpu to cpu
      CLImage *image_p = const_cast<CLImage *>(pClImage);
      int width = image_p->ImageDims()[0];
      int height = image_p->ImageDims()[1];

      half_t *image_data = new half_t[height * width * 4];
      cl_int err;
      cl_mem image = image_p->GetCLImage();
      size_t origin[3] = {0, 0, 0};
      size_t region[3] = {static_cast<size_t>(width),
                          static_cast<size_t>(height), 1};
      err = clEnqueueReadImage(image_p->CommandQueue(), image, CL_TRUE, origin,
                               region, 0, 0, image_data, 0, NULL, NULL);
      CL_CHECK_ERRORS(err);

      LoDTensor *pTensor = this->GetCpu();
      pTensor->Resize(image_p->dims());
      auto converter = image_p->Converter();
      converter->ImageToNCHW(image_data, pTensor->data<float>(),
                             image_p->ImageDims(), image_p->dims());

      delete[](image_data);
      return pTensor;
    }
#else
    return NULL;
#endif
  }

 private:
  inline LoDTensor *GetCpu() const {
    PADDLE_MOBILE_ENFORCE(holder_cpu_ != nullptr, "holder_cpu_ is not init");

    return holder_cpu_.get();
  }
#ifdef PADDLE_MOBILE_CL

  inline CLImage *GetGpu() const {
    PADDLE_MOBILE_ENFORCE(holder_gpu_ != nullptr, "holder_cpu_ is not init");
    return holder_gpu_.get();
  }
#endif

  std::shared_ptr<LoDTensor> holder_cpu_;

#ifdef PADDLE_MOBILE_CL
  std::shared_ptr<CLImage> holder_gpu_;
  CLHelper cl_helper_;
#endif
};
using TensorWrapperArray = std::vector<TensorWrapper>;

/*template class PlaceholderImpl<LoDTensor>;
template class PlaceholderImpl<CLImage>;*/

}  // namespace framework
}  // namespace paddle_mobile
/* template <typename Type, typename RequestDeviceType>
 Type *getInner() {
   //    const Type *currentMem = this->Get<Type>();
   //    return const_cast<Type *>(currentMem);
   // 当前参数类型代表当前的kernel类型,
   const Type *currentMem = this->Get<Type>();
   if (std::is_same<GPU_CL, RequestDeviceType>::value &&
       this->GetRunTimeType() ==  MEM_GPU) {
     // gpu kernel gpu mem
     return const_cast<Type *>(currentMem);

   } else if (std::is_same< RequestDeviceType>::value &&
              this->GetRunTimeType() ==  MEM_CPU) {
     // cpu cpu mem
     return const_cast<Type *>(currentMem);

   } else if (std::is_same< RequestDeviceType>::value &&
              this->GetRunTimeType() ==  MEM_GPU) {
     if (mem_type_ ==  MEM_GPU) {
       const CLImage *pClImage = this->Get<CLImage>();

     } else if (mem_type_ ==  MEM_CPU) {
       const LoDTensor *pLoDTensor = this->Get<LoDTensor>();

     } else {
       return nullptr;
     }

     const CLImage *pClImage = this->Get<CLImage>();

     // cast gpu to cpu
     CLImage *image_p = const_cast<CLImage *>(pClImage);
     int width = image_p->ImageDims()[0];
     int height = image_p->ImageDims()[1];

     half_t *image_data = new half_t[height * width * 4];
     cl_int err;
     cl_mem image = image_p->GetCLImage();
     size_t origin[3] = {0, 0, 0};
     size_t region[3] = {static_cast<size_t>(width),
                         static_cast<size_t>(height), 1};
     err = clEnqueueReadImage(image_p->CommandQueue(), image, CL_TRUE, origin,
                              region, 0, 0, image_data, 0, NULL, NULL);
     CL_CHECK_ERRORS(err);

     LoDTensor *pTensor = this->GetMutableCPU<LoDTensor>();
     pTensor->Resize(image_p->dims());
     //      float *tensor_data = new float[image_p->numel()];
     auto converter = image_p->Converter();
     converter->ImageToNCHW(image_data, pTensor->data<float>(),
                            image_p->ImageDims(), image_p->dims());
     //           int stride = image_p->numel() / 20;
     //           stride = stride > 0 ? stride : 1;

     //      printer << " dims: " << image_p->dims() << "\n";
     //      for (int i = 0; i < image_p->numel(); i += stride) {
     //        printer << tensor_data[i] << " ";
     //      }

     //      delete[](tensor_data);
     delete[](image_data);

     // reinterpret_cast to suite template trash
     return reinterpret_cast<Type *>(pTensor);

   } else if (std::is_same<GPU_CL, RequestDeviceType>::value &&
              this->GetMemType() ==  MEM_CPU) {
     // cast gpu to cpu
     const LoDTensor *input = this->Get<LoDTensor>();
     const float *input_data = input->data<float>();

     CLImage *output = this->GetMutableGPU<CLImage>();

     cl_context context = CLEngine::Instance()->getContext();
     cl_command_queue command_queue =
         CLEngine::Instance()->getClCommandQueue();
     output->InitEmptyImage(context, command_queue, output->dims());
     this->cl_helper_.AddKernel("feed", "feed_kernel.cl");

     cl_mem output_image = output->GetCLImage();
     const int out_C = output->dims()[1];
     const int out_H = output->dims()[2];
     const int out_W = output->dims()[3];
     const int Stride2 = out_C * out_H * out_W;
     const int Stride1 = out_H * out_W;
     const int Stride0 = out_W;

     framework::CLTensor input_cl_tensor(this->cl_helper_.CLContext(),
                                         this->cl_helper_.CLCommandQueue());
     input_cl_tensor.Resize(input->dims());
     cl_mem inputBuffer =
 input_cl_tensor.mutable_with_data<float>(input_data);

     auto kernel = this->cl_helper_.KernelAt(0);

     auto default_work_size = this->cl_helper_.DefaultWorkSize(*(output));

     cl_int status;

     status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
     CL_CHECK_ERRORS(status);
     status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
     CL_CHECK_ERRORS(status);
     status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
     CL_CHECK_ERRORS(status);
     status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
     CL_CHECK_ERRORS(status);
     status = clSetKernelArg(kernel, 4, sizeof(cl_int), &out_C);
     CL_CHECK_ERRORS(status);
     status = clSetKernelArg(kernel, 5, sizeof(cl_int), &Stride0);
     CL_CHECK_ERRORS(status);
     status = clSetKernelArg(kernel, 6, sizeof(cl_int), &Stride1);
     CL_CHECK_ERRORS(status);
     status = clSetKernelArg(kernel, 7, sizeof(cl_int), &Stride2);
     CL_CHECK_ERRORS(status);

     status = clEnqueueNDRangeKernel(
         this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
         NULL, default_work_size.data(), NULL, 0, NULL, NULL);

     CL_CHECK_ERRORS(status);

     return reinterpret_cast<Type *>(output);

   } else {
     PADDLE_MOBILE_ENFORCE(false, "undefined mem get via op params");
   }
 }*/
