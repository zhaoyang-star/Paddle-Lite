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
  //  // This is the type we obtained in variable.
  //  typedef framework::CLImage gtype;
  //  // This type will be the parent class type
  //  // or the same type.
  //  typedef framework::CLImage rtype;

  int GetMemType() const {
    if (holder_cpu_->is_init_) {
      return MEM_CPU;
    }
#ifdef PADDLE_MOBILE_CL

    else if (holder_gpu_->is_init_) {
      return MEM_GPU;
    }
#endif
    else {
      PADDLE_MOBILE_ENFORCE(false, "Mem get but not init!");
      return MEM_UNKNOWN;
    }
  }

  //  void SetMemType(int memType) const { mem_type_ = memType; }

  bool IsInitialized() const { return holder_cpu_ != nullptr; }
#ifdef PADDLE_MOBILE_CL

  framework::CLImage *MuteClImage() {
    Clear();
    return this->GetMutableGPU();
  }
#endif
  framework::LoDTensor *MuteLodTensor() {
    Clear();
    return this->GetMutableCPU();
  }

  void Clear() {
    holder_cpu_.reset();
    holder_cpu_->is_init_ = false;
#ifdef PADDLE_MOBILE_CL
    holder_gpu_.reset();
    holder_cpu_->is_init_ = false;
#endif
  }
#ifdef PADDLE_MOBILE_CL
  CLImage *InnerCLImage() {
    if (this->GetMemType() == MEM_GPU) {
      return const_cast<CLImage *>(this->GetGpu());
    } else {
      // conver cpu mem to gpu
      // cast gpu to cpu
      const LoDTensor *input = this->GetCpu();
      const float *input_data = input->data<float>();
      CLImage *output = this->GetMutableGPU();
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
      this->holder_gpu_->is_init_ = false;
      return output;
    }
  }
#endif
  framework::LoDTensor *InnerLoDTensor() {
    if (this->GetMemType() == TYPE_CPU) {
      return const_cast<LoDTensor *>(this->GetCpu());
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

      LoDTensor *pTensor = this->GetMutableCPU();
      pTensor->Resize(image_p->dims());
      //      float *tensor_data = new float[image_p->numel()];
      auto converter = image_p->Converter();
      converter->ImageToNCHW(image_data, pTensor->data<float>(),
                             image_p->ImageDims(), image_p->dims());

      delete[](image_data);
      this->holder_cpu_->is_init_ = true;
      return pTensor;
    }
#else
    return NULL;
#endif
  }

 private:
  bool IsTypeCpu() const {
    return holder_cpu_ != nullptr && holder_cpu_->Type() == typeid(LoDTensor);
  }
  const inline LoDTensor *GetCpu() const {
    PADDLE_MOBILE_ENFORCE(holder_cpu_->is_init_, "holder_cpu_ is not init");

    return static_cast<const LoDTensor *>(holder_cpu_->Ptr());
  }
#ifdef PADDLE_MOBILE_CL

  const inline CLImage *GetGpu() {
    PADDLE_MOBILE_ENFORCE(holder_gpu_->is_init_, "holder_cpu_ is not init");
    return static_cast<const CLImage *>(holder_gpu_->Ptr());
  }
#endif

  LoDTensor *GetMutableCPU() {
    if (!IsTypeCpu()) {
      holder_cpu_.reset(new PlaceholderImpl<LoDTensor>(new LoDTensor()));
    }
    holder_cpu_->is_init_ = true;

    return static_cast<LoDTensor *>(holder_cpu_->Ptr());
  }

  class Placeholder {
   public:
    Placeholder() = default;
    virtual ~Placeholder() = default;

    virtual const std::type_info &Type() const = 0;
    virtual void *Ptr() const = 0;
  };

  template <typename T>
  class PlaceholderImpl : public Placeholder {
   public:
    PlaceholderImpl(T *ptr) : ptr_(ptr), type_(typeid(T)) {}
    virtual const std::type_info &Type() const { return type_; }
    virtual void *Ptr() const override {
      return static_cast<void *>(ptr_.get());
    }
    std::unique_ptr<T> ptr_;
    const std::type_info &type_;
    bool is_init_ = false;
  };

  std::unique_ptr<PlaceholderImpl<LoDTensor>> holder_cpu_;

#ifdef PADDLE_MOBILE_CL
  std::unique_ptr<PlaceholderImpl<CLImage>> holder_gpu_;
  CLHelper cl_helper_;

  bool IsTypeGpu() const {
    return holder_gpu_ != nullptr && holder_gpu_->Type() == typeid(CLImage);
  }
  CLImage *GetMutableGPU() {
    if (!IsTypeGpu()) {
      holder_gpu_.reset(new PlaceholderImpl<CLImage>(new CLImage()));
    }
    holder_gpu_->is_init_ = true;
    return static_cast<CLImage *>(holder_gpu_->Ptr());
  }

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
       this->GetRunTimeType() == TYPE_GPU) {
     // gpu kernel gpu mem
     return const_cast<Type *>(currentMem);

   } else if (std::is_same< RequestDeviceType>::value &&
              this->GetRunTimeType() == TYPE_CPU) {
     // cpu cpu mem
     return const_cast<Type *>(currentMem);

   } else if (std::is_same< RequestDeviceType>::value &&
              this->GetRunTimeType() == TYPE_GPU) {
     if (mem_type_ == TYPE_GPU) {
       const CLImage *pClImage = this->Get<CLImage>();

     } else if (mem_type_ == TYPE_CPU) {
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
              this->GetMemType() == TYPE_CPU) {
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
