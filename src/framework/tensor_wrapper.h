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
#include "framework/image_converter.h"
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
  bool IsPersistable() { return is_persistable_; }
  void SetPersistable(bool is_persistable) {
    this->is_persistable_ = is_persistable;
  }
  int GetMemType() const {
    if (holder_cpu_ && mem_type == MEM_CPU) {
      return MEM_CPU;
    }
#ifdef PADDLE_MOBILE_CL

    else if (holder_gpu_ && mem_type == MEM_GPU) {
      return MEM_GPU;
    }
#endif
    else {
      // todo ---> 这里应该知道是什么op了.  内存应该预先处理一次
      PADDLE_MOBILE_ENFORCE(false, "Mem get but not init!");
      return MEM_UNKNOWN;
    }
  }

#ifdef PADDLE_MOBILE_CL

  framework::CLImage *MuteClImage() { return this->InnerCLImage(); }
#endif
  framework::LoDTensor *MuteLodTensor() { return this->InnerLoDTensor(); }

#ifdef PADDLE_MOBILE_CL
  CLImage *InnerCLImage() {
    if (this->GetMemType() == MEM_GPU) {
      return this->GetGpu();
    } else {
      DLOG << "----------begin-----  cpu ---> gpu----------------------";
      // conver cpu mem to gpu
      // cast gpu to cpu
      const LoDTensor *input = this->GetCpu();
      CLImage *output = this->GetGpu();
      CLHelper *helper = GpuRumtimeHelper::Instance()->GetClHelper();
      DLOG << "input->IsInitialized(): " << input->IsInitialized();
      if (input->IsInitialized()) {
        const float *input_data = input->data<float>();
        cl_context context = CLEngine::Instance()->getContext();
        cl_command_queue command_queue =
            CLEngine::Instance()->getClCommandQueue();

        const DDim &dims = output->dims();
        DLOG << "output->isInit():  " << output->isInit();
        DLOG << "IsPersistable()=======>  " << IsPersistable();
        if (!output->isInit() && IsPersistable()) {
          // do nothing until init ?
          output->SetTensorData(const_cast<float *>(input_data), input->dims());
        } else {
          //      this->GetClHelper().AddKernel("feed", "feed_kernel.cl");
          cl_mem output_image = output->GetCLImage();
          if (output_image == nullptr) {
            output->InitEmptyImage(context, command_queue, dims);
            output_image = output->GetCLImage();
          } else {
            output->SetTensorData(const_cast<float *>(input_data),
                                  input->dims());
          }
          size_t new_dims[] = {1, 1, 1, 1};
          for (int j = 0; j < dims.size(); ++j) {
            new_dims[4 - dims.size() + j] = dims[j];
          }

          size_t N, C, H, W;
          N = new_dims[0];
          C = new_dims[1];
          H = new_dims[2];
          W = new_dims[3];

          const int out_C = C;
          const int out_H = H;
          const int out_W = W;
          const int Stride2 = out_C * out_H * out_W;
          const int Stride1 = out_H * out_W;
          const int Stride0 = out_W;

          framework::CLTensor input_cl_tensor(helper->CLContext(),
                                              helper->CLCommandQueue());
          input_cl_tensor.Resize(input->dims());
          cl_mem inputBuffer =
              input_cl_tensor.mutable_with_data<float>(input_data);
          auto kernel = helper->KernelAt(0);
          auto default_work_size = helper->DefaultWorkSize(*(output));
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
              helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
              default_work_size.data(), NULL, 0, NULL, NULL);
          CL_CHECK_ERRORS(status);
        }

        mem_type = MEM_GPU;
      } else {
        output->Resize(input->dims());
        mem_type = MEM_GPU;
        //        output->InitEmptyImage(helper->CLContext(),helper->CLCommandQueue(),input->dims());
      }
      DLOG << "----------success---  cpu ---> gpu----------------------";

      return output;
    }
  }
#endif
  framework::LoDTensor *InnerLoDTensor() {
    if (this->GetMemType() == MEM_CPU) {
      return this->GetCpu();
    }
#ifdef PADDLE_MOBILE_CL
    else {

      CLImage *input_climage = this->GetGpu();

      LoDTensor *output_lodtensor = this->GetCpu();
      DLOG << "----------begin-----  gpu ---> cpu----------------------";

      DLOG << "input_climage->isInit():  " << input_climage->isInit();

      if (input_climage->isInit()) {
        // cast gpu to cpu
        //        CLImage *image_p = input_climage;
        int width = input_climage->ImageDims()[0];
        int height = input_climage->ImageDims()[1];

        half_t *image_data = new half_t[height * width * 4];
        cl_int err;
        cl_mem image = input_climage->GetCLImage();
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {static_cast<size_t>(width),
                            static_cast<size_t>(height), 1};
        err =
            clEnqueueReadImage(input_climage->CommandQueue(), image, CL_TRUE,
                               origin, region, 0, 0, image_data, 0, NULL, NULL);
        CL_CHECK_ERRORS(err);

        output_lodtensor->Resize(input_climage->dims());
        auto converter = input_climage->Converter();
        if (!output_lodtensor->IsInitialized()) {
          output_lodtensor->mutable_data<float>();
        }
        converter->ImageToNCHW(image_data, output_lodtensor->data<float>(),
                               input_climage->ImageDims(),
                               input_climage->dims());
        delete[](image_data);
      } else {
        const DDim &dims = input_climage->dims();
        output_lodtensor->ResizeSafe(dims);
      }

      mem_type = MEM_CPU;
      DLOG << "----------success---  gpu ---> cpu----------------------";

      return output_lodtensor;
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
  int mem_type = MEM_CPU;

  bool is_persistable_ = false;

#ifdef PADDLE_MOBILE_CL
  std::shared_ptr<CLImage> holder_gpu_;
//  CLHelper GetClHelper() const { return cl_helper_; }
//  CLHelper cl_helper_;
#endif
};
using TensorWrapperArray = std::vector<TensorWrapper>;

/*template class PlaceholderImpl<LoDTensor>;
template class PlaceholderImpl<CLImage>;*/

}  // namespace framework
}  // namespace paddle_mobile
