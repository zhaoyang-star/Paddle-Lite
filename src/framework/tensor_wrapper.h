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

#include "common/enforce.h"
#include "framework/data_layout.h"
#include "framework/tensor_base.h"
#include "cl/cl_image.h"
#include "framework/lod_tensor.h"
#include "memory/t_malloc.h"

namespace paddle_mobile {
namespace framework {
/**
 * 包装 各种数据类型, 以便自由在读取数据时转换成相应的数据类型.
 * 设计初心:  初始时可以是任意一种内存形式, 取的时候可以根据需要去进行转换.
 *
 * 如果没初始化, 以第一次初始化为准
 *
 */
class TensorWrapper {
 public:
  //  // This is the type we obtained in variable.
  //  typedef framework::CLImage gtype;
  //  // This type will be the parent class type
  //  // or the same type.
  //  typedef framework::CLImage rtype;

  MemType GetMemType() const { return mem_type_; }

  void SetMemType(paddle_mobile::MemType &memType) { mem_type_ = memType; }

  bool IsInitialized() const { return holder_cpu_ != nullptr; }

  framework::CLImage *MuteClImage() {
    mem_type_ = ComputeGPU;
    Clear();
    return this->GetMutableGPU<framework::CLImage>();
  }

  framework::LoDTensor *MuteLodTensor() {
    mem_type_ = ComputeCPU;
    Clear();
    return this->GetMutableCPU<framework::LoDTensor>();
  }

  template <typename T>
  bool IsType() const {
    return holder_cpu_ != nullptr && holder_cpu_->Type() == typeid(T);
  }

  void Clear() {
    holder_cpu_.reset();
    holder_gpu_.reset();
  }

  /*
    template <typename T, typename RequestDeviceType>
    T *template getInner<RType,Dtype>() {
      // 当前参数类型代表当前的kernel类型,
      if (std::is_same<GPU_CL, RequestDeviceType>::value &&
          this->GetMemType() == ComputeGPU) {
        // gpu kernel gpu mem
        return this->Get<CLImage>();

      } else if (std::is_same<CPU, RequestDeviceType>::value) {
        return this->Get<LoDTensor>();

      } else {
        PADDLE_MOBILE_ENFORCE(false, "undefined mem get via op params");
      }
  //
  //    //
  //    //    if(std::is_same<GPU_CL, RequestDeviceType>::value &&
  //    //    this->GetMemType()== ComputeGPU){
  //    //
  //    //    }
  //    return nullptr;
    }*/

  template <typename T, typename RequestDeviceType>
  T *getInner() const {
   /* // 当前参数类型代表当前的kernel类型,
    const T *currentMem = this->Get<T>();
    if (std::is_same<GPU_CL, RequestDeviceType>::value &&
        this->GetMemType() == ComputeGPU) {
      // gpu kernel gpu mem
      return const_cast<T *>(currentMem);

    } else if (std::is_same<CPU, RequestDeviceType>::value &&
               this->GetMemType() == ComputeCPU) {
      // cpu cpu mem
      return const_cast<T *>(currentMem);

    } else if (std::is_same<CPU, RequestDeviceType>::value &&
               this->GetMemType() == ComputeGPU) {
      // cast cpu to cpu

      CLImage *image_p = const_cast<CLImage *>(currentMem);
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

      float *tensor_data = new float[image_p->numel()];
      auto converter = image_p->Converter();
      converter->ImageToNCHW(image_data, tensor_data, image_p->ImageDims(),
                             image_p->dims());
      int stride = image_p->numel() / 20;
      stride = stride > 0 ? stride : 1;

      //      printer << " dims: " << image_p->dims() << "\n";
      //      for (int i = 0; i < image_p->numel(); i += stride) {
      //        printer << tensor_data[i] << " ";
      //      }

      delete[](tensor_data);
      delete[](image_data);

      return nullptr;

    } else if (std::is_same<GPU_CL, RequestDeviceType>::value &&
               this->GetMemType() == ComputeCPU) {
      // cast gpu to cpu
      return const_cast<T *>(currentMem);

    } else {
      PADDLE_MOBILE_ENFORCE(false, "undefined mem get via op params");
    }

    return nullptr;*/
  }

 private:
  template <typename T>
  const T *Get() const {
    if (mem_type_ == MemType::ComputeCPU) {
      return static_cast<const T *>(holder_cpu_->Ptr());
    } else if (mem_type_ == MemType::ComputeGPU) {
      return static_cast<const T *>(holder_gpu_->Ptr());
    } else {
      PADDLE_MOBILE_ENFORCE(false, "not support memtype pleae impl in Memtype");
    }
  }

  template <typename T>
  T *GetMutableCPU() {
    if (!IsType<T>()) {
      holder_cpu_.reset(new PlaceholderImpl<T>(new T()));
    }
    return static_cast<T *>(holder_cpu_->Ptr());
  }

  template <typename T>
  T *GetMutableGPU() {
    if (!IsType<T>()) {
      holder_cpu_.reset(new PlaceholderImpl<T>(new T()));
    }
    return static_cast<T *>(holder_cpu_->Ptr());
  }

  std::type_index Type() const { return holder_cpu_->Type(); }

  struct Placeholder {
    Placeholder() = default;
    virtual ~Placeholder() = default;

    virtual const std::type_info &Type() const = 0;
    virtual void *Ptr() const = 0;
  };

  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    explicit PlaceholderImpl(T *ptr) : ptr_(ptr), type_(typeid(T)) {}
    virtual const std::type_info &Type() const { return type_; }
    virtual void *Ptr() const override {
      return static_cast<void *>(ptr_.get());
    }
    std::unique_ptr<T> ptr_;
    const std::type_info &type_;
  };

  std::unique_ptr<Placeholder> holder_cpu_;
  std::unique_ptr<Placeholder> holder_gpu_;
  // holoder others

  paddle_mobile::MemType mem_type_;
};

}  // namespace framework
}  // namespace paddle_mobile
