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
#include "memory/t_malloc.h"

namespace paddle_mobile {
namespace framework {
/**
 * 包装 各种数据类型, 以便自由在读取数据时转换成相应的数据类型.
 *
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

  template <typename T>
  const T *Get() const {
    return static_cast<const T *>(holder_->Ptr());
  }

  bool IsInitialized() const { return holder_ != nullptr; }

  template <typename T>
  T *GetMutable() {
    if (!IsType<T>()) {
      holder_.reset(new PlaceholderImp<T>(new T()));
    }
    return static_cast<T *>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    return holder_ != nullptr && holder_->Type() == typeid(T);
  }

  void Clear() { holder_.reset(); }

  std::type_index Type() const { return holder_->Type(); }

/*
  template <typename T, typename RequestDeviceType>
  T *template getInner<RType>() {
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
  }
*/

  template <typename T>
  T *getInner() const {
    return nullptr;
  }
private:
  struct Placeholder {
    Placeholder() = default;
    virtual ~Placeholder() = default;

    virtual const std::type_info &Type() const = 0;
    virtual void *Ptr() const = 0;
  };

  template <typename T>
  struct PlaceholderImp : public Placeholder {
    explicit PlaceholderImp(T *ptr) : ptr_(ptr), type_(typeid(T)) {}
    virtual const std::type_info &Type() const { return type_; }
    virtual void *Ptr() const override {
      return static_cast<void *>(ptr_.get());
    }
    std::unique_ptr<T> ptr_;
    const std::type_info &type_;
  };

  std::unique_ptr<Placeholder> holder_;
  paddle_mobile::MemType mem_type_;
};

}  // namespace framework
}  // namespace paddle_mobile
