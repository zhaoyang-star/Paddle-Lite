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

#include "framework/operator.h"
#include <memory>
#include "operators/op_param.h"
namespace paddle_mobile {
namespace framework {

vector<string> OperatorBase::GetOutKeys() const {
  auto it = op_input_output_key.find(type_);
  if (it == op_input_output_key.end()) {
    DLOG << type_ << " has no outputs";
    return {};
  }
  return it->second.second;
}

vector<string> OperatorBase::GetInputKeys() const {
  auto it = op_input_output_key.find(type_);
  if (it == op_input_output_key.end()) {
    DLOG << type_ << " has no inputs";
    return {};
  }
  return it->second.first;
}

OperatorBase::OperatorBase(const std::string &type,
                           const VariableNameMap &inputs,
                           const VariableNameMap &outputs,
                           const AttributeMap &attrs, framework::Scope *scope)
    : type_(type),
      inputs_(inputs),
      outputs_(outputs),
      attrs_(attrs),
      scope_(scope)

{
  // DLOG << "construtor of OperatorBase !";
  CheckAllInputOutputSet();
}

void OperatorBase::CheckAllInputOutputSet() const {}

void OperatorBase::Run() {
#ifdef PADDLE_MOBILE_DEBUG
  DLOG << "-------------" << type_ << "--------------------------";
#endif
  RunImpl(expected_kernel_type_);
#ifdef PADDLE_MOBILE_DEBUG
  vector<string> input_keys = GetInputKeys();
  for (const auto key : input_keys) {
    auto var_vec_in = inputs_.at(key);
    for (int i = 0; i < var_vec_in.size(); ++i) {
      auto var = this->scope_->FindVar(var_vec_in[i]);
      if (var->IsInitialized() &&
          var->template IsType<framework::MobileTensor>()) {
        MobileTensor *tensor_w =
            const_cast<MobileTensor *>(var->template Get<MobileTensor>());
        if (tensor_w) {
          auto *tensor = tensor_w->LodTensor();
          if (tensor) DLOG << type_ << " input- " << key << "=" << *tensor;
        }

#ifdef PADDLE_MOBILE_FPGA
        DLOG << var_vec_in[i];
#endif
      }
    }
  }
  for (const auto key : GetOutKeys()) {
    auto var_vec_out = outputs_.at(key);
    for (int i = 0; i < var_vec_out.size(); ++i) {
      auto var = scope_->FindVar(var_vec_out[i]);
      if (var->IsInitialized() &&
          var->template IsType<framework::MobileTensor>()) {
        MobileTensor *tensor_w =
            const_cast<MobileTensor *>(var->template Get<MobileTensor>());
        if (tensor_w) {
          auto *tensor = tensor_w->LodTensor();
          if (tensor) DLOG << type_ << " output- " << key << "=" << *tensor;
        }
#ifdef PADDLE_MOBILE_FPGA
        DLOG << var_vec_out[i];
#endif
      }
    }
  }
#endif
}

#ifdef PADDLE_MOBILE_FPGA
template <typename Dtype>
void OperatorBase<Dtype>::InsertTensors() {
  static int feed_num = 0;
  static int fetch_num = 0;
  if (type_ == "feed") {
    auto new_name = string("feed") + std::to_string(feed_num++);
    auto var = scope_->Var(new_name);
    var->template GetMutable<framework::LoDTensor>();
    inputs_.at("X") = {string(new_name)};
  } else if (type_ == "fetch") {
    auto new_name = string("fetch") + std::to_string(fetch_num++);
    auto var = scope_->Var(new_name);
    var->template GetMutable<framework::LoDTensor>();
    outputs_.at("Out") = {string(new_name)};
  }
}
#endif

// template class OperatorBase;

}  // namespace framework
}  // namespace paddle_mobile
