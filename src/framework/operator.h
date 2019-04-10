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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/enforce.h"
#include "common/type_define.h"
#include "common/types.h"
#include "common/variant.h"
#include "framework/attribute.h"
#include "framework/op_info.h"
#include "framework/op_kernel_type.h"
#include "framework/op_registry.h"
#include "framework/program/block_desc.h"
#include "framework/program/program-optimize/node.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/variable.h"
#ifdef PADDLE_MOBILE_CL
#include "framework/cl/cl_helper.h"
#include "framework/cl/cl_scope.h"
#endif

namespace paddle_mobile {
namespace framework {

template <typename T>
static T *GetVarValue(const std::string &key, const VariableNameMap &var_map,
                      const Scope &scope) {
  auto var_vec = var_map.at(key);
  if (!var_vec.empty()) {
    auto var = scope.FindVar(var_vec[0]);
    return var->GetMutable<T>();
  } else {
    return nullptr;
  }
}

class OperatorBase {
 public:
  OperatorBase(const std::string &type, const VariableNameMap &inputs,
               const VariableNameMap &outputs, const AttributeMap &attrs,
               framework::Scope *scope);
  virtual ~OperatorBase() {}

  virtual void Init() = 0;
  virtual void InferShape() const = 0;
  virtual void Run();
  virtual void RunImpl(int kernelType = TYPE_CPU) = 0;

  std::vector<std::string> GetOutKeys() const;
  std::vector<std::string> GetInputKeys() const;

  const VariableNameMap &Inputs() const { return inputs_; }
  const VariableNameMap &Outputs() const { return outputs_; }
  const std::string &Type() const { return type_; }
  const AttributeMap &Attrs() const { return attrs_; }

  void SetExpectedKernelRunningType(int expected_type) {
    expected_kernel_type_ = expected_type;
  }
  void ClearVariables(const std::vector<std::string> &var_names) const {
    if (this->scope_) {
      this->scope_->EraseVars(var_names);
    }
  }
#ifdef PADDLE_MOBILE_FPGA
  void InsertTensors();
#endif
  int expected_kernel_type_ = TYPE_CPU;

 protected:
  framework::Scope *scope_;
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;

 private:
  void CheckAllInputOutputSet() const;
};

template <typename P>
class OpKernelBase {
 public:
  OpKernelBase(){};

#ifdef PADDLE_MOBILE_CL
  virtual void InitCLHelper(CLScope *clScope) {
    cl_helper_ = CLHelper(clScope);
    // DLOG << "InitCLHelper end";
  }
#endif

  virtual void Compute(const P &para) = 0;
  virtual bool Init(P *para) { return true; }
  virtual ~OpKernelBase(){};

 protected:
#ifdef PADDLE_MOBILE_CL
  CLHelper cl_helper_;
#endif
};

template <typename T, typename ParamType>
class OperatorWithKernels : public OperatorBase {
 public:
  OperatorWithKernels(const std::string &type, const VariableNameMap &inputs,
                      const VariableNameMap &outputs, const AttributeMap &attrs,
                      framework::Scope *scope)

      : OperatorBase(type, inputs, outputs, attrs, scope),
        param_(inputs, outputs, attrs, scope) {}
  virtual void RunImpl(int kernelType) { DLOG << "op run impl @ ......."; }
  virtual void InferShape() const = 0;
  ParamType param_;
  void Init() {}

 protected:
};

class FusionOpMatcher {
 public:
  FusionOpMatcher() {}

  virtual std::string Type() = 0;

  virtual void FolderNodes(
      Node *node,
      std::vector<std::shared_ptr<framework::Node>> *removed_nodes) {
    node->Folder(node_.Depth(), Type(), {}, removed_nodes);
  }

  virtual Node &BeginNode() { return node_; }

  std::string BeginType() { return node_.Type(); }

  virtual std::vector<std::pair<int, std::string>> NeedCheck() { return {}; }

 protected:
  Node node_;
  std::string type_;
  std::shared_ptr<OpDesc> new_opdesc_;
};

#ifdef PADDLE_MOBILE_CPU
#else
#endif
// define operators
#define REGIST_KERNEL_TYPE(OpName, DeviceName, KernelNamePrifix) \
  KernelNamePrifix##DeviceName<T> kernel##DeviceName##_;

#define REGIST_INIT(DeviceName, OpParam) \
  kernel##DeviceName##_.Init(            \
      &(framework::OperatorWithKernels<T, OpParam>::param_));

#ifdef PADDLE_MOBILE_CPU
#define REGIST_KERNEL_CPU_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix) \
  REGIST_KERNEL_TYPE(OpName, Cpu, KernelNamePrifix);
#define REGIST_KERNEL_CPU(OpName) \
  REGIST_KERNEL_CPU_WITH_KERNEL_PREFIX(OpName, OpName##Kernel);

#define INIT_KERNEKS_CPU(OpName, OpParam) ;
#define REGIST_INIT_CPU(OpParam) REGIST_INIT(Cpu, OpParam)

#else
#define REGIST_KERNEL_CPU(OpName) ;
#define INIT_KERNEKS_CPU(OpName, OpParam) ;
#define REGIST_KERNEL_CPU_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix) ;
#define REGIST_INIT_CPU(OpParam) ;

#endif
#ifdef PADDLE_MOBILE_CL

#define REGIST_KERNEL_GPU_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix) \
  REGIST_KERNEL_TYPE(OpName, Gpu, KernelNamePrifix);

#define REGIST_KERNEL_GPU(OpName) \
  REGIST_KERNEL_GPU_WITH_KERNEL_PREFIX(OpName, OpName##Kernel);

#define INIT_KERNEKS_GPU(OpName, OpParam) \
  kernelGpu_.InitCLHelper(scope->GetCLScpoe());
#define REGIST_INIT_GPU(OpParam) REGIST_INIT(Gpu, OpParam)
#else
#define REGIST_KERNEL_GPU(OpName) ;
#define REGIST_KERNEL_GPU_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix) ;
#define INIT_KERNEKS_GPU(OpName, OpParam) ;
#define REGIST_INIT_GPU(OpParam) ;
#endif

#ifdef PADDLE_MOBILE_FPGA
#define REGIST_KERNEL_FPGA_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix) \
  REGIST_KERNEL_TYPE(OpName, Fpga, KernelNamePrifix);
#define REGIST_KERNEL_FPGA(OpName) \
  REGIST_KERNEL_FPGA_WITH_KERNEL_PREFIX(OpName, OpName##Kernel);
#define INIT_KERNEKS_GPU(OpName, OpParam) ;
#define REGIST_INIT_FPGA(OpParam) REGIST_INIT(Fpga, OpParam)

#else
#define REGIST_KERNEL_FPGA_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix) ;
#define REGIST_KERNEL_FPGA(OpName) ;
#define INIT_KERNEKS_FPGA(OpName, OpParam) ;
#define REGIST_INIT_FPGA(OpParam) ;

#endif

#define DECLARE_OPERATOR_WITH_PARAMS(OpName, OpParam, KernelNamePrifix)       \
  template <typename T>                                                       \
  class OpName##Op : public framework::OperatorWithKernels<T, OpParam> {      \
   public:                                                                    \
    OpName##Op(const std::string &type, const VariableNameMap &inputs,        \
               const VariableNameMap &outputs,                                \
               const framework::AttributeMap &attrs, framework::Scope *scope) \
        : framework::OperatorWithKernels<T, OpParam>(type, inputs, outputs,   \
                                                     attrs, scope) {          \
      INIT_KERNEKS_CPU(OpName, OpParam);                                      \
    }                                                                         \
    void Init() { REGIST_INIT_CPU(OpParam); }                                 \
    void RunImpl(int kernelType) {                                            \
      /* use pre defined kernel to run */                                     \
      printf("kernel cpu run \n");                                            \
      kernelCpu_.Compute(framework::OperatorWithKernels<T, OpParam>::param_); \
    }                                                                         \
                                                                              \
    void InferShape() const override;                                         \
    REGIST_KERNEL_CPU_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix);           \
  };

#ifdef PADDLE_MOBILE_CL
#define DECLARE_OPERATOR_MIXED_WITH_PARAMS(OpName, OpParam, KernelNamePrifix) \
  template <typename T>                                                       \
  class OpName##Op : public framework::OperatorWithKernels<T, OpParam> {      \
   public:                                                                    \
    OpName##Op(const std::string &type, const VariableNameMap &inputs,        \
               const VariableNameMap &outputs,                                \
               const framework::AttributeMap &attrs, framework::Scope *scope) \
        : framework::OperatorWithKernels<T, OpParam>(type, inputs, outputs,   \
                                                     attrs, scope) {          \
      INIT_KERNEKS_GPU(OpName, OpParam);                                      \
      INIT_KERNEKS_CPU(OpName, OpParam);                                      \
      INIT_KERNEKS_FPGA(OpName, OpParam);                                     \
    }                                                                         \
    void Init() {                                                             \
      if (framework::OperatorWithKernels<                                     \
              T, OpParam>::OperatorBase::expected_kernel_type_ == TYPE_CPU) { \
        REGIST_INIT_CPU(OpParam);                                             \
      } else if (framework::OperatorWithKernels<                              \
                     T, OpParam>::OperatorBase::expected_kernel_type_ ==      \
                 TYPE_GPU) {                                                  \
        REGIST_INIT_GPU(OpParam);                                             \
      }                                                                       \
                                                                              \
      /* REGIST_INIT_FPGA(OpParam); */                                        \
    }                                                                         \
    void RunImpl(int kernelType) {                                            \
      /* use pre defined kernel to run */                                     \
      if (kernelType == TYPE_CPU) {                                           \
        printf("kernel cpu run \n");                                          \
        kernelCpu_.Compute(                                                   \
            framework::OperatorWithKernels<T, OpParam>::param_);              \
      } else if (kernelType == TYPE_GPU) {                                    \
        printf("kernel gpu run \n");                                          \
        kernelGpu_.Compute(                                                   \
            framework::OperatorWithKernels<T, OpParam>::param_);              \
      }                                                                       \
    }                                                                         \
                                                                              \
    void InferShape() const override;                                         \
    REGIST_KERNEL_CPU_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix);           \
    REGIST_KERNEL_GPU_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix);           \
    REGIST_KERNEL_FPGA_WITH_KERNEL_PREFIX(OpName, KernelNamePrifix);          \
  };

#else
#define DECLARE_OPERATOR_MIXED_WITH_PARAMS(OpName, OpParam, KernelNamePrifix) \
  DECLARE_OPERATOR_WITH_PARAMS(OpName, OpParam, KernelNamePrifix)
#endif

#define DECLARE_OPERATOR(OpName) \
  DECLARE_OPERATOR_WITH_PARAMS(OpName, OpName##Param, OpName##Kernel)

#define DECLARE_OPERATOR_MIXED(OpName) \
  DECLARE_OPERATOR_MIXED_WITH_PARAMS(OpName, OpName##Param, OpName##Kernel)

#define DECLARE_KERNEL_WITHPARAMS(OpName, DeviceName, DeviceType, OpParam)     \
  template <typename T>                                                        \
  class OpName##Kernel##DeviceName : public framework::OpKernelBase<OpParam> { \
   public:                                                                     \
    bool Init(OpParam *param);                                                 \
    void Compute(const OpParam &param);                                        \
  };

// define kernels
#define DECLARE_KERNEL(OpName, DeviceName, DeviceType) \
  DECLARE_KERNEL_WITHPARAMS(OpName, DeviceName, DeviceType, OpName##Param);

#ifdef PADDLE_MOBILE_CPU
#define DECLARE_KERNEL_CPU(OpName) DECLARE_KERNEL(OpName, Cpu, CPU);
#else
#define DECLARE_KERNEL_CPU(OpName) ;
#endif

#ifdef PADDLE_MOBILE_CL
#define DECLARE_KERNEL_GPU(OpName) DECLARE_KERNEL(OpName, Gpu, GPU_CL);
#else
#define DECLARE_KERNEL_GPU(OpName) ;
#endif

#ifdef PADDLE_MOBILE_FPGA
#define DECLARE_KERNEL_FPGA(OpName) DECLARE_KERNEL(OpName, Fpga, FPGA);
#else
#define DECLARE_KERNEL_FPGA(OpName) ;
#endif

#define DECLARE_KERNEL_ALL(OpName) \
  DECLARE_KERNEL_CPU(OpName);      \
  DECLARE_KERNEL_GPU(OpName);      \
  DECLARE_KERNEL_FPGA(OpName);

#define DECLARE_KERNEL_CPU_WITH_PARAMS(OpName, OpParam) \
  DECLARE_KERNEL_WITHPARAMS(OpName, Cpu, CPU, OpParam);

#define DECLARE_KERNEL_GPU_WITH_PARAMS(OpName, OpParam) \
  DECLARE_KERNEL_WITHPARAMS(OpName, Gpu, GPU_CL, OpParam);

#define DECLARE_KERNEL_FPGA_WITH_PARAMS(OpName, OpParam) \
  DECLARE_KERNEL_WITHPARAMS(OpName, Fpga, FPGA, OpParam);

#define DECLARE_KERNEL_ALL_WITH_PARAMS(OpName, OpParam) \
  DECLARE_KERNEL_CPU_WITH_PARAMS(OpName, OpParam);      \
  DECLARE_KERNEL_GPU_WITH_PARAMS(OpName, OpParam);      \
  DECLARE_KERNEL_FPGA_WITH_PARAMS(OpName, OpParam);

}  // namespace framework
}  // namespace paddle_mobile
