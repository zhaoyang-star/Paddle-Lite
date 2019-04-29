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

#include "framework/executor.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "common/enforce.h"
#include "common/log.h"
#include "framework/context.h"
#include "framework/framework.pb-c.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/program-optimize/program_optimize.h"
#include "framework/program/program_desc.h"
#include "framework/program/var_desc.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/tensor_wrapper.h"
#include "memory/t_malloc.h"
#include "pass/memory_optimize.h"
#ifdef PADDLE_MOBILE_CL
#include "framework/cl/cl_image.h"
#endif

namespace paddle_mobile {
namespace framework {

#pragma mark - executor

template <typename T>
void Executor<T>::SetThreadNum(int threads) {
  set_global_num_threads(threads);
}

template <typename T>
Executor<T>::Executor(const Program<T> &program,
                      paddle_mobile::PaddleMobileConfigInternal config,
                      int batch_size, const bool use_optimize,
                      const bool lod_mode)
    : program_(program),
      batch_size_(batch_size),
      use_optimize_(use_optimize),
      lod_mode_(lod_mode),
      config_(config) {
  DLOG << "executor in lod mode: " << lod_mode_;

  Variable *variable_ptr = program_.scope->Var("batch_size");
  variable_ptr->SetValue<int>(batch_size);

  program_desc_ =
      use_optimize_ ? program_.optimizeProgram : program_.originProgram;
  PADDLE_MOBILE_ENFORCE(program_desc_ != nullptr,
                        "program_desc_ should not be nullptr");
#if !defined(PADDLE_MOBILE_FPGA) && !defined(PADDLE_MOBILE_FPGA_KD) && \
    !defined(PADDLE_MOBILE_CL)
  pass::MemoryOptPass()(program_desc_.get(), program_.scope.get());
#endif
  // resize feed and fetch list
  // should init feed and fetch variables before infer shape
  InitFeedFetchList();
  const auto &blocks = program_desc_->Blocks();
  std::shared_ptr<BlockDesc> block_desc = blocks[0];
  std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
  for (int j = 0; j < ops.size(); ++j) {
    std::shared_ptr<OpDesc> op_desc = ops[j];
    DLOG << "create op: " << op_desc->Type();

    auto op_handler = OpRegistry::CreateOp(
        op_desc->Type(), op_desc->GetInputs(), op_desc->GetOutputs(),
        op_desc->GetAttrMap(), program_.scope.get());

    for (auto iter = config.running_expected_map_.begin();
         iter != config.running_expected_map_.end(); iter++) {

      printf("当前的: %s   ------被查找的 %s \n ",op_desc->Type().c_str(),iter->first.c_str());

      if (op_desc->Type().find(iter->first.c_str()) != string::npos) {
        printf("命中!\n");
        op_handler->SetExpectedKernelRunningType(iter->second);
      }
    }

    // infer shape to reshape inputs and outputs before predict,
    // but for lod mode, it still need to infer shape in runtime
    if (!lod_mode) {
      op_handler->InferShape();
    }
    ops_of_block0_.push_back(op_handler);
  }
#ifdef PADDLE_MOBILE_FPGA_V2
  InitQuantMemory();
#endif
  if (program_.combined) {
    InitCombineMemory();
  } else {
    InitMemory();
  }

  int count = 0;
  for (auto &op_handler : ops_of_block0_) {
    DLOG << "Initialize op[" << count++ << "]: " << op_handler->Type();
    op_handler->Init();
  }
}

template <typename T>
void Executor<T>::InitFeedFetchList() {
  std::unordered_map<std::string, int> feed_indices, fetch_indices;
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &op_desc : block->Ops()) {
      if (op_desc->Type() == "feed") {
        std::string name = op_desc->Output("Out")[0];
        feed_indices[name] = op_desc->GetAttr("col").Get<int>();
      } else if (op_desc->Type() == "fetch") {
        std::string name = op_desc->Input("X")[0];
        fetch_indices[name] = op_desc->GetAttr("col").Get<int>();
      }
    }
  }
  feed_indices_.swap(feed_indices);
  fetch_indices_.swap(fetch_indices);

  auto *feed_var = program_.scope->Var("feed");
  auto *feed_list =
      feed_var->template GetMutable<framework::MobileTensorArray>();
  feed_list->resize(feed_indices_.size());

  auto *fetch_var = program_.scope->Var("fetch");
  auto *fetch_list =
      fetch_var->template GetMutable<framework::MobileTensorArray>();
  fetch_list->resize(fetch_indices_.size());
}

template <typename T>
static void LoadMemInternal(void **data, LoDTensor *tensor,
                            bool quant_uint8 = false) {
  char **data_buf = reinterpret_cast<char **>(data);
  int64_t size = tensor->numel();
  T *tensor_data = tensor->mutable_data<T>();
  if (quant_uint8) {
    // should be moved into operator init function
    float min_value;
    float max_value;
    memory::Copy(&min_value, *data_buf, sizeof(float));
    memory::Copy(&max_value, *data_buf + sizeof(float), sizeof(float));
    *data_buf += 2 * sizeof(float);
    const float factor = (max_value - min_value) / 255.0;
    const uint8_t *uint8_data = reinterpret_cast<uint8_t *>(*data_buf);
    for (int k = 0; k < size; ++k) {
      tensor_data[k] = uint8_data[k] * factor + min_value;
    }
    *data_buf += size * sizeof(uint8_t);
  } else {
    memory::Copy(tensor_data, *data_buf, size * sizeof(T));
    *data_buf += size * sizeof(T);
  }
}

template <typename T>
void Executor<T>::LoadMemory(void **data,
                             const std::shared_ptr<VarDesc> var_desc,
                             LoDTensor *tensor) {
  char **data_buf = reinterpret_cast<char **>(data);
  // version
  uint32_t version = *(reinterpret_cast<uint32_t *>(*data_buf));
  *data_buf += sizeof(uint32_t);
  // lod information
  // uint64_t lod_level = *(reinterpret_cast<uint64_t *>(*data_buf));
  uint64_t lod_level = 0;
  memory::Copy(&lod_level, *data_buf, sizeof(uint64_t));
  *data_buf += sizeof(uint64_t);

  auto *lod = tensor->mutable_lod();
  lod->resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size = *(reinterpret_cast<uint64_t *>(*data_buf));
    *data_buf += sizeof(uint64_t);
    std::vector<size_t> tmp_dim(size / sizeof(size_t));
    memory::Copy(tmp_dim.data(), *data_buf, size);
    (*lod)[i] = std::move(tmp_dim);
    *data_buf += size;
  }
  // tensor version
  uint32_t tensor_version = *(reinterpret_cast<uint32_t *>(*data_buf));
  *data_buf += sizeof(uint32_t);
  // tensor desc size
  int32_t tensor_desc_size = *(reinterpret_cast<int32_t *>(*data_buf));
  *data_buf += sizeof(int32_t);
  // skip tensor desc
  *data_buf += tensor_desc_size;

  const TensorDesc &tensor_desc = var_desc->Tensor_desc();
  tensor->Resize(make_ddim(tensor_desc.Dims()));
  // parse tensor from stream
  switch (tensor_desc.DataType()) {
    case VARTYPE_TYPE_FP32:
      LoadMemInternal<float>(reinterpret_cast<void **>(data_buf), tensor,
                             program_.quantification);
      break;
    case VARTYPE_TYPE_INT8:
      LoadMemInternal<int8_t>(reinterpret_cast<void **>(data_buf), tensor);
      break;
    case VARTYPE_TYPE_INT32:
      LoadMemInternal<int>(reinterpret_cast<void **>(data_buf), tensor);
      break;
    default:
      LOG(kLOG_ERROR) << "data type is not supported";
  }
}

template <typename T>
void Executor<T>::InitMemory() {
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::MobileTensorArray>();
          continue;
        }
        DLOG << "init persistable var: " << var_desc->Name();
        char *origin_data =
            ReadFileToBuff(program_.model_path + "/" + var_desc->Name());
        char *data = origin_data;
        auto tensor_w = var->template GetMutable<MobileTensor>();
        tensor_w->SetPersistable(true);
        LoadMemory(reinterpret_cast<void **>(&data), var_desc,
                   tensor_w->MuteLodTensor());
        delete[] origin_data;
      } else {
        DLOG << "init no persistable var: " << var_desc->Name();
        varInputMemory(var_desc, var);
      }
    }
  }
}

template <typename T>
void Executor<T>::InitCombineMemory() {
  char *origin_data = nullptr;
  bool self_alloc = false;
  if (program_.combined_params_buf && program_.combined_params_len) {
    origin_data = reinterpret_cast<char *>(
        const_cast<uint8_t *>(program_.combined_params_buf));
  } else {
    self_alloc = true;
    origin_data = ReadFileToBuff(program_.para_path);
  }
  PADDLE_MOBILE_ENFORCE(origin_data != nullptr, "data == nullptr");
  char *data = origin_data;
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::MobileTensorArray>();
          continue;
        }

        DLOG << " init combine memory persistable: " << var_desc->Name();
        auto tensor_wrapper = var->template GetMutable<MobileTensor>();
        //        LoDTensor *tensor = tensor_wrapper->MuteLodTensor();
        tensor_wrapper->SetPersistable(true);

        LoadMemory(reinterpret_cast<void **>(&data), var_desc,
                   tensor_wrapper->MuteLodTensor());
      } else {
        DLOG << " init combine memory no persistable: " << var_desc->Name();
        varInputMemory(var_desc, var);
      }
    }
  }
  if (self_alloc) {
    delete[] origin_data;
  }
  LOG(kLOG_INFO) << "init combine memory finish";
}

static void ClearNoPersistableTensorArray(const framework::ProgramDesc *program,
                                          framework::Scope *scope) {
  for (const auto &block : program->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      if (!var_desc->Persistable() &&
          var_desc->Type() == VARTYPE_TYPE_STEP_LOD_TENSOR_ARRAY) {
        auto var = scope->Var(var_desc->Name());
        auto array = var->template GetMutable<framework::MobileTensorArray >();
        array->resize(1);
      }
    }
  }
}

template <typename T>
void Executor<T>::InitNoPersistableMemory(const Tensor &input_tensor) {
  for (const auto &block : program_desc_->Blocks()) {
    for (const auto &var_desc : block->Vars()) {
      auto var = program_.scope->Var(var_desc->Name());
      auto tensor_w = var->template GetMutable<MobileTensor>();
      LoDTensor *const tensor = tensor_w->MuteLodTensor();
      if (var_desc->Persistable()) {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
          var->template GetMutable<framework::MobileTensorArray>();
          continue;
        }
      } else {
        if (var_desc->Type() == VARTYPE_TYPE_LOD_TENSOR) {
          DDim tensor_dim = tensor->dims();
          DDim new_dim =
              make_ddim({tensor_dim[0], tensor_dim[1], input_tensor.dims()[2],
                         input_tensor.dims()[3]});
          tensor->Resize(new_dim);
          tensor->template mutable_data<T>();
        } else {
          PADDLE_MOBILE_THROW_EXCEPTION("Unsupported var type `%d`",
                                        var_desc->Type());
        }
      }
    }
  }

  std::shared_ptr<LoDTensor> output = GetOutput("fetch");
  output->Resize(input_tensor.dims());
  output->mutable_data<T>();
}

template <typename T>
bool Executor<T>::varInputMemory(const std::shared_ptr<VarDesc> &var_desc,
                                 Variable *var) const {
#ifdef PADDLE_MOBILE_FPGA
  framework::LoDTensor *tensor = var->template GetMutable<LoDTensor>();
  tensor->init(type_id<float>().hash_code());
  return true;
#endif

  auto type = var_desc->Type();
  if (type == VARTYPE_TYPE_LOD_TENSOR) {
    auto data_type = var_desc->Tensor_desc().DataType();
    auto *tensor_w = var->template GetMutable<MobileTensor>();
    LoDTensor *tensor = tensor_w->MuteLodTensor();
  } else if (type == VARTYPE_TYPE_STEP_SCOPES) {
    std::vector<framework::Scope *> *step_scopes =
        var->template GetMutable<std::vector<framework::Scope *>>();
  } else if (type == VARTYPE_TYPE_STEP_LOD_TENSOR_ARRAY) {
    framework::MobileTensorArray *tensor_array =
        var->template GetMutable<framework::MobileTensorArray>();

  } else {
    PADDLE_MOBILE_THROW_EXCEPTION("got unhandled var type `%d`", type);
  }
  return true;
}

template <typename T>
PMStatus Executor<T>::Predict(
    const std::vector<std::pair<std::string, Tensor>> &inputs) {
  for (const auto &input : inputs) {
    SetInput(input.second, input.first);
  }
  return this->Predict();
}

template <typename T>
PMStatus Executor<T>::Predict(
    const std::vector<std::pair<std::string, LoDTensor>> &inputs) {
  for (const auto &input : inputs) {
    SetInput(input.second, input.first);
  }
  return this->Predict();
}

template <typename T>
std::vector<T> Executor<T>::Predict(const std::vector<T> &input,
                                    const std::vector<int64_t> &dims) {
  PADDLE_MOBILE_ENFORCE(feed_indices_.size() != 0,
                        "We don't know which tensor should be assign, since no"
                        "feed op found in this model");
  PADDLE_MOBILE_ENFORCE(fetch_indices_.size() != 0,
                        "We don't know which tensor should be fetch out, since"
                        "no fetch op found in this model");
  std::string input_name = feed_indices_.begin()->first;
  Tensor feed_tensor(input, make_ddim(dims));
  SetInput(feed_tensor, input_name);
  std::vector<T> output;
  if (this->Predict() == PMSuccess) {
    std::string output_name = fetch_indices_.begin()->first;
    const auto output_tensor = GetOutput(output_name);
    output.resize(output_tensor->numel());
    memcpy(output.data(), output_tensor->template data<T>(),
           output.size() * sizeof(T));
  }
  return output;
}

template <typename T>
void Executor<T>::SetInput(const Tensor &input, const std::string &var_name) {
  int index = 0;
  if (feed_indices_.find(var_name) != feed_indices_.end()) {
    index = feed_indices_.find(var_name)->second;
  }
  auto *feed_var = program_.scope->Var("feed");
  framework::LoDTensor &target =
      *feed_var->template GetMutable<framework::MobileTensorArray>()
           ->at(index)
           .MuteLodTensor();

  target.Resize(input.dims());
  target.ShareDataWith(input);
}

template <typename T>
void Executor<T>::SetInput(const LoDTensor &input,
                           const std::string &var_name) {
  int index = 0;
  if (feed_indices_.find(var_name) != feed_indices_.end()) {
    index = feed_indices_.find(var_name)->second;
  }
  auto *feed_var = program_.scope->Var("feed");
  framework::LoDTensor &target =
      *feed_var->template GetMutable<framework::MobileTensorArray>()
           ->at(index)
           .MuteLodTensor();

  target.Resize(input.dims());
  target.ShareDataWith(input);
  target.set_lod(input.lod());
}

template <typename T>
std::shared_ptr<LoDTensor> Executor<T>::GetOutput(const std::string &var_name) {
  const auto &iter = fetch_indices_.find(var_name);
  if (var_name == "fetch" || iter != fetch_indices_.end()) {
    int index = 0;
    if (iter != fetch_indices_.end()) {
      index = iter->second;
    }
    auto *fetch_var = program_.scope->Var("fetch");
    framework::LoDTensor &target =
        *fetch_var->template GetMutable<framework::MobileTensorArray>()
             ->at(index)
             .MuteLodTensor();

    return std::make_shared<LoDTensor>(target);
  } else {
    auto *fetch_var = program_.scope->Var(var_name);
    framework::LoDTensor *target =
        fetch_var->template GetMutable<framework::LoDTensor>();
    return std::make_shared<LoDTensor>(*target);
  }
}

template <typename T>
PMStatus Executor<T>::Predict() {
#if _OPENMP
  omp_set_num_threads(get_global_num_threads());
#endif
  // clear all no persistable tensor array since write_to_array
  // is always push back a new tensor in the array
  ClearNoPersistableTensorArray(program_desc_.get(), program_.scope.get());

#ifdef PADDLE_MOBILE_PROFILE
  std::vector<ProfInfo> profile(ops_of_block0_.size());
  struct timespec ts;
  int op_index = 0;
#endif
  for (auto &op_handler : ops_of_block0_) {
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[op_index].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    DLOG << "run op: " << op_handler->Type();
    if (lod_mode_) {
      op_handler->InferShape();
    }
    op_handler->Run();
#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[op_index].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
    ++op_index;
#endif
  }
#ifdef PADDLE_MOBILE_PROFILE
  std::unordered_map<std::string, uint64_t> _tp;
  for (int i = 0; i < profile.size(); i++) {
    const auto &pInfo = profile[i];
    uint64_t timeCost = pInfo.runEnd - pInfo.runBegin;
    if (ops_of_block0_[i]->Type() == "conv2d" ||
        ops_of_block0_[i]->Type() == "depthwise_conv2d") {
      auto inputs = ops_of_block0_[i]->Inputs();
      auto *filter =
          GetVarValue<LoDTensor>("Filter", inputs, *(program_.scope));
      int kernel_size = filter->dims()[2];
      _tp[ops_of_block0_[i]->Type() + "_" + std::to_string(kernel_size)] +=
          timeCost;
    } else {
      _tp[ops_of_block0_[i]->Type()] += timeCost;
    }
  }
  printf("====================[ profile ]======================\n");
  typedef std::pair<std::string, uint64_t> prof_t;
  std::vector<prof_t> _tv(_tp.begin(), _tp.end());
  uint64_t _ptotal = 0;
  for (auto const &p : _tv) {
    _ptotal += p.second;
  }
  auto compf = [](const prof_t &a, const prof_t &b) {
    return a.second > b.second;
  };
  std::sort(_tv.begin(), _tv.end(), compf);
  _tv.push_back(std::make_pair("total", _ptotal));
  for (auto const &p : _tv) {
    printf("%-16s\t%-10.0f\t%-2.4f\n", p.first.c_str(),
           static_cast<float>(p.second),
           static_cast<float>(p.second) / _ptotal * 100.0);
  }
  printf("====================[---------]======================\n");
#endif
  return PMSuccess;
}

template <typename T>
void Executor< T>::FeedTensorData(const vector<framework::Tensor> &v) {
  auto input_size = v.size();
  auto *feed_var = program_.scope->Var("feed");

  PADDLE_MOBILE_ENFORCE(input_size == feed_indices_.size(),
                        "input data number not correct");
  for (int i = 0; i < input_size; i++) {
    framework::LoDTensor &target =
        feed_var->template GetMutable<framework::LoDTensorArray>()->at(i);
    target.ShareDataWith(v[input_size - i - 1]);
  }
}

template < typename T>
void Executor<T>::GetTensorResults(
    std::vector<framework::Tensor *> *v) {
  auto *fetch_var = program_.scope->Var("fetch");
  auto output_size = fetch_indices_.size();
  for (int i = 0; i < output_size; i++) {
    framework::LoDTensor &target =
        fetch_var->template GetMutable<framework::LoDTensorArray>()->at(i);
    v->push_back(&target);
  }
}

#ifdef PADDLE_MOBILE_FPGA
template <typename T>
void Executor<T>::InjectVariable(const Tensor &t, std::string var_name) {
  Variable *g_feed_value = program_.scope->Var(var_name);
  Tensor *feed_tensor = g_feed_value->template GetMutable<LoDTensor>();
  feed_tensor->Resize(t.dims());
  feed_tensor->ShareDataWith(t);
}

template <typename T>
void Executor<T>::FeedData(const Tensor &t) {
  InjectVariable(t, "feed0");
}

template <typename T>
void Executor<T>::FeedData(const std::vector<void *> &v) {
  auto input_size = v.size();
  int index = 0;
  auto vars = program_.scope->VarContain("feed", &index);
  PADDLE_MOBILE_ENFORCE(input_size == vars.size(),
                        "input data number not correct");
  for (int i = 0; i < input_size; i++) {
    auto var = program_.scope->Var("feed", i + index);
    auto feed_tensor = var->template GetMutable<LoDTensor>();
    feed_tensor->external_data = v[i];
  }
}

template <typename T>
void Executor<T>::GetResults(std::vector<void *> *v) {
  auto output_size = v->size();
  PADDLE_MOBILE_ENFORCE(output_size > 0, "Empty output");
  int index = 0;
  auto vars = program_.scope->VarContain("fetch", &index);
  PADDLE_MOBILE_ENFORCE(output_size == vars.size(),
                        "output data number not correct");

  for (int i = 0; i < output_size; i++) {
    auto var = program_.scope->Var("fetch", i + index);
    auto fetch_tensor = var->template GetMutable<LoDTensor>();
    (*v)[i] = fetch_tensor->template data<float>();
  }
}

template <typename Device, typename T>
framework::Tensor *Executor<Device, T>::GetTensorByName(
    const std::string &name) {
  auto var = program_.scope->Var(name);
  return var->template GetMutable<LoDTensor>();
}

template <typename T>
std::shared_ptr<Tensor> Executor<T>::FetchResult(int id) {
  auto &ops = ops_of_block0_;

  PADDLE_MOBILE_ENFORCE(id < (int)ops.size(), "Index out of range");
  auto op = id < 0 ? ops[ops.size() - 1] : ops[id];
  auto output_map = op->Outputs();
  std::vector<std::string> out_keys = op->GetOutKeys();
  PADDLE_MOBILE_ENFORCE(!out_keys.empty(), "this op contains no output");
  auto *output_tensor =
      GetVarValue<LoDTensor>(out_keys[0], output_map, *(program_.scope));
  return std::make_shared<Tensor>(Tensor(*output_tensor));
}

template <typename T>
void Executor<T>::Predict_From_To(int start, int end) {
  auto &ops = ops_of_block0_;
  end = end < 0 ? static_cast<int>(ops.size()) : end;
  PADDLE_MOBILE_ENFORCE(start >= 0 && start < end && end <= ops.size(),
                        "start or end parameter is wrong");

#ifdef PADDLE_MOBILE_PROFILE
  std::vector<ProfInfo> profile(ops.size());
#endif
  for (int i = start; i < end; i++) {
#ifdef PADDLE_MOBILE_PROFILE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runBegin = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
    DLOG << "Running op: " << i << "  " << ops[i]->Type();
    ops[i]->Run();

#ifdef PADDLE_MOBILE_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &ts);
    profile[i].runEnd = (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
  }
}

template <typename T>
void Executor<T>::Predict_From(int start) {
  Predict_From_To(start);
}

template <typename T>
void Executor<T>::Predict_To(int end) {
  Predict_From_To(0, end);
}
#ifdef PADDLE_MOBILE_FPGA_V2
std::map<std::string, float> LoadQuantValFromFile(std::string filename) {
  std::map<std::string, float> quantValList;
  std::ifstream in;
  in.open(filename, std::ios::in);
  if (!in.is_open()) {
    std::cout << "open File Failed." << std::endl;
    exit(-1);
  }

  std::string line;
  while (getline(in, line)) {
    std::string splitStr = " : ";
    std::string::size_type pos;
    pos = line.find(splitStr);
    std::string subStr[2];
    subStr[0] = line.substr(0, pos);
    subStr[1] = line.substr(pos + splitStr.size(), line.size());
    quantValList.insert(std::make_pair(subStr[0], atof(subStr[1].c_str())));
  }
  in.close();
  return quantValList;
}

template <typename Device, typename T>
void Executor<Device, T>::InitQuantMemory() {
  std::string quantValFilePath;
  if (program_.combined) {
    quantValFilePath = program_.para_path;
    quantValFilePath =
        quantValFilePath.substr(0, (quantValFilePath.length() - 6));
    quantValFilePath = quantValFilePath + "scale";
  } else {
    quantValFilePath = program_.model_path + "/scale";
  }
  std::map<std::string, float> quantValList =
      LoadQuantValFromFile(quantValFilePath);
  auto ops = ops_of_block0_;
  for (int id = 0; id < ops.size(); id++) {
    auto op = ops[id];
    auto input_keys = op->GetInputKeys();
    auto inputs = op->Inputs();
    for (auto key = input_keys.begin(); key != input_keys.end(); key++) {
      auto inputs_vars = inputs[*key];
      int count = inputs_vars.size();
      for (int i = 0; i < count; i++) {
        auto tensor = GetTensorByName(inputs_vars[i]);
        tensor->scale[0] = quantValList[inputs_vars[i]];
        std::cout << "input variance name : " << inputs_vars[i]
                  << ", scale value : " << tensor->scale[0] << std::endl;
      }
    }
    auto output_keys = op->GetOutKeys();
    auto outputs = op->Outputs();
    for (auto key = output_keys.begin(); key != output_keys.end(); key++) {
      auto outputs_vars = outputs[*key];
      int count = outputs_vars.size();
      for (int i = 0; i < count; i++) {
        auto tensor = GetTensorByName(outputs_vars[i]);
        tensor->scale[0] = quantValList[outputs_vars[i]];
        std::cout << "output variance name : " << outputs_vars[i]
                  << ", scale value : " << tensor->scale[0] << std::endl;
      }
    }
  }
}
#endif
#endif


template class Executor<float>;


}  // namespace framework
}  // namespace paddle_mobile
