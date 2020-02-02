// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/opencl/conv_compute.h"
#include <sstream>
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ConvCompute::PrepareForRun() {
  const auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);

  int bs = x_dims[0];
  int c_in = x_dims[1];
  int h_out = output_dims[2];
  int w_out = output_dims[3];
  int kernel_h = filter_dims[2];  // oihw
  int kernel_w = filter_dims[3];
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int groups = param.groups;
  bool relu_fused = param.fuse_relu;
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  bool zero_pad = (pad_h == 0) && (pad_w == 0);

  bool pad_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));

  VLOG(3) << "Is relu fused? / " << (relu_fused ? "Yes" : "No");
  VLOG(3) << "groups:" << groups << " stride_h:" << stride_h
          << " stride_w:" << stride_w << " pad_h:" << pad_h
          << " pad_w:" << pad_w << " kernel_h:" << kernel_h
          << " kernel_h:" << kernel_h;
  VLOG(3) << "x_dims:" << x_dims[0] << " " << x_dims[1] << " " << x_dims[2]
          << " " << x_dims[3];
  VLOG(3) << "output_dims:" << output_dims[0] << " " << output_dims[1] << " "
          << output_dims[2] << " " << output_dims[3];
  VLOG(3) << "filter_dims:" << filter_dims[0] << " " << filter_dims[1] << " "
          << filter_dims[2] << " " << filter_dims[3];

  if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 &&
      zero_pad && no_dilation && pad_equal) {
    // conv2d_1x1
    kernel_func_names_.push_back("gemm_batch");
    kernel_func_paths_.push_back("buffer/fc_kernel.cl");
    if (relu_fused) {
      build_options_.push_back("-DCL_DTYPE_float -DRELU");
    } else {
      build_options_.push_back("-DCL_DTYPE_float");
    }
    impl_ = &ConvCompute::Conv2d1x1;
  } else if (pad_equal) {
    kernel_func_names_.push_back("im2col");
    kernel_func_names_.push_back("gemm_batch");
    kernel_func_paths_.push_back("buffer/im2col_kernel.cl");
    kernel_func_paths_.push_back("buffer/fc_kernel.cl");
    if (relu_fused) {
      build_options_.push_back("-DCL_DTYPE_float -DRELU");
    } else {
      build_options_.push_back("-DCL_DTYPE_float");
    }
    impl_ = &ConvCompute::GemmlikeConv2d;
    col_buffer_.reset(new lite::Tensor);
    col_buffer_->Resize({bs, c_in, kernel_h * kernel_w, h_out * w_out});
    col_buffer_->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  } else {
    LOG(FATAL) << "This pad not support ! " << paddings[0] << ", "
               << paddings[1] << ", " << paddings[2] << ", " << paddings[3];
  }

  for (size_t i = 0; i < kernel_func_names_.size(); i++) {
    context.cl_context()->AddKernel(
        kernel_func_names_[i], kernel_func_paths_[i], build_options_[i]);
  }
}

void ConvCompute::GemmlikeConv2d() {
  const auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();
  int bs = x_dims[0];
  int c_in = x_dims[1];
  int h_in = x_dims[2];
  int w_in = x_dims[3];
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int c_out = output_dims[1];
  int h_out = output_dims[2];
  int w_out = output_dims[3];
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dilation_h = dilations[0];
  int dilation_w = dilations[1];

  auto* x_buf = param.x->data<float, cl::Buffer>();
  auto* filter_buf = param.filter->data<float, cl::Buffer>();
  auto* bias_buf = (param.bias == nullptr)
                       ? static_cast<cl::Buffer*>(nullptr)
                       : param.bias->data<float, cl::Buffer>();
  auto* output_buf =
      param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto* col_buf = col_buffer_->mutable_data<float, cl::Buffer>();

  auto& context = ctx_->As<OpenCLContext>();
  std::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto img2col_kernel = context.cl_context()->GetKernel(kernel_key.str());

  int n_threads = c_in * h_out * w_out;
  int in_stride = c_in * h_in * w_in;
  int out_stride = c_in * kernel_h * kernel_w * h_out * w_out;
  int img_offset = 0;
  int col_offset = 0;
  int arg_idx = 0;
  cl_int status;
  for (int b = 0; b < bs; b++) {
    img_offset = b * in_stride;
    col_offset = b * out_stride;
    arg_idx = 0;
    status = img2col_kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, img_offset);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, n_threads);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, h_in);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, w_in);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, kernel_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, kernel_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, pad_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, pad_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, stride_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, stride_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, dilation_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, dilation_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, h_out);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, w_out);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, *col_buf);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, col_offset);
    CL_CHECK_FATAL(status);

    auto global_work_size = cl::NDRange{static_cast<size_t>(out_stride)};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        img2col_kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        nullptr);
    CL_CHECK_FATAL(status);
  }

  int m = c_out;
  int k = c_in * kernel_h * kernel_w;
  int n = h_out * w_out;
  VLOG(4) << "m = " << m << " n = " << n << " k = " << k;
  kernel_key.str("");
  kernel_key << kernel_func_names_[1] << build_options_[1];
  auto gemm_kernel = context.cl_context()->GetKernel(kernel_key.str());
  GemmBatched(
      gemm_kernel, col_buf, filter_buf, bias_buf, output_buf, bs, m, n, k);
}

void ConvCompute::Conv2d1x1() {
  const auto& param = *param_.get_mutable<param_t>();
  const int batch_size = param.x->dims()[0];
  const int k = param.x->dims()[1];  // K: input_channel
  const int n = param.x->dims()[2] *
                param.x->dims()[3];       // N == X_HxW == input_h * input_w
  const int m = param.output->dims()[1];  // M: output_channel == filter number

  VLOG(4) << "m = " << m << " n = " << n << " k = " << k;

  if (param.groups != 1) {
    LOG(FATAL) << "conv2d_1x1 with group > 1 not supported and param.groups = "
               << param.groups;
  }

  auto* x_d = param.x->data<float, cl::Buffer>();
  auto* filter_d = param.filter->data<float, cl::Buffer>();
  auto* bias_d = (param.bias == nullptr)
                     ? static_cast<cl::Buffer*>(nullptr)
                     : param.bias->data<float, cl::Buffer>();
  auto* output_d =
      param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  auto& context = ctx_->As<OpenCLContext>();
  std::stringstream kernel_key;
  kernel_key << kernel_func_names_.front() << build_options_.front();
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());

  GemmBatched(kernel, x_d, filter_d, bias_d, output_d, batch_size, m, n, k);
}

// a: filter_d ==> <m, k> <=> <oc, ic>
// b: x_d      ==> <k, n> <=> <ic, ih*iw>
// c: output_d ==> <m, n> <=> <oc, ih*iw>
void ConvCompute::GemmBatched(cl::Kernel& kernel,
                              const cl::Buffer* x_d,
                              const cl::Buffer* filter_d,
                              const cl::Buffer* bias_d,
                              cl::Buffer* output_d,
                              const int batch_size,
                              const int m,
                              const int n,
                              const int k) {
  auto global_work_size = cl::NDRange{static_cast<size_t>((m + 7) / 8),
                                      static_cast<size_t>((n + 3) / 4),
                                      static_cast<size_t>(batch_size)};
  auto local_work_size = cl::NDRange{16, 16};  // cl::NullRange;

  auto& context = ctx_->As<OpenCLContext>();
  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, *filter_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *x_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *bias_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *output_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, m);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, n);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, k);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, batch_size);
  CL_CHECK_FATAL(status);

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      local_work_size,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);

  context.cl_wait_list()->emplace(output_d, event_);
}

void ConvCompute::Run() { (this->*impl_)(); }

// ConvComputeImage
void ConvComputeImage::PrepareForRun() {
  const auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  int bs = x_dims[0];
  int c_in = x_dims[1];
  int h_out = output_dims[2];
  int w_out = output_dims[3];
  int kernel_h = filter_dims[2];  // oihw
  int kernel_w = filter_dims[3];
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int groups = param.groups;
  bool relu_fused = param.fuse_relu;
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);

  bool zero_pad = (pad_h == 0) && (pad_w == 0);
  bool pad_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));

  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();

  VLOG(3) << "Is relu fused? / " << (relu_fused ? "Yes" : "No");
  VLOG(3) << "Has bias? / " << (has_bias ? "Yes" : "No");
  VLOG(3) << "is_element_wise_bias? / "
          << (is_element_wise_bias ? "Yes" : "No");
  VLOG(3) << "groups:" << groups << " stride_h:" << stride_h
          << " stride_w:" << stride_w << " pad_h:" << pad_h
          << " pad_w:" << pad_w << " kernel_h:" << kernel_h
          << " kernel_h:" << kernel_h;
  VLOG(3) << "x_dims:" << x_dims[0] << " " << x_dims[1] << " " << x_dims[2]
          << " " << x_dims[3];
  VLOG(3) << "output_dims:" << output_dims[0] << " " << output_dims[1] << " "
          << output_dims[2] << " " << output_dims[3];
  VLOG(3) << "filter_dims:" << filter_dims[0] << " " << filter_dims[1] << " "
          << filter_dims[2] << " " << filter_dims[3];

  // Choose concret conv kernel impl_
  if (kernel_h == 3 && kernel_w == 3 && pad_equal) {
    // conv_3x3
    kernel_func_names_.push_back("conv2d_3x3");
    kernel_func_paths_.push_back("image/conv2d_3x3_kernel.cl");
    if (relu_fused) {
      build_options_.push_back("-DCL_DTYPE_float -DRELU");
    } else {
      build_options_.push_back("-DCL_DTYPE_float");
    }
    if (has_bias) {
      build_options_[0] +=
          is_element_wise_bias ? " -DBIASE_ELE" : " -DBIASE_CH";
    }
    Conv3x3Prepare();
    impl_ = &ConvComputeImage::Conv3x3Run;
  } else {
    LOG(FATAL) << "unsupported conv2d kernel! " << paddings[0] << ", "
               << paddings[1] << ", " << paddings[2] << ", " << paddings[3];
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  for (size_t i = 0; i < kernel_func_names_.size(); i++) {
    context.cl_context()->AddKernel(
        kernel_func_names_[i], kernel_func_paths_[i], build_options_[i]);
  }
}

void ConvComputeImage::Conv3x3Prepare() {
  // do layout trans for filter and bias
  // 1. filter/bias: buffer to Image2D(folder)
  // 2. create image with cpu buffer
  // 3. create image with gpu image2d from cpu buffer
  // 4. assign gpu image2d to member variables: filter_img_gpu_, bias_img_gpu_
  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto filter_dims = param.filter->dims();
  auto filter_data = param.filter->data<float>();
  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();

  VLOG(3) << "filter_dims[" << filter_dims.size() << "-D]:" << filter_dims[0]
          << "," << filter_dims[1] << "," << filter_dims[2] << ","
          << filter_dims[3];
  VLOG(3) << "Has bias? / " << (has_bias ? "Yes" : "No");
  VLOG(3) << "is_element_wise_bias? / "
          << (is_element_wise_bias ? "Yes" : "No");

  VLOG(3) << "output_dims:" << output_dims[0] << " " << output_dims[1] << " "
          << output_dims[2] << " " << output_dims[3];

  paddle::lite::CLImageConverterFolder folder_convertor;
  lite::Tensor filter_img_cpu;
  auto filter_img_dims = folder_convertor.InitImageDimInfoWith(filter_dims);
  auto* filter_img_cpu_data = static_cast<float*>(filter_img_cpu.mutable_data(
      filter_img_dims.production() * 4 * sizeof(float)));  // 4 for RGBA

  folder_convertor.NCHWToImage(
      const_cast<float*>(filter_data), filter_img_cpu_data, filter_img_dims);
  CHECK_LE(filter_dims.production(),
           4 * filter_img_dims[0] * filter_img_dims[1]);
  VLOG(3) << "filter_img_dims:" << filter_img_dims[0] << ","
          << filter_img_dims[1];
  VLOG(3) << "filter_img_cpu_data:" << filter_img_cpu_data;
  filter_img_gpu_t_.mutable_data<float, cl::Image2D>(
      filter_img_dims[0], filter_img_dims[1], filter_img_cpu_data);
  if (has_bias) {
    auto bias_dims = param.bias->dims();
    for (size_t i = 0; i < bias_dims.size(); ++i) {
      VLOG(3) << "bias_dims[" << i << "]:" << bias_dims[i];
    }
    lite::Tensor bias_img_cpu;
    auto bias_img_dims = folder_convertor.InitImageDimInfoWith(bias_dims);
    bias_img_cpu.Resize(bias_img_dims);
    auto* bias_img_cpu_data = static_cast<float*>(bias_img_cpu.mutable_data(
        bias_img_dims.production() * 4 * sizeof(float)));  // 4 for RGBA
    auto bias_data = param.bias->data<float>();
    folder_convertor.NCHWToImage(
        const_cast<float*>(bias_data), bias_img_cpu_data, bias_img_dims);
    VLOG(3) << "bias_img_dims:" << bias_img_dims;
    VLOG(3) << "bias_img_cpu_data:" << bias_img_cpu_data;
    bias_img_gpu_t_.mutable_data<float, cl::Image2D>(
        bias_img_dims[0], bias_img_dims[1], bias_img_cpu_data);
#if 1
    LOG(INFO) << "filter_img_gpu_t_.data(gpu):"
              << filter_img_gpu_t_.mutable_data<float, cl::Image2D>(
                     filter_img_dims[0],
                     filter_img_dims[1],
                     filter_img_cpu_data);

    LOG(INFO) << "bias_img_gpu_t_.data(gpu):"
              << bias_img_gpu_t_.mutable_data<float, cl::Image2D>(
                     bias_img_dims[0], bias_img_dims[1], bias_img_cpu_data);
//  exit(0);
#endif
  }
  VLOG(4) << "finished PrepareConv3x3";
}

void ConvComputeImage::Conv3x3Run() {
  VLOG(4) << "1111";
  const auto& param = this->Param<param_t>();
  auto paddings = *param.paddings;
  auto strides = param.strides;
  VLOG(4) << "1111";

  auto dilations = *param.dilations;
  int group = param.groups;
  VLOG(4) << "1111";

  // input
  auto* input_image = param.x->data<float, cl::Image2D>();
  auto input_dims = param.x->dims();
  int input_channel = input_dims[1];
  int input_width = input_dims[3];
  int input_height = input_dims[2];
  auto input_image_shape = InitImageDimInfoWith(input_dims);
  int input_c_block = input_image_shape["width"] / input_dims[3];
  int input_c = input_c_block;
  VLOG(4) << "1111";
  VLOG(4) << "+============================================";
  VLOG(4) << "+============================================";
  VLOG(4) << "input_image_shape:" << input_image_shape["width"] << " "
          << input_image_shape["height"];
  VLOG(4) << "input_c_block:" << input_c_block << " input_c:" << input_c;
  VLOG(4) << "input_dims:" << input_dims;

  // output
  auto output_dims = param.output->dims();
  int output_width = output_dims[3];
  int output_height = output_dims[2];
  auto out_image_shape = InitImageDimInfoWith(output_dims);
  auto* output_image = param.output->mutable_data<float, cl::Image2D>(
      out_image_shape["width"], out_image_shape["height"]);
  int output_c = output_dims[1];

  // filter_img
  auto* filter_image = filter_img_gpu_t_.data<float, cl::Image2D>();
  VLOG(4) << "1111";
  auto filter_dims = param.filter->dims();
  VLOG(4) << "1111";
  int filter_channel = filter_dims[1];
  VLOG(4) << "1111";

  // bias_img
  auto* bias_image = bias_img_gpu_t_.data<float, cl::Image2D>();
  VLOG(4) << "bias_image:" << bias_image;
  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();
  int offset = static_cast<int>(param.filter->dims()[2]) / 2 -
               static_cast<int>(paddings[0]);
  VLOG(4) << "1111";

#if 1
  LOG(INFO) << "--------------------------- filter_image:" << filter_image;
  LOG(INFO) << "--------------------------- bias_image:" << bias_image;
// exit(0);
#endif

  // other misc
  const std::vector<size_t>& default_work_size =
      DefaultWorkSize(output_dims,
                      DDim(std::vector<DDim::value_type>{
                          static_cast<int64_t>(out_image_shape["width"]),
                          static_cast<int64_t>(out_image_shape["height"])}));

  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];
  int maped_w = maptofactor(w, 4);

  VLOG(4) << "============ conv2d_3x3 params ============";
  VLOG(4) << "input_image_shape: " << input_image_shape["width"] << ","
          << input_image_shape["height"];
  VLOG(4) << "input_c_block: " << input_c_block;
  VLOG(4) << "input_c: " << input_c;
  VLOG(4) << "input_image: " << input_image;
  VLOG(4) << "output_image: " << output_image;
  VLOG(4) << "filter_dims: " << filter_dims;
  VLOG(4) << "&filter_img_gpu_t_: " << &filter_img_gpu_t_;
  VLOG(4) << "filter_image:" << filter_image;
  VLOG(4) << "&bias_img_gpu_t_: " << &bias_img_gpu_t_;
  VLOG(4) << "output_dims: " << output_dims;
  VLOG(4) << "out_image_shape: " << out_image_shape["width"] << ", "
          << out_image_shape["height"];
  VLOG(4) << "paddings: " << paddings[0] << "," << paddings[1];
  VLOG(4) << "has bias: " << has_bias;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << strides[0] << "," << strides[1];
  VLOG(4) << "offset: " << offset;
  VLOG(4) << "dilations.size : " << dilations.size();
  VLOG(4) << "dilations: " << dilations[0] << ", " << dilations[1];
  VLOG(4) << "default work size{c_block, w, nh}: "
          << "{" << c_block << ", " << w << ", " << nh << ""
          << "}";

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  std::stringstream kernel_key;
  kernel_key << kernel_func_names_.front() << build_options_.front();
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());
  auto global_work_size =
      cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                  static_cast<size_t>(default_work_size.data()[1]),
                  static_cast<size_t>(default_work_size.data()[2])};

  VLOG(4) << "global_work_size[3D]: {" << global_work_size[0] << ","
          << global_work_size[1] << "," << global_work_size[2] << "}";

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, static_cast<const int>(c_block));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(w));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(nh));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_image);
  CL_CHECK_FATAL(status);
  if (has_bias) {
    status = kernel.setArg(++arg_idx, *bias_image);
    CL_CHECK_FATAL(status);
  }
  status = kernel.setArg(++arg_idx, *output_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(strides[0]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(offset));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(input_c));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(dilations[0]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(input_width));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(input_height));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(output_width));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(output_height));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(output_c));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(filter_channel));
  CL_CHECK_FATAL(status);

  if (filter_dims[0] == output_dims[1] && filter_dims[1] == input_dims[1]) {
    group = 1;
  } else if (!(filter_dims[0] == input_dims[1] &&
               filter_dims[1] == 1)) {  // not depthwise
    group = input_channel / filter_channel;
  }

  status = kernel.setArg(++arg_idx, static_cast<const int>(group));
  CL_CHECK_FATAL(status);

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(output_image, event_);
  context.cl_context()->GetCommandQueue().finish();

  LOG(INFO) << "c_block:" << c_block;
  LOG(INFO) << "w:" << w;
  LOG(INFO) << "nh:" << nh;
  LOG(INFO) << "strides[0]:" << strides[0];
  LOG(INFO) << "offset:" << offset;
  LOG(INFO) << "input_c:" << input_c;
  LOG(INFO) << "dilations[0]:" << dilations[0];
  LOG(INFO) << "input_width:" << input_width;
  LOG(INFO) << "input_height:" << input_height;
  LOG(INFO) << "output_width:" << output_width;
  LOG(INFO) << "output_height:" << output_height;
  LOG(INFO) << "output_c:" << output_c;
  LOG(INFO) << "filter_channel:" << filter_channel;
  LOG(INFO) << "group:" << group;

  // debug_image(param.output);
}

void ConvComputeImage::Run() { (this->*impl_)(); }

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::ConvCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();

REGISTER_LITE_KERNEL(conv2d,
                     kOpenCL,
                     kFloat,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvComputeImage,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
