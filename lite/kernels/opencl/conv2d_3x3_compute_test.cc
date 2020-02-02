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

#include <gtest/gtest.h>
#include <random>
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/target_wrapper.h"

#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

template <typename Dtype1, typename Dtype2>
static void conv_basic(const Dtype1* din,
                       Dtype2* dout,
                       int num,
                       int chout,
                       int hout,
                       int wout,
                       int chin,
                       int hin,
                       int win,
                       const Dtype1* weights,
                       const Dtype2* bias,
                       int group,
                       int kernel_w,
                       int kernel_h,
                       int stride_w,
                       int stride_h,
                       int dila_w,
                       int dila_h,
                       int pad_w,
                       int pad_h,
                       bool flag_bias,
                       bool flag_relu) {
  Dtype2 beta = 0;
  auto src_data = din;
  auto dst_data_ref = dout;
  auto weights_data = weights;
  auto with_bias = flag_bias;
  auto bias_data = bias;

  int in_num = num;
  int out_channels = chout;
  int out_h = hout;
  int out_w = wout;

  int in_channel = chin;
  int in_h = hin;
  int in_w = win;
  int out_c_group = out_channels / group;
  int in_c_group = in_channel / group;

  for (int n = 0; n < in_num; ++n) {
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            int out_idx = n * group * out_c_group * out_h * out_w +
                          g * out_c_group * out_h * out_w + oc * out_h * out_w +
                          oh * out_w + ow;
            Dtype2 bias_d =
                with_bias ? (bias_data[g * out_c_group + oc]) : (Dtype2)0;
            dst_data_ref[out_idx] = bias_d;  // + dst_data_ref[out_idx] * beta;
            for (int ic = 0; ic < in_c_group; ++ic) {
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int iw = ow * stride_w - pad_w + kw * (dila_w);
                  int ih = oh * stride_h - pad_h + kh * (dila_h);
                  if (iw < 0 || iw >= in_w) continue;
                  if (ih < 0 || ih >= in_h) continue;

                  int iidx = n * in_channel * in_h * in_w +
                             g * in_c_group * in_h * in_w + ic * in_h * in_w +
                             ih * in_w + iw;
                  int widx =
                      g * out_c_group * in_c_group * kernel_h * kernel_w +
                      oc * in_c_group * kernel_h * kernel_w +
                      ic * kernel_h * kernel_w + kh * kernel_w + kw;

                  dst_data_ref[out_idx] += src_data[iidx] * weights_data[widx];
                }
              }
            }
            if (flag_relu) {
              dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                          ? dst_data_ref[out_idx]
                                          : (Dtype2)0;
            }
          }
        }
      }
    }
  }
}

TEST(conv2d_3x3_image2d, compute) {
  // conv infos // // 32,3,3,3,  s2p1g1
  const int ksize = 3;
  const int stride = 2;
  const int pad = 1;
  const int group = 1;
  const int dilation = 1;
//  int loop_cnt = 0;

#if 1

#if 1
  const bool bias_flag = false;
  const bool relu_flag = false;
  const int batch_size = 1;
  const int oc = 1;
  const int ih = 3;
  const int iw = 3;
  const int ic = 1;
#else
  const bool bias_flag = false;
  const bool relu_flag = false;
  const int batch_size = 1;
  const int oc = 2;
  const int ih = 5;
  const int iw = 5;
  const int ic = 3;
#endif

#else
  const bool bias_flag = false;
  const bool relu_flag = false;
  const int batch_size = 1;
  const int oc = 32;
  const int ih = 224;  // / 8;
  const int iw = 224;  // / 8;
  const int ic = 3;
#endif

  const int oh = (ih - ksize + 2 * pad) / stride + 1;
  const int ow = (iw - ksize + 2 * pad) / stride + 1;

  const DDim& input_dim =
      lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};
  const DDim& filter_dim =
      lite::DDim{std::vector<int64_t>({oc, ic, ksize, ksize})};
  const DDim& out_dim =
      lite::DDim{std::vector<int64_t>({batch_size, oc, oh, ow})};
  // element wise bias
  const DDim& bias_dim = lite::DDim{std::vector<int64_t>({oc})};

  std::vector<int> paddings = {pad, pad, pad, pad};
  std::vector<int> dilations = {dilation, dilation};

  // set op param
  lite::Tensor input, filter, bias, output;
  operators::ConvParam param;
  param.x = &input;
  param.filter = &filter;
  param.output = &output;
  if (bias_flag) {
    param.bias = &bias;
  }
  param.fuse_relu = relu_flag;
  param.paddings = std::make_shared<std::vector<int>>(paddings);
  param.dilations = std::make_shared<std::vector<int>>(dilations);
  param.strides = std::vector<int>{stride, stride};

  param.x->Resize(input_dim);
  param.filter->Resize(filter_dim);
  param.output->Resize(out_dim);
  if (bias_flag) {
    param.bias->Resize(bias_dim);
  }

  // create kernel
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "conv2d", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  LOG(INFO) << "created kernel, kernel->doc():" << kernel->doc();

  // create context, set param to kernel
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  std::unique_ptr<KernelContext> conv_3x3_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(conv_3x3_context->As<OpenCLContext>()));

  kernel->SetContext(std::move(conv_3x3_context));
  kernel->SetParam(param);

  // generate input, filter, bias, output
  LOG(INFO) << "generate input, filter, bias ...";
  std::default_random_engine engine;
  std::uniform_real_distribution<float> gen(-5, 5);
  std::vector<float> input_v(input_dim.production());
  std::vector<float> output_v(out_dim.production());
  float* filter_data = filter.mutable_data<float>();  // mutable filter
  float* bias_data = bias.mutable_data<float>();      // mutable bias
  float* input_v_data = &input_v[0];
  float* output_v_data = &output_v[0];

  LOG(INFO) << "input_v.size():" << input_v.size();
  LOG(INFO) << "filter.dims().production():" << filter.dims().production();
  LOG(INFO) << "output_v.size():" << output_v.size();

  for (size_t i = 0; i < input_v.size(); ++i) {
    input_v[i] = i + 1;  // gen(engine);
  }
  for (auto& o : output_v) {
    o = 0.;
  }
  for (size_t i = 0; i < filter_dim.production(); ++i) {
    filter_data[i] = 1;  // gen(engine);
  }
  for (size_t i = 0; bias_flag && (i < bias_dim.production()); ++i) {
    bias_data[i] = 3.;  // gen(engine);
  }

  LOG(INFO) << "after gen input and filter ...";
  LOG(INFO) << "input_dim.production(): " << input_dim.production();
  LOG(INFO) << "filter_dim.production(): " << filter_dim.production();
  LOG(INFO) << "out_dim.production(): " << out_dim.production();
  LOG(INFO) << "bias_dim.production(): " << bias_dim.production();

  paddle::lite::CLImageConverterDefault default_convertor;
  // input: compute input dim shape, image shape, mutable
  DDim input_image_shape = default_convertor.InitImageDimInfoWith(input_dim);
  std::vector<float> input_image_data(input_image_shape.production() *
                                      4);  // RGBA
  default_convertor.NCHWToImage(
      input_v.data(), input_image_data.data(), input_dim);
  input.mutable_data<float, cl::Image2D>(
      input_image_shape[0], input_image_shape[1], input_image_data.data());
  // output: compute output dim shape, image shape, mutable
  DDim output_image_shape = default_convertor.InitImageDimInfoWith(out_dim);
  LOG(INFO) << "output_image_shape:" << output_image_shape[0] << " "
            << output_image_shape[1];
  auto* output_image = output.mutable_data<float, cl::Image2D>(
      output_image_shape[0], output_image_shape[1]);

  CHECK(input_dim.production() == input_v.size());
  CHECK_LE(input_dim.production(),
           4 * input_image_shape[0] * input_image_shape[1]);
  CHECK_LE(out_dim.production(),
           4 * output_image_shape[0] * output_image_shape[1]);

  // cpu conv basic calc
  lite::Tensor out_ref;
  out_ref.Resize(out_dim);

  // kernel launch
  LOG(INFO) << "kernel launch ...";
  kernel->Launch();

  // get output from gpu
  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  auto* output_image_data = new float[output_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(output_image_data,
                              output_image,
                              output_image_shape[0],
                              output_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  auto* output_data_cpu = new float[output_image_shape.production() * 4];

  default_convertor.ImageToNCHW(
      output_image_data, output_data_cpu, output_image_shape, output.dims());

  // run cpu ref
  auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
  LOG(INFO) << " conv_basic beigin ..... ";
  conv_basic<float, float>(input_v_data,
                           out_ref_data,
                           batch_size,
                           oc,
                           oh,
                           ow,
                           ic,
                           ih,
                           iw,
                           filter_data,
                           bias_data,
                           group,
                           ksize,
                           ksize,
                           stride,
                           stride,
                           dilation,
                           dilation,
                           pad,
                           pad,
                           bias_flag,
                           relu_flag);
  LOG(INFO) << " conv_basic end ..... ";

  for (int i = 0; i < out_dim.production(); i++) {
    LOG(ERROR) << "output_data_cpu[" << i << "]:" << output_data_cpu[i]
               << " out_ref_data[" << i << "]:" << out_ref_data[i];
  }

  for (int i = 0; i < out_dim.production(); i++) {
    // EXPECT_NEAR(output_data_cpu[i], out_ref_data[i], 1e-3);
    if (abs(output_data_cpu[i] - out_ref_data[i]) > 1e-3) {
      LOG(ERROR) << "---------------------------------------------- error idx:"
                 << i;
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d, kOpenCL, kFloat, kImageDefault, image2d);
