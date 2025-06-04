#pragma once

#include "tensor_core.h"

#ifdef __cplusplus
extern "C" {
#endif

// 卷积操作
int cnn_conv2d(const cnn_core_tensor_t *input, const cnn_core_tensor_t *kernel,
               cnn_core_tensor_t *output, int stride_h, int stride_w, int pad_h,
               int pad_w, int dilation_h, int dilation_w);

int cnn_conv2d_backward_input(const cnn_core_tensor_t *grad_output,
                              const cnn_core_tensor_t *kernel,
                              cnn_core_tensor_t *grad_input, int stride_h,
                              int stride_w, int pad_h, int pad_w);

int cnn_conv2d_backward_kernel(const cnn_core_tensor_t *input,
                               const cnn_core_tensor_t *grad_output,
                               cnn_core_tensor_t *grad_kernel, int stride_h,
                               int stride_w, int pad_h, int pad_w);

// 池化操作
int cnn_maxpool2d(const cnn_core_tensor_t *input, cnn_core_tensor_t *output,
                  cnn_core_tensor_t *indices, int kernel_h, int kernel_w,
                  int stride_h, int stride_w, int pad_h, int pad_w);

int cnn_maxpool2d_backward(const cnn_core_tensor_t *grad_output,
                           const cnn_core_tensor_t *indices,
                           cnn_core_tensor_t *grad_input);

int cnn_avgpool2d(const cnn_core_tensor_t *input, cnn_core_tensor_t *output,
                  int kernel_h, int kernel_w, int stride_h, int stride_w,
                  int pad_h, int pad_w);

int cnn_avgpool2d_backward(const cnn_core_tensor_t *grad_output,
                           cnn_core_tensor_t *grad_input, int kernel_h,
                           int kernel_w, int stride_h, int stride_w);

// 优化实现：im2col + GEMM
int cnn_conv2d_im2col(const cnn_core_tensor_t *input,
                      const cnn_core_tensor_t *kernel,
                      cnn_core_tensor_t *output, cnn_core_tensor_t *workspace,
                      int stride_h, int stride_w, int pad_h, int pad_w);

void cnn_im2col(const float *data_im, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                int stride_w, int dilation_h, int dilation_w, float *data_col);

void cnn_col2im(const float *data_col, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                int stride_w, int dilation_h, int dilation_w, float *data_im);

// 批标准化
int cnn_batch_norm_forward(
    const cnn_core_tensor_t *input, const cnn_core_tensor_t *gamma,
    const cnn_core_tensor_t *beta, cnn_core_tensor_t *running_mean,
    cnn_core_tensor_t *running_var, cnn_core_tensor_t *output,
    cnn_core_tensor_t *save_mean, cnn_core_tensor_t *save_var, float momentum,
    float eps, int training);

int cnn_batch_norm_backward(
    const cnn_core_tensor_t *grad_output, const cnn_core_tensor_t *input,
    const cnn_core_tensor_t *gamma, const cnn_core_tensor_t *save_mean,
    const cnn_core_tensor_t *save_var, cnn_core_tensor_t *grad_input,
    cnn_core_tensor_t *grad_gamma, cnn_core_tensor_t *grad_beta, float eps);

// 工具函数
void cnn_compute_conv_output_size(int input_size, int kernel_size, int stride,
                                  int padding, int dilation, int *output_size);

void cnn_compute_conv_padding_same(int input_size, int kernel_size, int stride,
                                   int dilation, int *pad_before,
                                   int *pad_after);

size_t cnn_conv2d_workspace_size(const cnn_core_tensor_t *input,
                                 const cnn_core_tensor_t *kernel, int stride_h,
                                 int stride_w, int pad_h, int pad_w);

#ifdef __cplusplus
}
#endif