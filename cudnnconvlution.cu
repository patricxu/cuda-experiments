#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cudnn.h>

using namespace cv;
int main(int argc, char** argv)
{
	Mat img = imread("D:\\selfworkspace\\cudnndemo\\test1.jpg");
    Mat img_float;
    img.convertTo(img_float, CV_32F);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        1, img_float.channels(), img_float.rows, img_float.cols);

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        1, img_float.channels(), img_float.rows, img_float.cols);

    int kernelShape[] = {3,3,3,3};
    Mat kernel = Mat(4, kernelShape, CV_32F, Scalar::all(0));
    float kernelMap[9] = {2,1,0,1,0,-1,0,-1,-2};
    for(int i = 0; i < 9; i++) {
        memcpy(kernel.data + i *9*sizeof(float), kernelMap, 9 * sizeof(float));
    }

    cudnnFilterDescriptor_t kernel_desc;
    cudnnCreateFilterDescriptor(&kernel_desc);
    cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            kernelShape[0], kernelShape[1], kernelShape[2], kernelShape[3]);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 1,1,1,1,1,1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    size_t space_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, kernel_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, &space_size);
    void *workspace = nullptr;
    cudaMalloc(&workspace, space_size);

    auto alpha = 1.0f;
    auto beta = 0.0f;
    size_t fm_size = img_float.channels() * img_float.rows * img_float.cols*sizeof(float);
    size_t wt_size = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3] * sizeof(float);
    void *dev_input = nullptr;
    cudaMalloc(&dev_input, fm_size);
    cudaMemcpy(dev_input, img_float.data, fm_size, cudaMemcpyHostToDevice);
    void *dev_kernel = nullptr;
    cudaMalloc(&dev_kernel, wt_size);
    cudaMemcpy(dev_kernel, kernel.data, wt_size, cudaMemcpyHostToDevice);

    void *dev_output = nullptr;
    cudaMalloc(&dev_output, fm_size);
    cudnnConvolutionForward(handle, &alpha, input_desc, dev_input, kernel_desc, dev_kernel, 
        conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, workspace,space_size,&beta, output_desc, dev_output);

    Mat output(img_float);
    cudaMemcpy(output.data, dev_output, fm_size, cudaMemcpyDeviceToHost);
    Mat img_output;
    output.convertTo(img_output, CV_8UC3);
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    imshow("output", img_output);
    waitKey(0);
    destroyWindow("output");
    return 0;
}