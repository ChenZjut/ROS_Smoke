#pragma once

#include "NvInfer.h"
#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

namespace detector
{
struct Deleter
{
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, Deleter>;

// 打印日志文件
class Logger : public nvinfer1::ILogger
{
public:
    Logger(bool verbose) : verbose_(verbose) {}
    void log(Severity severity, const char * msg) noexcept override {
        if (verbose_ || ((severity != Severity::kINFO) && (severity != Severity::kVERBOSE)))
            std::cout << msg << std::endl;
    }

private:
    bool verbose_{false};
};

struct BboxDim {
    float x;
    float y;
    float z;
};

class Net
{
public:
    // create engine from engine path
    Net(const std::string & engine_path, bool verbose = false);

    // create engine from serialized onnx model
    Net(
      const std::string & onnx_file_path, const std::string & precision, const int max_batch_size,
      bool verbose = false, size_t workspace_size = (1U << 30));
    ~Net();

    // save model to path
    void save(const std::string & path);

    // Infer using pre-allocated GPU buffers {data}
    void infer(const cv::Mat &in_img, std::vector<void *> & buffers, const int batch_size, cv::Mat & res_image, cv::Mat & intrinsic);

    // Get (c, h, w) size of the fixed input
    std::vector<int> getInputSize() const;

    std::vector<int> getOutputSize() const;

    // Get max allowed batch size
    int getMaxBatchSize() const;

    // Get (c, h, w) size of the fixed input
    std::vector<int> getInputDims() const;

    cv::Mat PostProcess(cv::Mat & input_img);

private:
    unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    unique_ptr<nvinfer1::IHostMemory> plan_ = nullptr;
    unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    std::vector<float> image_data_;
    std::vector<float> bbox_preds_;
    std::vector<float> topk_scores_;
    std::vector<float> topk_indices_;
    
    cv::Mat intrinsic_;
    std::vector<float> base_depth_;
    std::vector<BboxDim> base_dims_;

    void load(const std::string & path);
    bool prepare();
};

} // namespace segment
