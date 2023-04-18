#include <fstream>
#include <stdexcept>

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include "smoke.hpp"

#include "trt_modulated_deform_conv.hpp"

#define IMAGE_H 375
#define IMAGE_W 1242
#define SCORE_THRESH 0.3f
#define TOPK 100

namespace detector
{
void Net::load(const std::string &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        std::cout << "read serialized file failed\n";
        return;
    }

    // 读出文件里面的内容
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char * buffer = new char[size];
    file.read(buffer, size);
    file.close();
    std::cout << "modle size: " << size << std::endl;
    if (runtime_) {
        engine_ =
          unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer, size, nullptr));
    }
    delete[] buffer;
}

bool Net::prepare()
{
    if (!engine_) {
        return false;
    }
    context_ = unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        return false;
    }
    image_data_.resize(getInputSize()[0] * getInputSize()[1] * getInputSize()[2]);

    // https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/smoke.py#L41
    base_depth_ = {28.01f, 16.32f};
    base_dims_.resize(3);  //pedestrian, cyclist, car
    base_dims_[0].x = 0.88f;
    base_dims_[0].y = 1.73f;
    base_dims_[0].z = 0.67f;
    base_dims_[1].x = 1.78f;
    base_dims_[1].y = 1.70f;
    base_dims_[1].z = 0.58f;
    base_dims_[2].x = 3.88f;
    base_dims_[2].y = 1.63f;
    base_dims_[2].z = 1.53f;    

    cudaStreamCreate(&stream_);
    return true;
}

Net::Net(const std::string &engine_path, bool verbose)
{
    Logger logger(verbose);
    initLibNvInferPlugins(&logger, "");
    runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    load(engine_path);
    if (!prepare()) {
        std::cout << "Fail to prepare engine" << std::endl;
        return;
    }
}

Net::~Net()
{
    if (stream_) cudaStreamDestroy(stream_);
}

Net::Net(const std::string& onnx_file_path, const std::string& precision, const int max_batch_size,
         bool verbose, size_t workspace_size)
{
    Logger logger(verbose);
    initLibNvInferPlugins(&logger, "");
    // create runtime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);

    // create builder
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));

    // create config from builder
    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    
    // fp16精度的模型类别预测错误,会造成后处理崩溃.
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);
    #if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    #else
        config->setMaxWorkspaceSize(workspace_size);
    #endif

    // create network
    // const auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    const auto flag = 1U;
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

    // create parser to fufill network
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    parser->parseFromFile(
        onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

    // Build engine
    std::cout << "Applying optimizations and building TRT CUDA engine..." << std::endl;
    plan_ = unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan_) {
        std::cout << "Fail to create serialized network" << std::endl;
        return;
    }
    engine_ = unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan_->data(), plan_->size()));
    if (!prepare()) {
        std::cout << "Fail to prepare engine" << std::endl;
        return;
    }
}

void Net::save(const std::string & path)
{
    std::cout << "Writing to " << path << "..." << std::endl;
    std::ofstream file(path, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
}

void Net::infer(const cv::Mat &in_img, std::vector<void *> & buffers, const int batch_size, cv::Mat & res_image, cv::Mat & intrinsic)
{
    intrinsic_ = intrinsic;
    const int INPUT_H = getInputSize()[1];
    const int INPUT_W = getInputSize()[2];
    
    // Modify camera intrinsics due to scaling
    intrinsic_.at<float>(0, 0) *= static_cast<float>(INPUT_W) / IMAGE_W;
    intrinsic_.at<float>(0, 2) *= static_cast<float>(INPUT_W) / IMAGE_W;
    intrinsic_.at<float>(1, 1) *= static_cast<float>(INPUT_H) / IMAGE_H;
    intrinsic_.at<float>(1, 2) *= static_cast<float>(INPUT_H) / IMAGE_H;

    // Preprocessing
    cv::Mat img_resize;
    cv::resize(in_img, img_resize, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    // img_resize.convertTo(img_resize, CV_32FC3, 1.0f);
    float mean[3] {123.675f, 116.280f, 103.530f};
    float std[3] = {58.395f, 57.120f, 57.375f};
    uint8_t* data_hwc = reinterpret_cast<uint8_t*>(img_resize.data);
    float* data_chw = image_data_.data();
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0, img_size = INPUT_H * INPUT_W; j < img_size; ++j) {
            data_chw[c * img_size + j] = (data_hwc[j * 3 + 2 - c] - mean[c]) / std[c];  //bgr2rgb
        }
    }

    const int in_size{static_cast<int>(image_data_.size())};
    const int out_size(TOPK * 8);
    const int size(TOPK);
    bbox_preds_.resize(out_size);
    topk_scores_.resize(size);
    topk_indices_.resize(size);
    

    if (!context_) {
        throw std::runtime_error("Fail to create context");
    }

    cudaError_t state;
    state = cudaMalloc(&buffers[0], in_size * sizeof(float));
    if (state) {
        std::cout << "Allocate memory failed" << std::endl;
        return;
    }

    state = cudaMalloc(&buffers[1], out_size * sizeof(float));
    if (state) {
        std::cout << "Allocate memory failed" << std::endl;
        return;
    }

    state = cudaMalloc(&buffers[2], size * sizeof(float));
    if (state) {
        std::cout << "Allocate memory failed" << std::endl;
        return;
    }

    state = cudaMalloc(&buffers[3], size * sizeof(float));
    if (state) {
        std::cout << "Allocate memory failed" << std::endl;
        return;
    }

    state = cudaMemcpyAsync(buffers[0], image_data_.data(), 
              in_size * sizeof(float), cudaMemcpyHostToDevice, stream_);

    if (state) {
        std::cout << "Transmit to device failed" << std::endl;
        return;
    }
    context_->enqueueV2(&buffers[0], stream_, nullptr);
    state = cudaMemcpyAsync(
            bbox_preds_.data(), buffers[1], out_size * sizeof(float),
            cudaMemcpyDeviceToHost, stream_);
    if (state) {
        std::cout << "Transmit to host failed" << std::endl;
        return;
    }

    state = cudaMemcpyAsync(topk_scores_.data(), buffers[2], size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    if (state) {
        std::cout << "Transmit to host failed" << std::endl;
        return;
    }

    state = cudaMemcpyAsync(topk_indices_.data(), buffers[3], size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    if (state) {
        std::cout << "Transmit to host failed" << std::endl;
        return;
    }

    cudaStreamSynchronize(stream_);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaFree(buffers[3]);
    res_image = PostProcess(img_resize);
}

std::vector<int> Net::getInputSize() const
{
    auto dims = engine_->getBindingDimensions(0);
    return {dims.d[1], dims.d[2], dims.d[3]};
}

std::vector<int> Net::getOutputSize() const
{
    auto dims = engine_->getBindingDimensions(1);
    return {dims.d[2], dims.d[3]};
}

int Net::getMaxBatchSize() const
{
    return engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
}

std::vector<int> Net::getInputDims() const
{
    auto dims = engine_->getBindingDimensions(0);
    return {dims.d[1], dims.d[2], dims.d[3]};
}

float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

cv::Mat Net::PostProcess(cv::Mat & input_img)
{
    const int INPUT_H = getInputSize()[1];
    const int INPUT_W = getInputSize()[2];

    const int OUTPUT_H = (INPUT_H / 4);
    const int OUTPUT_W = (INPUT_W / 4);

    for (int i = 0; i < TOPK; ++i) {
        if (topk_scores_[i] < SCORE_THRESH) {
            continue;
        }
        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/coders/smoke_bbox_coder.py#L52
        int class_id = static_cast<int>(topk_indices_[i] / OUTPUT_H / OUTPUT_W);
        int location = static_cast<int>(topk_indices_[i]) % (OUTPUT_H * OUTPUT_W);
        int img_x = location % OUTPUT_W;
        int img_y = location / OUTPUT_W;
        // Depth
        float z = base_depth_[0] + bbox_preds_[8*i] * base_depth_[1];
        // Location
        cv::Mat img_point(3, 1, CV_32FC1);
        img_point.at<float>(0) = 4.0f * (static_cast<float>(img_x) + bbox_preds_[8*i + 1]);
        img_point.at<float>(1) = 4.0f * (static_cast<float>(img_y) + bbox_preds_[8*i + 2]);
        img_point.at<float>(2) = 1.0f;
        cv::Mat cam_point = intrinsic_.inv() * img_point * z;
        float x = cam_point.at<float>(0);
        float y = cam_point.at<float>(1);
        // Dimension
        float w = base_dims_[class_id].x * expf(Sigmoid(bbox_preds_[8*i + 3]) - 0.5f);
        float l = base_dims_[class_id].y * expf(Sigmoid(bbox_preds_[8*i + 4]) - 0.5f);
        float h = base_dims_[class_id].z * expf(Sigmoid(bbox_preds_[8*i + 5]) - 0.5f);
        // Orientation
        float ori_norm = sqrtf(powf(bbox_preds_[8*i + 6], 2.0f) + powf(bbox_preds_[8*i + 7], 2.0f));
        bbox_preds_[8*i + 6] /= ori_norm;  //sin(alpha)
        bbox_preds_[8*i + 7] /= ori_norm;  //cos(alpha)
        float ray = atan(x / (z + 1e-7f));
        float alpha = atan(bbox_preds_[8*i + 6] / (bbox_preds_[8*i + 7] + 1e-7f));
        if (bbox_preds_[8*i + 7] > 0.0f) {
            alpha -= M_PI / 2.0f;
        } else {
            alpha += M_PI / 2.0f;
        }
        float angle = alpha + ray;
        if (angle > M_PI) {
            angle -= 2.0f * M_PI;
        } else if (angle < -M_PI) {
            angle += 2.0f * M_PI;
        }

        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py#L117
        //              front z
        //                   /
        //                  /
        //    (x0, y0, z1) + -----------  + (x1, y0, z1)
        //                /|            / |
        //               / |           /  |
        // (x0, y0, z0) + ----------- +   + (x1, y1, z1)
        //              |  /      .   |  /
        //              | / origin    | /
        // (x0, y1, z0) + ----------- + -------> x right
        //              |             (x1, y1, z0)
        //              |
        //              v
        //         down y
        cv::Mat cam_corners = (cv::Mat_<float>(8, 3) << 
            -w, -l, -h,     // (x0, y0, z0)
            -w, -l,  h,     // (x0, y0, z1)
            -w,  l,  h,     // (x0, y1, z1)
            -w,  l, -h,     // (x0, y1, z0)
             w, -l, -h,     // (x1, y0, z0)
             w, -l,  h,     // (x1, y0, z1)
             w,  l,  h,     // (x1, y1, z1)
             w,  l, -h);    // (x1, y1, z0)
        cam_corners = 0.5f * cam_corners;
        cv::Mat rotation_y = cv::Mat::eye(3, 3, CV_32FC1);
        rotation_y.at<float>(0, 0) = cosf(angle);
        rotation_y.at<float>(0, 2) = sinf(angle);
        rotation_y.at<float>(2, 0) = -sinf(angle);
        rotation_y.at<float>(2, 2) = cosf(angle);
        // cos, 0, sin
        //   0, 1,   0
        //-sin, 0, cos
        cam_corners = cam_corners * rotation_y.t();
        for (int i = 0; i < 8; ++i) {
            cam_corners.at<float>(i, 0) += x;
            cam_corners.at<float>(i, 1) += y;
            cam_corners.at<float>(i, 2) += z;
        }
        cam_corners = cam_corners * intrinsic_.t();
        std::vector<cv::Point2f> img_corners(8);
        for (int i = 0; i < 8; ++i) {
            img_corners[i].x = cam_corners.at<float>(i, 0) / cam_corners.at<float>(i, 2);
            img_corners[i].y = cam_corners.at<float>(i, 1) / cam_corners.at<float>(i, 2);
        }
        for (int i = 0; i < 4; ++i) {
            const auto& p1 = img_corners[i];
            const auto& p2 = img_corners[(i + 1) % 4];
            const auto& p3 = img_corners[i + 4];
            const auto& p4 = img_corners[(i + 1) % 4 + 4];
            cv::line(input_img, p1, p2, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            cv::line(input_img, p3, p4, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            cv::line(input_img, p1, p3, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
        }
    }
    return input_img;
}
}