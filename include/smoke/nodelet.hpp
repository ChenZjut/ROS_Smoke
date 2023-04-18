/*
 * @Description: 
 * @Author: ubuntu
 * @Date: 2022/11/16 下午4:07
 * @LastEditors: ubuntu
 * @LastEditTime: 2022/11/16 下午4:07
 * @Version 1.0
 */
#ifndef LANE_DETECTION_NODELET_HPP
#define LANE_DETECTION_NODELET_HPP

#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <mutex>
#include <random>
#include <chrono>

// nvidia
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include "NvInferRuntimeCommon.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui//highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/subscriber_filter.h>

#include "smoke.hpp"

namespace lane_detection
{
class TensorrtSegmentNodelet : public nodelet::Nodelet
{
public:
    virtual void onInit();
    void connectCb();
    void callback(const sensor_msgs::Image::ConstPtr & in_image_msg);

private:
    ros::NodeHandle nh_, pnh_;
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Subscriber image_sub_;
    std::mutex connect_mutex_;
    image_transport::Publisher output_segment_pub_;

    std::string onnx_file;
    std::string engine_file;
    std::string mode;
    bool debug;
    int max_batch_size;

    std::unique_ptr<detector::Net> net_ptr_;

};

}

#endif //LANE_DETECTION_NODELET_HPP
