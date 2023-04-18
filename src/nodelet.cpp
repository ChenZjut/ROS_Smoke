/*
 * @Description: 
 * @Author: ubuntu
 * @Date: 2022/11/16 下午4:06
 * @LastEditors: ubuntu
 * @LastEditTime: 2022/11/16 下午4:06
 * @Version 1.0
 */

#include <smoke/nodelet.hpp>

namespace lane_detection
{
void TensorrtSegmentNodelet::onInit()
{
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    image_transport_.reset(new image_transport::ImageTransport(nh_));
    pnh_.param<std::string>("onnx_file", onnx_file, "");
    pnh_.param<std::string>("engine_file", engine_file, "");
    pnh_.param<std::string>("mode", mode, "FP32");
    pnh_.param<int>("max_batch_size", max_batch_size, 1);

    std::ifstream fs(engine_file);

    if (fs.is_open()) {
        NODELET_INFO("Found %s", engine_file.c_str());
        net_ptr_.reset(new detector::Net(engine_file, false));
        if (net_ptr_->getMaxBatchSize() != 1) {
            NODELET_INFO(
              "Max batch size %d should be 1. Rebuild engine from file", net_ptr_->getMaxBatchSize());
            net_ptr_.reset(
              new detector::Net(onnx_file, mode, max_batch_size));
            net_ptr_->save(engine_file);
        }
    } else {
        NODELET_INFO("Could not find %s, try making TensorRT engine from onnx", engine_file.c_str());
        net_ptr_.reset(new detector::Net(onnx_file, mode, max_batch_size));
        net_ptr_->save(engine_file);
    }

    image_transport::SubscriberStatusCallback connect_cb = boost::bind(&TensorrtSegmentNodelet::connectCb, this);
    std::lock_guard<std::mutex> lock(connect_mutex_);
    output_segment_pub_ = image_transport_->advertise("out/image", 1, connect_cb, connect_cb);
}

void TensorrtSegmentNodelet::connectCb()
{
    std::lock_guard<std::mutex> lock(connect_mutex_);
    if (output_segment_pub_.getNumSubscribers() == 0) {
        image_sub_.shutdown();
    } else if (!image_sub_) {
        image_sub_ = image_transport_->subscribe("in/image", 1, &TensorrtSegmentNodelet::callback, this);
    }
}

void TensorrtSegmentNodelet::callback(const sensor_msgs::Image::ConstPtr &in_image_msg)
{
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    const auto exe_start_time = high_resolution_clock::now();
    cv_bridge::CvImagePtr cv_image;
    try {
        cv_image = cv_bridge::toCvCopy(in_image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception & e) {
        NODELET_ERROR("Failed to convert sensor_msgs::Image to cv::Mat \n%s", e.what());
    }


    std::vector<void *>buffers(4);
    cv::Mat res_image;
    cv::Mat intrinsic = (cv::Mat_<float>(3, 3) << 
        721.5377, 0.0, 609.5593, 0.0, 721.5377, 172.854, 0.0, 0.0, 1.0);
    // Call engine
    try {
        net_ptr_->infer(cv_image->image, buffers, max_batch_size, res_image, intrinsic);
    } catch (std::exception & e) {
        NODELET_ERROR("%s", e.what());
        return;
    }

    // cv::imshow("laneatt_trt", res_image);
    // cv::waitKey(10);
    sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", res_image).toImageMsg();
    output_segment_pub_.publish(output_msg);
    const auto exe_end_time = high_resolution_clock::now();
    const double exe_time =
            std::chrono::duration_cast<milliseconds>(exe_end_time - exe_start_time).count();
    std::cout << "FPS: " << (1000.f / exe_time) << std::endl;
}


} // namespace lane_detection

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(lane_detection::TensorrtSegmentNodelet, nodelet::Nodelet)
