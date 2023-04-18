/*
 * @Description: 
 * @Author: ubuntu
 * @Date: 2022/11/16 下午4:06
 * @LastEditors: ubuntu
 * @LastEditTime: 2022/11/16 下午4:06
 * @Version 1.0
 */

#include <nodelet/loader.h>
#include <ros/ros.h>

int main(int argc, char ** argv)
{
    ros::init(argc, argv, "smoke_node");
    ros::NodeHandle private_nh("~");

    nodelet::Loader nodelet;
    nodelet::M_string remap(ros::names::getRemappings());
    nodelet::V_string nargv;
    std::string nodelet_name = ros::this_node::getName();
    nodelet.load(
            nodelet_name, "lane_detection/smoke", remap, nargv);

    ros::spin();
    return 0;
}