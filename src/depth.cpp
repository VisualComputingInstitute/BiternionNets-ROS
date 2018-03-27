#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <ros/ros.h>

void dump_32FC1(std::string fname, cv::Mat img)
{
    if(img.type() != CV_32FC1) {
        ROS_ERROR("Image not of 32FC1 type, but %d", img.type());
        return;
    }

    std::ofstream of(fname.c_str());
    if(!of) {
        ROS_ERROR("Error writing depth image %s", (fname + "_d.csv").c_str());
        return;
    }

    for(size_t iy = 0 ; iy < img.rows ; ++iy) {
        float *py = img.ptr<float>(iy);
        for(size_t ix = 0 ; ix < img.cols ; ++ix, ++py) {
            of << *py;
            if(ix + 1 < img.cols)
                of << " ";
        }
        of << std::endl;
    }
}
