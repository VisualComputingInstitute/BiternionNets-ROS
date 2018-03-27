#include <iostream>
#include <sstream>

#include <ros/ros.h>

#include <message_filters/subscriber.h>

#ifdef STRANDS_FRAMEWORK
#include <upper_body_detector/UpperBodyDetector.h>
using namespace upper_body_detector;
#endif
#ifdef SPENCER_FRAMEWORK
#include <rwth_perception_people_msgs/UpperBodyDetector.h>
using namespace rwth_perception_people_msgs;
#endif


size_t g_counter = 0;


void cb(const UpperBodyDetector::ConstPtr& ubd)
{
    g_counter += ubd->pos_x.size();

    std::cout << "\rDetections: " << g_counter << std::flush;
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "count_ubd");
    ros::NodeHandle nh;

    ros::NodeHandle nh_("~");

    std::string topic_ubd;
    nh_.param("ubd", topic_ubd, std::string("/upper_body_detector/detections"));

    message_filters::Subscriber<UpperBodyDetector> sub_ubd(nh_, topic_ubd.c_str(), 1);
    sub_ubd.registerCallback(boost::bind(&cb, _1));

    ros::spin();

    std::cout << std::endl
              << "Found a total of " << g_counter << " detections." << std::endl;
    return 0;
}

