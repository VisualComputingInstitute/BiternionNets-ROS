#include <iostream>
#include <sstream>

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#ifdef STRANDS_FRAMEWORK
#include <upper_body_detector/UpperBodyDetector.h>
using namespace upper_body_detector;
#endif
#ifdef SPENCER_FRAMEWORK
#include <rwth_perception_people_msgs/UpperBodyDetector.h>
using namespace rwth_perception_people_msgs;
#endif

// These are set by params to the node.
std::string g_dir;
bool g_fullbody;

// For filename and stats.
size_t g_counter = 0;


template<typename T>
std::string to_s(const T& v)
{
    std::ostringstream oss;
    oss << v;
    return oss.str();
}


void cb(const UpperBodyDetector::ConstPtr& ubd, const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& d)
{
    size_t ndet = ubd->pos_x.size();
    if(ndet == 0)
        return;

    cv_bridge::CvImageConstPtr cv_rgb;
    cv_bridge::CvImageConstPtr cv_d;
    try {
        // TODO: Careful, the above is RGB but opencv "thinks" in BGR!
        //cv_rgb = cv_bridge::toCvShare(rgb);
        cv_rgb = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
        cv_d = cv_bridge::toCvShare(d);
    } catch(const cv_bridge::Exception& e) {
        ROS_ERROR("Couldn't convert image: %s", e.what());
        return;
    }

    for(size_t idet = 0 ; idet < ndet ; ++idet) {
        cv::Rect bbox(ubd->pos_x[idet], ubd->pos_y[idet], ubd->width[idet], ubd->height[idet] * (g_fullbody ? 3 : 1));
        cv::Rect bbox_rgb = bbox & cv::Rect(0, 0, cv_rgb->image.cols, cv_rgb->image.rows);
        cv::Rect bbox_d = bbox & cv::Rect(0, 0, cv_d->image.cols, cv_d->image.rows);

        // TODO: setw
        std::string fname = g_dir + "/" + to_s(g_counter);
        if(!cv::imwrite(fname + "_rgb.png", cv::Mat(cv_rgb->image, bbox_rgb))) {
            ROS_ERROR("Error writing image %s", (fname + "_rgb.png").c_str());
        }

        std::cerr << "\r" << g_counter << std::flush;
        g_counter++;
    }
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "dump_ubd");
    ros::NodeHandle nh;

    ros::NodeHandle nh_("~");
    nh_.param("dir", g_dir, std::string("."));
    nh_.param("fullbody", g_fullbody, false);

    // TODO: dir = abspath(expanduser(dir))

    ROS_INFO("Dumping %s-bodies into %s", g_fullbody ? "full" : "upper", g_dir.c_str());

    // TODO: create target directory, or warn if non-emtpy.

    std::string topic_ubd, topic_rgb, topic_d;
    nh_.param("ubd", topic_ubd, std::string("/upper_body_detector/detections"));
    nh_.param("rgb", topic_rgb, std::string("/head_xtion/rgb/image_rect_color"));
    nh_.param("d", topic_d, std::string("/head_xtion/depth/image_rect_meters"));

    message_filters::Subscriber<UpperBodyDetector> sub_ubd(nh_, topic_ubd.c_str(), 1);

    image_transport::ImageTransport it(nh_);
    image_transport::SubscriberFilter sub_rgb(it, topic_rgb.c_str(), 1);
    image_transport::SubscriberFilter sub_d(it, topic_d.c_str(), 1);

    typedef message_filters::sync_policies::ApproximateTime<UpperBodyDetector, sensor_msgs::Image, sensor_msgs::Image> SyncType;
    const SyncType sync_policy(20);

    message_filters::Synchronizer<SyncType> sync(sync_policy, sub_ubd, sub_rgb, sub_d);
    sync.registerCallback(boost::bind(&cb, _1, _2, _3));

    ros::spin();

    // TODO: This is never being reached. Sadface.
    //ROS_INFO("Dumped a total of %d detections.", g_counter);
    std::cerr << "Dumped a total of " << g_counter << " detections." << std::endl;
    return 0;
}
