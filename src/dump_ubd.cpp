#include <iostream>
#include <sstream>

#include <wordexp.h>

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

extern "C" int mkpath(const char *path);
void subtractbg(cv::Mat &rgb, const cv::Mat &d, float fgdepth, float thresh=1.0f);
void dump_32FC1(std::string fname, cv::Mat img);

// These are set by params to the node.
std::string g_dir;
bool g_fullbody;
double g_hfactor;
double g_wfactor;

// For background-subtraction.
bool g_subbg;

// For the sticky case, keep the last seen box in mind.
bool g_sticky;
UpperBodyDetector::ConstPtr g_last_ubd;

// For filename and stats.
size_t g_counter = 0;


template<typename T>
std::string to_s(const T& v, int zeropadw=0)
{
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(zeropadw) << v;
    return oss.str();
}


void cb(const UpperBodyDetector::ConstPtr& ubd, const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& d)
{
    size_t ndet = ubd->pos_x.size();
    if((ndet > 0 && g_sticky) || !g_last_ubd)
        g_last_ubd = ubd;

    ndet = g_last_ubd->pos_x.size();
    if(ndet == 0)
        return;

    cv_bridge::CvImageConstPtr cv_rgb;
    cv_bridge::CvImageConstPtr cv_d;
    try {
        // We need to copy instead of share as the above is RGB but opencv "thinks" in BGR!
        //cv_rgb = cv_bridge::toCvShare(rgb);
        cv_rgb = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
        cv_d = cv_bridge::toCvShare(d);
    } catch(const cv_bridge::Exception& e) {
        ROS_ERROR("Couldn't convert image: %s", e.what());
        return;
    }

    for(size_t idet = 0 ; idet < g_last_ubd->pos_x.size() ; ++idet) {
        uint32_t h = g_hfactor > 0 ? uint32_t(g_hfactor * g_last_ubd->height[idet])
                                   : g_last_ubd->height[idet];
        uint32_t w = uint32_t(g_wfactor * g_last_ubd->width[idet]);
        cv::Rect bbox(g_last_ubd->pos_x[idet], g_last_ubd->pos_y[idet], w, h);
        cv::Rect bbox_rgb = bbox & cv::Rect(0, 0, cv_rgb->image.cols, cv_rgb->image.rows);
        cv::Rect bbox_d = bbox & cv::Rect(0, 0, cv_d->image.cols, cv_d->image.rows);

        // Cut-out and optionally subtract background.
        cv::Mat rgbimg(cv_rgb->image, bbox_rgb);
        cv::Mat dimg(cv_d->image, bbox_d);
        if(g_subbg)
            subtractbg(rgbimg, dimg, g_last_ubd->median_depth[idet]);

        std::string fname = g_dir + "/" + to_s(rgb->header.seq, 8) + "_" + to_s(idet);
        //std::string fname = g_dir + "/" + to_s(bbox.x, 4) + "," + to_s(bbox.y, 4) + "_" + to_s(idet) + "_" + to_s(rgb->header.seq);
        if(!cv::imwrite(fname + "_rgb.png", cv::Mat(cv_rgb->image, bbox_rgb))) {
            ROS_ERROR("Error writing image %s", (fname + "_rgb.png").c_str());
        }

        dump_32FC1(fname + "_d.csv", dimg);

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
    nh_.param("sticky", g_sticky, false);
    nh_.param("hfactor", g_hfactor, -1.0);
    nh_.param("wfactor", g_wfactor,  1.0);
    g_wfactor = g_wfactor <= 0 ? 1.0 : g_wfactor;
    nh_.param("subbg", g_subbg, false);

    // equivalend to Python's expanduser(dir)
    wordexp_t exp_result;
    if(0 !=wordexp(g_dir.c_str(), &exp_result, WRDE_NOCMD | WRDE_SHOWERR))
        return 1;
    g_dir = exp_result.we_wordv[0];
    wordfree(&exp_result);

    if(mkpath(g_dir.c_str()) != 0) {
        perror("Couldn't create output directory");
        return 2;
    }
    // TODO: warn if non-emtpy?

    ROS_INFO("Dumping %s%s%s-bodies into %s",
             g_sticky ? "sticky " : "",
             g_subbg ? "bg-subtracted, " : "",
             g_hfactor > 0 ? to_s(g_hfactor).c_str() : "upper",
             g_dir.c_str());

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
