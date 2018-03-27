#include <cmath>
#include <iostream>
#include <fstream>
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
#include <mdl_people_tracker/TrackedPersons2d.h>
using namespace mdl_people_tracker;
#endif
#ifdef SPENCER_FRAMEWORK
#include <spencer_tracking_msgs/TrackedPersons2d.h>
using namespace spencer_tracking_msgs;
#endif

extern "C" int mkpath(const char *path);
void subtractbg(cv::Mat &rgb, const cv::Mat &d, float fgdepth, float thresh=1.0f);
float depthpercentile(const cv::Mat &m, double percentile);
void dump_32FC1(std::string fname, cv::Mat img);

// These are set by params to the node.
std::string g_dir;
double g_hfactor;
double g_wfactor;
bool g_subbg;
double g_bgpercentile;

// For filename and stats.
size_t g_counter = 0;


template<typename T>
std::string to_s(const T& v, int zeropadw=0)
{
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(zeropadw) << v;
    return oss.str();
}


void cb(const TrackedPersons2d::ConstPtr& t2d, const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& d)
{
    size_t ndet = t2d->boxes.size();
    if(ndet == 0)
        return;

    cv_bridge::CvImageConstPtr cv_rgb;
    cv_bridge::CvImageConstPtr cv_d;
    try {
        // Careful, the above is RGB but opencv "thinks" in BGR, so need to copy!
        //cv_rgb = cv_bridge::toCvShare(rgb);
        cv_rgb = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
        cv_d = cv_bridge::toCvShare(d);
    } catch(const cv_bridge::Exception& e) {
        ROS_ERROR("Couldn't convert image: %s", e.what());
        return;
    }

    for(size_t idet = 0 ; idet < ndet ; ++idet) {
        const TrackedPerson2d& p2d = t2d->boxes[idet];
        // Compute this detection's cut-out boxes.
        uint32_t h = g_hfactor > 0 ? uint32_t(round(g_hfactor * p2d.w)) //std::min(uint32_t(g_hfactor * p2d.w), p2d.h)
                                   : p2d.h;
        uint32_t w = uint32_t(round(g_wfactor * p2d.w));
        int32_t x = p2d.x + int32_t(round(.5*(1. - g_wfactor)*p2d.w));
        cv::Rect bbox(x, p2d.y, w, h);
        cv::Rect bbox_rgb = bbox & cv::Rect(0, 0, cv_rgb->image.cols, cv_rgb->image.rows);
        cv::Rect bbox_d = bbox & cv::Rect(0, 0, cv_d->image.cols, cv_d->image.rows);

        // Cut-out and optionally subtract background.
        cv::Mat rgbimg(cv_rgb->image, bbox_rgb);
        cv::Mat dimg(cv_d->image, bbox_d);
        if(g_subbg) {
            // TODO: Compute the depth percentile on the actual body box,
            //       not the one multiplied by the factors.
            subtractbg(rgbimg, dimg, depthpercentile(dimg, g_bgpercentile));
        }

        // Save.
        std::string fname = g_dir + "/" + to_s(p2d.track_id, 4) + "_" + to_s(rgb->header.seq, 8);
        if(!cv::imwrite(fname + "_rgb.png", rgbimg)) {
            ROS_ERROR("Error writing image %s", (fname + "_rgb.png").c_str());
        }

        dump_32FC1(fname + "_d.csv", dimg);

        std::cout << "\rDump #" << g_counter << ": track " << p2d.track_id << "@seq" << rgb->header.seq << std::flush;
        g_counter++;
    }
}


void cb_nodepth(const TrackedPersons2d::ConstPtr& t2d, const sensor_msgs::ImageConstPtr& rgb)
{
    size_t ndet = t2d->boxes.size();
    if(ndet == 0)
        return;

    cv_bridge::CvImageConstPtr cv_rgb;
    try {
        // TODO: Careful, the above is RGB but opencv "thinks" in BGR!
        cv_rgb = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
    } catch(const cv_bridge::Exception& e) {
        ROS_ERROR("Couldn't convert image: %s", e.what());
        return;
    }

    for(size_t idet = 0 ; idet < ndet ; ++idet) {
        const TrackedPerson2d& p2d = t2d->boxes[idet];
        // Compute this detection's cut-out boxes.
        uint32_t h = g_hfactor > 0 ? uint32_t(round(g_hfactor * p2d.w)) //std::min(uint32_t(g_hfactor * p2d.w), p2d.h)
                                   : p2d.h;
        uint32_t w = uint32_t(round(g_wfactor * p2d.w));
        int32_t x = p2d.x + int32_t(round(.5*(1. - g_wfactor)*p2d.w));
        cv::Rect bbox(x, p2d.y, w, h);
        cv::Rect bbox_rgb = bbox & cv::Rect(0, 0, cv_rgb->image.cols, cv_rgb->image.rows);

        // Cut-out and optionally subtract background.
        cv::Mat rgbimg(cv_rgb->image, bbox_rgb);

        // Save.
        // TODO: setw
        std::string bbox_str = to_s(p2d.x) + "_" + to_s(p2d.y) + "_" + to_s(p2d.w) + "_" + to_s(p2d.h);
        std::string fname = g_dir + "/" + to_s(p2d.track_id) + "_" + to_s(rgb->header.seq) + "_" + bbox_str;
        if(!cv::imwrite(fname + "_rgb.png", rgbimg)) {
            ROS_ERROR("Error writing image %s", (fname + "_rgb.png").c_str());
        }

        std::cout << "\rDump #" << g_counter << ": track " << p2d.track_id << "@seq" << rgb->header.seq << std::flush;
        g_counter++;
    }
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "dump_tracks");
    ros::NodeHandle nh;

    ros::NodeHandle nh_("~");
    nh_.param("dir", g_dir, std::string("."));
    nh_.param("hfactor", g_hfactor, -1.0);
    nh_.param("wfactor", g_wfactor,  1.0);
    nh_.param("bgpercentile", g_bgpercentile, 0.5);
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

    ROS_INFO("Dumping %s%s-bodies into %s", g_subbg ? "bg-subtracted, " : " ", g_hfactor > 0 ? to_s(g_hfactor).c_str() : "full", g_dir.c_str());

    std::string topic_t2d, topic_rgb, topic_d;
    nh_.param("track2d", topic_t2d, std::string("/upper_body_detector/detections"));
    nh_.param("rgb", topic_rgb, std::string("/head_xtion/rgb/image_rect_color"));
    nh_.param("d", topic_d, std::string("/head_xtion/depth/image_rect_meters"));

    message_filters::Subscriber<TrackedPersons2d> sub_t2d(nh_, topic_t2d.c_str(), 1);

    image_transport::ImageTransport it(nh_);
    image_transport::SubscriberFilter sub_rgb(it, topic_rgb.c_str(), 1);

    // Run either with or without depth. This is a little ugly.
    if(!topic_d.empty()) {
        image_transport::SubscriberFilter sub_d(it, topic_d.c_str(), 1);

        typedef message_filters::sync_policies::ApproximateTime<TrackedPersons2d, sensor_msgs::Image, sensor_msgs::Image> SyncType;
        const SyncType sync_policy(20);

        message_filters::Synchronizer<SyncType> sync(sync_policy, sub_t2d, sub_rgb, sub_d);
        sync.registerCallback(boost::bind(&cb, _1, _2, _3));

        ros::spin();
    } else {
        typedef message_filters::sync_policies::ApproximateTime<TrackedPersons2d, sensor_msgs::Image> SyncType;
        const SyncType sync_policy(20);

        message_filters::Synchronizer<SyncType> sync(sync_policy, sub_t2d, sub_rgb);
        sync.registerCallback(boost::bind(&cb_nodepth, _1, _2));

        ros::spin();
    }

    // TODO: ROS_INFO isn't being output after the spin is done?
    //ROS_INFO("Dumped a total of %d track frames.", g_counter);
    std::cout << "Dumped a total of " << g_counter << " track frames." << std::endl;
    return 0;
}
