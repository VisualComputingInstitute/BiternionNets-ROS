#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

float mediandepth(const cv::Mat &m, double bgcoeff)
{
    std::vector<float> depths;
    depths.reserve(m.cols*m.rows);

    // Collect all non-nan values.
    for(size_t y = 0 ; y < m.rows ; ++y) {
        const float *pix = m.ptr<float>(y);
        for(size_t x = 0 ; x < m.cols ; ++x, ++pix) {
            if(!isnan(*pix)) {
                depths.push_back(*pix);
            }
        }
    }

    // Sort the first half.
    std::nth_element(depths.begin(), depths.begin() + depths.size() * bgcoeff, depths.end());

    // And there we can access the median.
    return depths[depths.size() * bgcoeff];
}

void subtractbg(cv::Mat &rgb, const cv::Mat &d, float thresh, float bgcoeff)
{
    // We require this, otherwise there's no correspondence!
    // TODO: Fail better.
    if(rgb.rows != d.rows || rgb.cols != d.cols) {
        std::cerr << "Oops from `subtractbg`." << std::endl;
        return;
    }

    float md = mediandepth(d, bgcoeff);

    for(size_t y = 0 ; y < d.rows ; ++y) {
        const float *dpix = d.ptr<float>(y);
        cv::Vec3b *rgbpix = rgb.ptr<cv::Vec3b>(y);
        for(size_t x = 0 ; x < d.cols ; ++x, ++dpix, ++rgbpix) {
            // Replace pixels whose depth is too far or nan with 0.
            if(md+thresh < *dpix || isnan(*dpix)) {
                (*rgbpix)[0] = (*rgbpix)[1] = (*rgbpix)[2] = 0;
            }
        }
    }
}
