/* Image Stitcher Implementation.
 * @file
 * @date 2023-12-24
 * @author Yaroslav Murenkov
 */

#include <cmath>
#include <vector>

#include "cvlib.hpp"

namespace cvlib
{
Stitcher::Stitcher(float ratio)
{
    this->detector = cv::AKAZE::create();
    this->matcher = cv::BFMatcher::create();
}

void Stitcher::initialize(cv::InputArray input)
{
    input.getMat().copyTo(this->dst);
}

void Stitcher::stitch(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat src;
    input.getMat().copyTo(src);

    this->detector->detectAndCompute(src, cv::noArray(), this->src_corners, this->src_descriptors);
    this->detector->detectAndCompute(this->dst, cv::noArray(), this->dst_corners, this->dst_descriptors);

    std::vector<std::vector<cv::DMatch>> matches;
    this->matcher->radiusMatch(this->src_descriptors, this->dst_descriptors, matches, 100);

    std::vector<cv::Point2f> src_kps;
    std::vector<cv::Point2f> dst_kps;
    for (const auto& match : matches)
        if (!match.empty())
        {
            src_kps.push_back(this->src_corners[match[0].queryIdx].pt);
            dst_kps.push_back(this->dst_corners[match[0].queryIdx].pt);
        }
    if ((src_kps.empty() || dst_kps.empty()))
        return;

    cv::Mat homography = cv::findHomography(src_kps, dst_kps, cv::RANSAC);

    const auto dst_size = cv::Size(dst.cols + src.cols, dst.rows);
    dst = cv::Mat(dst_size, CV_8U);
    cv::warpPerspective(src, dst, homography, dst.size(), cv::INTER_CUBIC);
    cv::Mat roi = cv::Mat(dst, cv::Rect(0, 0, dst.cols, dst.rows));

    dst.copyTo(output);
}
} // namespace cvlib
