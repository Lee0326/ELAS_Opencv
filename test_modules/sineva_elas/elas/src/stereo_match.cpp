#include "stereo_match.h"

cv::Mat StereoMatch::run(const cv::Mat &image_left, const cv::Mat &image_right)
{
    if (image_left.empty() || image_right.empty())
    {
        std::cout << "Image is empty" << std::endl;
        exit(0);
    }
    if (image_left.channels() > 1 || image_right.channels() > 1)
    {
        std::cout << "Images must be gray image." << std::endl;
        exit(0);
    }

    Elas elas(mParam);

    int width = image_left.cols;
    int height = image_left.rows;
    const int32_t dims[3] = {width, height, width}; // bytes per line = width

    cv::Mat disparity_left(height, width, CV_32F, Scalar(0));
    cv::Mat disparity_right(height, width, CV_32F, Scalar(0));

    bool valid_depth = elas.process(image_left, image_right, disparity_left, disparity_right);

    if (!valid_depth)
        return cv::Mat();
    return disparity_left;
}

cv::Mat StereoMatch::convertDisparityToDepth(const cv::Mat &disp, const float &baseline, const float &fx)
{
    cv::Mat depthMap = cv::Mat(disp.size(), CV_16U);
    for (int i = 0; i < disp.rows; i++)
    {
        for (int j = 0; j < disp.cols; j++)
        {
            double d = static_cast<double>(disp.at<float>(i, j));
            if (d < 1)
                depthMap.at<unsigned short>(i, j) = 0;
            else
                depthMap.at<unsigned short>(i, j) = 1000 * (baseline * fx) / d;

            if (std::isnan(depthMap.at<unsigned short>(i, j)) || std::isinf(depthMap.at<unsigned short>(i, j)))
                depthMap.at<unsigned short>(i, j) = 0;
        }
    }
    return depthMap;
}