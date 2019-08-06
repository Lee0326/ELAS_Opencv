#include <emmintrin.h>

#include "descriptor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

/**
 * @brief 
 * 
 */
Descriptor::Descriptor(const Mat &img, const int &width, const int &height, const bool &subsampling)
    : width_(width), height_(height), subsampling_(subsampling), image(img.clone())
{
  Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
  Sobel(img, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
  grad_x += 128;
  grad_x.convertTo(grad_x, CV_8U);
  blur(grad_x, grad_x, Size(3, 3));
  grad_y += 128;
  grad_y.convertTo(grad_y, CV_8U);
  blur(grad_y, grad_y, Size(5, 5));
  // imshow("A", grad_x);
  // waitKey();
}

Descriptor::~Descriptor()
{
}

// Mat Descriptor::CreateDescriptor()
// {
//   vector<KeyPoint> edge_points;
//   for (int u = 0; u < grad_x.cols; u += 10)
//   {
//     for (int v = 0; v < grad_x.rows; v += 10)
//     {
//       if (grad_x.at<uchar>(v, u) > 20)
//       {
//         edge_points.push_back(KeyPoint(Point2f(u, v), 16));
//       }
//     }
//   }
//   drawKeypoints(image, edge_points, image, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
//   imshow("output", image);
//   waitKey();
//   Ptr<BriefDescriptorExtractor> detector = BriefDescriptorExtractor::create();
//   Mat descriptors_1;
//   detector->compute(image, edge_points, descriptors_1);
//   cout << edge_points.size() << endl;
//   return Mat();
// }

Mat Descriptor::CreateDescriptor()
{
  Mat descriptor(width_ * height_, 16, CV_8U, Scalar(0));
  int width_step = descriptor.step[1], height_step = descriptor.step[0];
  int step = grad_x.step;
  for (int v = 2; v < height_ - 2; ++v)
  {
    const uchar *data_x = grad_x.ptr<uchar>(v);
    const uchar *data_y = grad_y.ptr<uchar>(v);

    for (int32_t u = 2; u < width_ - 2; ++u)
    {
      // descriptor.at<short>(v * 16 + u, 0) = data_x[u - 2 * step];
      // descriptor.at<short>(v * 16 + u, 1) = data_x[u - step - 2];
      // descriptor.at<short>(v * 16 + u, 2) = data_x[u - step];
      // descriptor.at<short>(v * 16 + u, 3) = data_x[u - step + 2];
      // descriptor.at<short>(v * 16 + u, 4) = data_x[u - 1];
      // descriptor.at<short>(v * 16 + u, 5) = data_x[u];
      // descriptor.at<short>(v * 16 + u, 6) = data_x[u];
      // descriptor.at<short>(v * 16 + u, 7) = data_x[u + 1];
      // descriptor.at<short>(v * 16 + u, 8) = data_x[u + step - 2];
      // descriptor.at<short>(v * 16 + u, 9) = data_x[u + step];
      // descriptor.at<short>(v * 16 + u, 10) = data_x[u + step + 2];
      // descriptor.at<short>(v * 16 + u, 11) = data_x[u + 2 * step];
      // descriptor.at<short>(v * 16 + u, 12) = data_y[u - step];
      // descriptor.at<short>(v * 16 + u, 13) = data_y[u - 1];
      // descriptor.at<short>(v * 16 + u, 14) = data_y[u + 1];
      // descriptor.at<short>(v * 16 + u, 15) = data_y[u + step];

      uchar *data_desc = descriptor.ptr<uchar>(v * width_ + u);
      data_desc[0] = (int)data_x[u - 2 * step];
      data_desc[1] = (int)data_x[u - step - 2];
      data_desc[2] = (int)data_x[u - step];
      data_desc[3] = (int)data_x[u - step + 2];
      data_desc[4] = (int)data_x[u - 1];
      data_desc[5] = (int)data_x[u];
      data_desc[6] = (int)data_x[u];
      data_desc[7] = (int)data_x[u + 1];
      data_desc[8] = (int)data_x[u + step - 2];
      data_desc[9] = (int)data_x[u + step];
      data_desc[10] = (int)data_x[u + step + 2];
      data_desc[11] = (int)data_x[u + 2 * step];
      data_desc[12] = (int)data_y[u - step];
      data_desc[13] = (int)data_y[u - 1];
      data_desc[14] = (int)data_y[u + 1];
      data_desc[15] = (int)data_y[u + step];
    }
  }

  // for (int i = 0; i < width_; i++)
  // {
  // uchar *data_desc = grad_x.ptr<uchar>(360);
  // cout << i << endl;
  // cout << (int)data_desc[0] << " " << (int)data_desc[1] << " " << (int)data_desc[2] << " " << (int)data_desc[3] << " "
  //      << (int)data_desc[4] << " " << (int)data_desc[5] << " " << (int)data_desc[6] << " " << (int)data_desc[7] << " "
  //      << (int)data_desc[8] << " " << (int)data_desc[9] << " " << (int)data_desc[10] << " " << (int)data_desc[11] << " "
  //      << (int)data_desc[12] << " " << (int)data_desc[13] << " " << (int)data_desc[14] << " " << (int)data_desc[15] << " " << endl;
  // }
  return descriptor.clone();
}
