#include <emmintrin.h>

#include "descriptor.h"

using namespace std;

/**
 * @brief 
 * 
 */
Descriptor::Descriptor(const Mat &img, const int &width, const int &height, const bool &subsampling)
    : width_(width), height_(height), subsampling_(subsampling)
{
  Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
  Sobel(img, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
}

Descriptor::~Descriptor()
{
}

Mat Descriptor::CreateDescriptor()
{
  Mat descriptor(width_ * height_, 16, CV_16S); //这个尺寸是否偏大，(width_-6)*(height_-4) 

  int step = grad_x.step;
  for (int v = 3; v < height_ - 2; ++v)
  {
    const short *data_x = grad_x.ptr<short>(v); //sobel_x中指向第ｖ行的第一个元素的指针
    const short *data_y = grad_y.ptr<short>(v); //sobel_y中指向第ｖ行的第一个元素的指针

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

      short *data_desc = descriptor.ptr<short>((v) * (width_) + u);
      data_desc[0] = data_x[u - 2 * step];
      data_desc[1] = data_x[u - step - 2];
      data_desc[2] = data_x[u - step];
      data_desc[3] = data_x[u - step + 2];
      data_desc[4] = data_x[u - 1];
      data_desc[5] = data_x[u];
      data_desc[6] = data_x[u];
      data_desc[7] = data_x[u + 1];
      data_desc[8] = data_x[u + step - 2];
      data_desc[9] = data_x[u + step];
      data_desc[10] = data_x[u + step + 2];
      data_desc[11] = data_x[u + 2 * step];
      data_desc[12] = data_y[u - step];
      data_desc[13] = data_y[u - 1];
      data_desc[14] = data_y[u + 1];
      data_desc[15] = data_y[u + step];
      // cout << (int)data_desc[0] << " " << (int)data_desc[1] << " " << (int)data_desc[2] << " " << (int)data_desc[3] << " "
      //      << (int)data_desc[4] << " " << (int)data_desc[5] << " " << (int)data_desc[6] << " " << (int)data_desc[7] << " "
      //      << (int)data_desc[8] << " " << (int)data_desc[9] << " " << (int)data_desc[10] << " " << (int)data_desc[11] << " "
      //      << (int)data_desc[12] << " " << (int)data_desc[13] << " " << (int)data_desc[14] << " " << (int)data_desc[15] << " " << endl;
    }
  }
  return descriptor.clone();
}
