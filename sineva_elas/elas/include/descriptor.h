#ifndef SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_DESCRIPTOR_H_
#define SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_DESCRIPTOR_H_

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>

using namespace cv;

class Descriptor
{

public:
  Descriptor() = delete;
  // deconstructor releases memory
  ~Descriptor();
  
  // constructor creates filters
  Descriptor(const Mat &image, const int &width, const int &height, const bool &subsampling);

  Mat CreateDescriptor();

private:
  Mat descriptor;
  Mat grad_x, grad_y;
  int width_, height_;
  bool subsampling_;
};

#endif // SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_DESCRIPTOR_H_
