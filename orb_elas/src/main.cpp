#include <unistd.h>
#include <sys/time.h>
#include <dirent.h>
#include <string>
#include <cv.hpp>
#include <opencv2/imgcodecs.hpp>
//#include "stereo_match.h"
//#include "stereo_rectify.h"
#define BOOST_ENABLE_ASSERT_HANDLER
#include <boost/format.hpp>
#include <boost/assert.hpp>
#include <algorithm>
#include <math.h>
#include "descriptor.h"
#include "elas.h"
#include "boost/progress.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using boost::format;
using cv::cvtColor;
using cv::FileStorage;
using cv::imread;
using cv::Mat;
using cv::Size;
using std::cout;
using std::endl;
using std::string;
using std::vector;


int main(int argc, char *argv[])
{
    string data_dir = "/home/colin/catkin_modelas_ws/src/ls_elas/img/";
    //string left_image = argv[1];
    string left_image = "im2.png"; 
    string right_image = "im6.png";
    // string right_image = argv[2];
    Mat left = imread(data_dir + left_image);
    Mat right = imread(data_dir +  right_image);
    int width_ = left.cols;
    int height_ = left.rows;		
    cv::Mat left_gray, right_gray;
    Mat disparity1(height_, width_, CV_32F);
    Mat disparity2(height_, width_, CV_32F);


    if (left.channels()>1)
    {
        cvtColor(left, left_gray, CV_BGR2GRAY);
        cvtColor(right, right_gray, CV_BGR2GRAY);        
    }
    else 
    {
        left_gray = left.clone();
        right_gray = right.clone();
    }

    Mat image_left = left_gray.clone();
    Mat image_right = right_gray.clone();
    // create a modified elas class 
    Elas::parameters param;
    // param.postprocess_only_left = false;
    Elas elas(param);
    elas.process(image_left, image_right, disparity1, disparity2);
    imwrite("disparity.pgm", disparity1);
    // test the function of descriptor
    //test_descriptor(left_gray, right_gray, width_, height_);
    return 0;
}
