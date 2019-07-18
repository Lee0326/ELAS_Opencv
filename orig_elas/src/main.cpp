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
#include <boost/filesystem.hpp>
#include <boost/assert.hpp>
#include <algorithm>
#include <math.h>
#include "descriptor.h"
#include "elas.h"
#include "boost/progress.hpp"
#include <fstream>
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
int command(const char *s)
{
    char str[100];
    strcpy(str, s);
    return system(strcat(str, " >> test.txt "));
}

void writeCSV(cv::Mat m, std::string filename)
{
   ofstream myfile;
   myfile.open(filename.c_str());
   myfile<< cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
   myfile.close();
   // cout << "Depth Images Saved!" << endl;
}
Mat convertDisparityToDepth(const Mat &disparityMap, const float baseline, const float fx) {
    Mat depthMap = Mat(disparityMap.size(), CV_32F);
    for (int i = 0; i < disparityMap.rows; i++) {
        for (int j = 0; j < disparityMap.cols; j++) {
            /*if (disparityMap.at<short>(i, j) <= 2)
                depthMap.at<double>(i, j) = 0;
            else */
            double d = static_cast<double>(disparityMap.at<float>(i, j));

            depthMap.at<float>(i, j) = (baseline * fx) / d;

            if (d < 5)
                depthMap.at<float>(i, j) = 0;
           

            if(d <= 0)
                depthMap.at<float>(i, j) = 0;

            if (std::isnan(depthMap.at<float>(i, j)) || std::isinf(depthMap.at<float>(i, j)))
                depthMap.at<float>(i, j) = 0;
        }
    }

    return depthMap;
}
vector<string> GetDirFile(const string dirPath, const char *extenStr)
{
  vector<string> files_name;
  DIR *dir = opendir(dirPath.c_str());
  dirent *pDirent = NULL;
  while ((pDirent = readdir(dir)) != NULL)
  {
    if (strstr(pDirent->d_name, extenStr))
    {
      files_name.push_back(string(pDirent->d_name));
    }
  }
  closedir(dir);
  return files_name;
}

int main(int argc, char *argv[])
{
    // process all the images in the /img/ folder
    string format_input = "jpg";
    string data_dir = "/home/lee/ELAS_Opencv/orig_elas/img/";
    
    if (argc==2 && !strcmp(argv[1], "demo")) 
    {
        float fx = 217.1;
        float baseline = 0.119170950703722;
        vector<string> all_left_files = GetDirFile(data_dir + "/left/", format_input.c_str());
        sort(all_left_files.begin(), all_left_files.end());
        // cout << "Load " << all_left_files.size() << " left images from " << data_dir << endl;
        vector<string> all_right_files = GetDirFile(data_dir + "/right/", format_input.c_str());
        sort(all_right_files.begin(), all_right_files.end());
        // cout << "Load " << all_right_files.size() << " right images from " << data_dir << endl;
        for (int i=0; i < all_left_files.size(); ++i)
        {
            Mat left = imread(data_dir + "/left/" + all_left_files[i]);
            Mat right = imread(data_dir + "/right/" + all_right_files[i]);
            int width_ = left.cols;
            int height_ = left.rows;
            const int32_t dims[3] = {width_,height_,width_}; // bytes per line = width
            Mat disparity1(height_, width_, CV_32F);
            Mat disparity2(height_, width_, CV_32F);
            Mat depth1(height_, width_, CV_32F);
            Mat depth2(height_, width_, CV_32F);
            Mat left_gray, right_gray;
            if (left.channels() > 1)
            {
                cvtColor(left, left_gray, CV_BGR2GRAY);
                cvtColor(right, right_gray, CV_BGR2GRAY);
            }
            else
            {
                left_gray = left.clone();
                right_gray = right.clone();
            }
            Elas::parameters param;
            Elas elas(param);
            // cout << "left image: " << all_left_files[i] << endl;
            // cout << "right image: " << all_right_files[i] << endl; 
            elas.process(left_gray, right_gray, (float *)disparity1.data, (float *)disparity2.data,dims);
            all_left_files[i];
            string str_file = all_left_files[i].substr(10,18);
            string depth_cal_name = "img/depth/DepthImage_CL_" + str_file + "csv";
            // cout<< depth_cal_name <<endl;
            depth1 = convertDisparityToDepth(disparity1, baseline, fx);
            writeCSV(disparity1, depth_cal_name);
            
        }    
    }else if (argc == 3)
    {
        string left_image = argv[1];
        string right_image = argv[2];
        Mat left = imread(data_dir + left_image);
        Mat right = imread(data_dir +  right_image);
        int width_ = left.cols;
        int height_ = left.rows;
        const int32_t dims[3] = {width_,height_,width_}; // bytes per line = width
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
        param.postprocess_only_left = false;
        Elas elas(param);

        elas.process(left_gray, right_gray, (float *)disparity1.data, (float *)disparity2.data,dims);
        //Descriptor descriptor_left(image_left, width_, height_, param.subsampling);
        //Descriptor descriptor_right(image_right, width_, height_, param.subsampling);
        //elas.descriptor_left_ = descriptor_left.CreateDescriptor();
        //elas.descriptor_right_ = descriptor_right.CreateDescriptor();
        //vector<Point3i> support_points = elas.ComputeSupportMatches();
        imwrite("disparity.pgm", disparity1);
    } else
    { 
        cout << endl;
        cout << "ELAS demo program usage: " << endl;
        cout << "./elas demo ................ process all test images (image dir)" << endl;
        cout << "./elas left.jpg right.jpg .. process a single stereo pair" << endl;
        cout << "./elas -h .................. shows this help" << endl;
        cout << endl;
        cout << "Note: disparities will be scaled such that disp_max = 255." << endl;
        cout << endl;
    }
    // command("./test_ori_elas demo");
    return 0;
}
