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
vector<Point3i> ComputeSupportMatches()


int main(int argc, char *argv[])
{
    // process all the images in the /img/ folder
    string format_input = "jpg";
    string data_dir = "img/";
    
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

        }    
    }else if (argc == 3)
    {
        string left_image = argv[1];
        string right_image = argv[2];
        Mat left = imread(data_dir + left_image);
        Mat right = imread(data_dir +  right_image);
        int width_ = left.cols;
        int height_ = left.rows;
        cv::Mat left_gray, right_gray;

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
