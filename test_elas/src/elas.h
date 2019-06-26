#ifndef SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_ELAS_H_
#define SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_ELAS_H_

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <emmintrin.h>
#include <stdint.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Elas
{

public:
    // parameter settings
    struct parameters
    {
        int disp_min = 10;               // min disparity
        int disp_max = 255;              // max disparity
        float support_threshold = 0.95;  // max. uniqueness ratio (best vs. second best support match)
        int support_texture = 10;        // min texture for support points
        int candidate_stepsize = 5;      // step size of regular grid on which support points are matched
        int incon_window_size = 5;       // window size of inconsistent support point check
        int incon_threshold = 10;         // disparity similarity threshold for support point to be considered consistent
        int incon_min_support = 5;       // minimum number of consistent support points
        bool add_corners = 0;            // add support points at image corners with nearest neighbor disparities
        int grid_size = 20;              // size of neighborhood for additional support point extrapolation
        float beta = 0.02;               // image likelihood parameter
        float gamma = 3;                 // prior constant
        float sigma = 1;                 // prior sigma
        float sradius = 2;               // prior sigma radius
        int match_texture = 1;           // min texture for dense matching
        int lr_threshold = 2;            // disparity threshold for left/right consistency check
        float speckle_sim_threshold = 1; // similarity threshold for speckle segmentation
        int speckle_size = 200;          // maximal size of a speckle (small speckles get removed)
        int ipol_gap_width = 3;          // interpolate small gaps (left<->right, top<->bottom)
        bool filter_median = 0;          // optional median filter (approximated)
        bool filter_adaptive_mean = 1;   // optional adaptive mean filter (approximated)
        bool postprocess_only_left = 1;  // saves time by not postprocessing the right image
        bool subsampling = 0;            // saves time by only computing disparities for each 2nd pixel
        // note: for this option D1 and D2 must be passed with size
        //       width/2 x height/2 (rounded towards zero)
    };

    // constructor, input: parameters
    Elas(parameters param) : param_(param) {}

    // deconstructor
    ~Elas() {}
    bool process(const Mat &image_left, const Mat &image_right, Mat &disparity_left, Mat &disparity_right);

private:
void SaveSupportPoints();
    vector<Point3i> ComputeSupportMatches();
    int ComputeMatchingDisparity(const int &u, const int &v, const bool &right_image);
    int ComputeSAD();
    void RemoveInconsistentSupportPoints(Mat &D_can);
    vector<Vec6f> ComputeDelaunayTriangulation(const bool &right_image);

    vector<Vec6f> ComputeDisparityPlanes(const vector<Vec6f> &triangulate_points, vector<Vec3d> &triangulate_d);
    void CreateGrid(Mat &disparity_grid, const Size &disparity_grid_size, const bool &right_image);
    void ComputeDisparity(const vector<Vec6f> &triangulate_points, const vector<Vec6f> &triangulate_params, const Mat &disparity_grid,
                          const Size &disparity_grid_size, const bool &right_image, const vector<Vec3d> &triangulate_d, Mat &disparity);
    void findMatch(const int32_t &u, const int32_t &v, const float &plane_a, const float &plane_b, const float &plane_c, const Mat &disparity_grid,
                   const Size &disparity_grid_size, int32_t *P, const int32_t &plane_radius, const bool &valid, const bool &right_image, Mat &disparity);
    void RemoveRedundantSupportPoints(Mat &D_can, int32_t redun_max_dist, int32_t redun_threshold, bool vertical);

    // parameter set
    parameters param_;

    // memory aligned input images + dimensions
    int width_, height_;
    Mat descriptor_left_, descriptor_right_;
    vector<Point3i> support_points_;
    Mat image_left_, image_right_;
};

#endif // SINEVA_AUTOWARE_ROS_SRC_COMPUTING_PERCEPTION_LOCALIZATION_PACKAGES_SINEVA_STEREO_INCLUDE_ELAS_H_
