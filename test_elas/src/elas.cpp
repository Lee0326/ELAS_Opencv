#include <algorithm>
#include <math.h>
#include <cassert>
#include "elas.h"
#include "descriptor.h"
#include "boost/progress.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

bool Elas::process(const Mat &image_left, const Mat &image_right, Mat &disparity_left, Mat &disparity_right)
{
    // get width, height and bytes per line
    width_ = image_left.cols;
    height_ = image_left.rows;
    image_left_ = image_left.clone();
    image_right_ = image_right.clone();
    // compute descriptor
    // cout << "Compute Descriptor" << endl;
    Descriptor descriptor_left(image_left, width_, height_);
    Descriptor descriptor_right(image_right, width_, height_);
    {
        boost::progress_timer t;
        descriptor_left_ = descriptor_left.CreateDescriptor();
        descriptor_right_ = descriptor_right.CreateDescriptor();
    }

    {
        boost::progress_timer t;
        cout << "Compute Support Matches" << endl;
        support_points_ = ComputeSupportMatches();
        cout << support_points_.size() << endl;
    }
SaveSupportPoints();

    cout << "Compute Triangulate" << endl;
    vector<Vec6f> triangulate_left, triangulate_right;
    {
        boost::progress_timer t;
        triangulate_left = ComputeDelaunayTriangulation(false);

        triangulate_right = ComputeDelaunayTriangulation(true);
    }

    vector<Vec3d> triangulate_left_d, triangulate_right_d;
    vector<Vec6f> plane_param_left, plane_param_right;
    cout << "Compute Disparity Planes" << endl;
    {
        boost::progress_timer t;
        plane_param_left = ComputeDisparityPlanes(triangulate_left, triangulate_left_d);
        plane_param_right = ComputeDisparityPlanes(triangulate_right, triangulate_right_d);
    }
    int grid_width = (int)ceil(width_ / param_.grid_size);
    int grid_height = (int)ceil(height_ / param_.grid_size);

    Mat disparity_grid_left(grid_height * grid_width, param_.disp_max + 2, CV_16S, Scalar(0));
    Mat disparity_grid_right(grid_height * grid_width, param_.disp_max + 2, CV_16S, Scalar(0));

    {
        boost::progress_timer t;
        cout << "Compute Disparity Grid Left" << endl;
        CreateGrid(disparity_grid_left, Size(grid_width, grid_height), false);
        cout << "Compute Disparity Grid Right" << endl;
        CreateGrid(disparity_grid_right, Size(grid_width, grid_height), true);
    }

    {
        boost::progress_timer t;
        cout << "Compute Disparity Left" << endl;
        ComputeDisparity(triangulate_left, plane_param_left, disparity_grid_left, Size(grid_width, grid_height), false,
                         triangulate_left_d, disparity_left);
        // cout << "Compute Disparity Right" << endl;
        // ComputeDisparity(triangulate_right, plane_param_right, disparity_grid_right, Size(grid_width, grid_height),
        //                  false, triangulate_right_d, disparity_right);
    }

    return true;
}

void Elas::SaveSupportPoints()
{
    string points_file = "Support_Points_mod.txt";  
    ofstream outFile(points_file.c_str());
    for (auto point : support_points_)
    {
        outFile << point.x << ' ' << point.y << ' ' << point.z; 
        outFile << endl;
    }
}

void Elas::RemoveRedundantSupportPoints(Mat &D_can, int32_t redun_max_dist, int32_t redun_threshold, bool vertical)
{
    int D_can_width = D_can.cols;
    int D_can_height = D_can.rows;
    // parameters
    int32_t redun_dir_u[2] = {0, 0};
    int32_t redun_dir_v[2] = {0, 0};
    if (vertical)
    {
        redun_dir_v[0] = -1;
        redun_dir_v[1] = +1;
    }
    else
    {
        redun_dir_u[0] = -1;
        redun_dir_u[1] = +1;
    }

    // for all valid support points do
    for (int32_t u_can = 0; u_can < D_can_width; u_can++)
    {
        for (int32_t v_can = 0; v_can < D_can_height; v_can++)
        {
            int16_t d_can = D_can.at<short>(v_can, u_can);
            if (d_can >= 0)
            {

                // check all directions for redundancy
                bool redundant = true;
                for (int32_t i = 0; i < 2; i++)
                {

                    // search for support
                    int32_t u_can_2 = u_can;
                    int32_t v_can_2 = v_can;
                    int16_t d_can_2;
                    bool support = false;
                    for (int32_t j = 0; j < redun_max_dist; j++)
                    {
                        u_can_2 += redun_dir_u[i];
                        v_can_2 += redun_dir_v[i];
                        if (u_can_2 < 0 || v_can_2 < 0 || u_can_2 >= D_can_width || v_can_2 >= D_can_height)
                            break;
                        d_can_2 = D_can.at<short>(v_can_2, u_can_2);
                        if (d_can_2 >= 0 && abs(d_can - d_can_2) <= redun_threshold)
                        {
                            support = true;
                            break;
                        }
                    }

                    // if we have no support => point is not redundant
                    if (!support)
                    {
                        redundant = false;
                        break;
                    }
                }

                // invalidate support point if it is redundant
                if (redundant)
                    D_can.at<short>(v_can, u_can) = -1;
            }
        }
    }
}

vector<Point3i> Elas::ComputeSupportMatches()
{
    vector<Point3i> support_points_;
    int candidate_stepsize = param_.candidate_stepsize;

    assert(candidate_stepsize > 0);

    Mat D_can(ceil(height_ / candidate_stepsize), ceil(width_ / candidate_stepsize), CV_16S, -1);
    // Mat D_can(height_ / candidate_stepsize, width_ / candidate_stepsize, CV_16S, -1);

    int d, d2;

    for (int v = candidate_stepsize; v < height_; v += candidate_stepsize)
    {
        for (int u = candidate_stepsize; u < width_; u += candidate_stepsize)
        {
            d = ComputeMatchingDisparity(u, v, false);

            if (d >= 0)
            {
                // find backwards
                d2 = ComputeMatchingDisparity(u - d, v, true);
                if (d2 >= 0 && abs(d - d2) <= param_.lr_threshold)
                {
                    D_can.at<short>(v / candidate_stepsize, u / candidate_stepsize) = d;
                    // if (v < 20)
                    //     cout << u << " " << v << " " << u - d << endl;
                    // support_points_.push_back(Point3i(u, v, d));
                }
            }
        }
    }
    // return support_points_;

    RemoveInconsistentSupportPoints(D_can);
    RemoveRedundantSupportPoints(D_can, 5, 1, true);
    RemoveRedundantSupportPoints(D_can, 5, 1, false);

    vector<KeyPoint> keypoints1, keypoints2;
    for (int v = candidate_stepsize; v < height_; v += candidate_stepsize)
    {
        for (int u = candidate_stepsize; u < width_; u += candidate_stepsize)
        {
            d = D_can.at<short>(v / candidate_stepsize, u / candidate_stepsize);
            if (d > 0)
            {
                support_points_.push_back(Point3i(u, v, d));
                // keypoints1.push_back(KeyPoint(Point2f(u, v), 5));
                // keypoints2.push_back(KeyPoint(Point2f(u - d, v), 5));
                // if (v == 20)
                //     cout << u << " " << v << " " << u - d << endl;
            }
        }
    }

    // Mat img_mathes;
    // vector<DMatch> a;
    // drawMatches(image_left_, keypoints1, image_right_, keypoints2, a, img_mathes);
    // imshow("Mathces", img_mathes);
    // waitKey(0);

    // exit(0);

    return support_points_;
}

void Elas::RemoveInconsistentSupportPoints(Mat &D_can)
{
    int width = D_can.cols;
    int height = D_can.rows;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            short d = D_can.at<short>(j, i);
            if (d > 0)
            {
                int support = 0;

                for (int d_neighbour_x = i - param_.incon_window_size;
                     d_neighbour_x <= i + param_.incon_window_size; d_neighbour_x++)
                {
                    for (int d_neighbour_y = i - param_.incon_window_size;
                         d_neighbour_y <= i + param_.incon_window_size; d_neighbour_y++)
                    {
                        if (d_neighbour_x >= 0 && d_neighbour_y >= 0 && d_neighbour_x < width &&
                            d_neighbour_y < height)
                        {
                            short d_neighbour = D_can.at<short>(d_neighbour_y, d_neighbour_x);
                            if (d_neighbour >= 0 && abs(d - d_neighbour) <= param_.incon_threshold)
                                support++;
                        }
                    }
                }

                if (support < param_.incon_min_support)
                {
                    D_can.at<short>(j, i) = -1;
                }
            }
        }
    }
}

int Elas::ComputeMatchingDisparity(const int &u, const int &v, const bool &right_image)
{
    const int up_shift = (v - 2) * width_ + u - 2;
    const int left_shift = (v - 2) * width_ + u + 2;
    const int right_shift = (v + 2) * width_ + u + 2;
    const int bottom_shift = (v + 2) * width_ + u - 2;

    if (right_shift > width_ * height_ || up_shift < 0)
        return -1;

    int disp_max_valid, disp_min_valid = max(param_.disp_min, 0);

    if (right_image)
        disp_max_valid = min(param_.disp_max, width_ - u);
    else
        disp_max_valid = min(param_.disp_max, u);

    uchar *left_up, *left_left, *left_right, *left_bottom, *right_up, *right_left, *right_right, *right_bottom;
    int sum, min_energy = INT_MAX, sec_min_energy = INT_MAX, min_disparity = -1, sec_min_disparity = -1;

    for (int d = disp_min_valid; d <= disp_max_valid; ++d)
    {
        if (right_image)
        {
            left_up = descriptor_left_.ptr<uchar>(up_shift + d);
            left_left = descriptor_left_.ptr<uchar>(left_shift + d);
            left_right = descriptor_left_.ptr<uchar>(right_shift + d);
            left_bottom = descriptor_left_.ptr<uchar>(bottom_shift + d);

            right_up = descriptor_right_.ptr<uchar>(up_shift);
            right_left = descriptor_right_.ptr<uchar>(left_shift);
            right_right = descriptor_right_.ptr<uchar>(right_shift);
            right_bottom = descriptor_right_.ptr<uchar>(bottom_shift);
        }
        else
        {
            left_up = descriptor_left_.ptr<uchar>(up_shift);
            left_left = descriptor_left_.ptr<uchar>(left_shift);
            left_right = descriptor_left_.ptr<uchar>(right_shift);
            left_bottom = descriptor_left_.ptr<uchar>(bottom_shift);

            right_up = descriptor_right_.ptr<uchar>(up_shift - d);
            right_left = descriptor_right_.ptr<uchar>(left_shift - d);
            right_right = descriptor_right_.ptr<uchar>(right_shift - d);
            right_bottom = descriptor_right_.ptr<uchar>(bottom_shift - d);
        }

        sum = 0;
        for (int i = 0; i < 16; ++i)
        {
            sum += abs(left_up[i] - right_up[i]);
            sum += abs(left_left[i] - right_left[i]);
            sum += abs(left_right[i] - right_right[i]);
            sum += abs(left_bottom[i] - right_bottom[i]);
        }

        if (sum == 0)
            continue;

        if (sum < min_energy)
        {
            sec_min_energy = min_energy;
            sec_min_disparity = min_disparity;
            min_energy = sum;
            min_disparity = d;
        }
        else if (sum < sec_min_energy)
        {
            sec_min_energy = sum;
            sec_min_disparity = d;
        }
    }

    if (min_disparity >= 0 && sec_min_disparity >= 0 &&
        (float)min_energy < param_.support_threshold * (float)sec_min_energy)
        return min_disparity;
    else
        return -1;
}

void draw_subdiv(Mat &img, Subdiv2D &subdiv, Scalar delaunay_color)
{
    int width_ = 1280;
    int height_ = 720;
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);

    for (vector<Vec6f>::iterator it = triangleList.begin(); it != triangleList.end();)
    {
        if ((*it)[0] > width_ || (*it)[2] > width_ || (*it)[4] > width_ || (*it)[0] < 0 || (*it)[2] < 0 ||
            (*it)[4] < 0 ||
            (*it)[1] > height_ || (*it)[3] > height_ || (*it)[5] > height_ || (*it)[1] < 0 || (*it)[3] < 0 ||
            (*it)[5] < 0)
        {
            it = triangleList.erase(it);
        }
        else
        {
            ++it;
        }
    }

    for (size_t i = 0; i < triangleList.size(); ++i)
    {
        Vec6f t = triangleList[i];

        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        line(img, pt[0], pt[1], delaunay_color, 1);
        line(img, pt[1], pt[2], delaunay_color, 1);
        line(img, pt[2], pt[0], delaunay_color, 1);
    }
}

vector<Vec6f> Elas::ComputeDelaunayTriangulation(const bool &right_image)
{
    Subdiv2D subdiv(Rect(0, 0, width_, height_));

    for (const auto point : support_points_)
    {
        if (right_image)
            subdiv.insert(Point2f((float)(point.x - point.z), (float)point.y));
        else
            subdiv.insert(Point2f((float)point.x, (float)point.y));
    }

    vector<Vec6f> triangles;
    subdiv.getTriangleList(triangles);
    for (vector<Vec6f>::iterator it = triangles.begin(); it != triangles.end();)
    {
        if ((*it)[0] > width_ || (*it)[2] > width_ || (*it)[4] > width_ || (*it)[0] < 0 || (*it)[2] < 0 ||
            (*it)[4] < 0 ||
            (*it)[1] > height_ || (*it)[3] > height_ || (*it)[5] > height_ || (*it)[1] < 0 || (*it)[3] < 0 ||
            (*it)[5] < 0)
        {
            it = triangles.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Scalar delaunay_color(255, 255, 255), point_color(0, 0, 255);
    // draw_subdiv(image_left_, subdiv, delaunay_color);
    // imshow("A", image_left_);
    // waitKey();
    return triangles;
}

vector<Vec6f> Elas::ComputeDisparityPlanes(const vector<Vec6f> &triangulate_points, vector<Vec3d> &triangulate_d)
{
    vector<Vec6f> result;
    for (auto tri_point : triangulate_points)
    {
        Mat A_left = (Mat_<double>(3, 3) << tri_point[0], tri_point[1], 1,
                      tri_point[2], tri_point[3], 1,
                      tri_point[4], tri_point[5], 1);
        double d1, d2, d3;
        for (auto point : support_points_)
        {
            if (point.x == tri_point[0] && point.y == tri_point[1])
                d1 = point.z;
            if (point.x == tri_point[2] && point.y == tri_point[3])
                d2 = point.z;
            if (point.x == tri_point[4] && point.y == tri_point[5])
                d3 = point.z;
        }
        Mat b_left = (Mat_<double>(3, 1) << d1, d2, d3);
        Mat A_right = (Mat_<double>(3, 3) << tri_point[0] - d1, tri_point[1], 1,
                       tri_point[2] - d2, tri_point[3], 1,
                       tri_point[4] - d3, tri_point[5], 1);

        Mat result_left, result_right;
        if (solve(A_left, b_left, result_left) && solve(A_right, b_left, result_right))
        {
            MatIterator_<double> it_left = result_left.begin<double>();
            MatIterator_<double> it_right = result_right.begin<double>();
            result.push_back(Vec6f((float)(*it_left), (float)(*(it_left + 1)), (float)(*(it_left + 2)),
                                   (float)(*it_right), (float)(*(it_right + 1)), (float)(*(it_right + 2))));
            triangulate_d.push_back(Vec3d(d1, d2, d3));
        }
    }

    return result;
}

void Elas::CreateGrid(Mat &disparity_grid, const Size &disparity_grid_size, const bool &right_image)
{
    int grid_width = disparity_grid_size.width;
    int grid_height = disparity_grid_size.height;

    Mat disparity_grid_temp(grid_height * grid_width, param_.disp_max + 1, CV_16S, Scalar(0));

    for (auto &point : support_points_)
    {
        int x_curr = point.x, y_curr = point.y, d_curr = point.z;
        int d_min = max(d_curr - 1, 0), d_max = min(d_curr + 1, param_.disp_max);
        int x, y = floor(y_curr / param_.grid_size);

        if (right_image)
            x = floor((x_curr - d_curr) / param_.grid_size);
        else
            x = floor(x_curr / param_.grid_size);

        if (x >= 0 && x < grid_width && y >= 0 && y < grid_height)
        {
            for (int d = d_min; d < d_max; ++d)
            {
                disparity_grid_temp.at<short>(x + grid_width * y, d) = 1;
            }
        }
    }
    for (int x = 1; x < grid_width - 1; x++)
    {
        for (int y = 1; y < grid_height - 1; y++)
        {
            int curr_ind = 1;

            for (int d = 0; d <= param_.disp_max; d++)
            {
                if (disparity_grid_temp.at<short>((x - 1) + grid_width * (y - 1), d) == 1 ||
                    disparity_grid_temp.at<short>(x + grid_width * (y - 1), d) == 1 ||
                    disparity_grid_temp.at<short>((x + 1) + grid_width * (y - 1), d) == 1 ||
                    disparity_grid_temp.at<short>((x - 1) + grid_width * y, d) == 1 ||
                    disparity_grid_temp.at<short>(x + grid_width * y, d) == 1 ||
                    disparity_grid_temp.at<short>((x + 1) + grid_width * y, d) == 1 ||
                    disparity_grid_temp.at<short>((x - 1) + grid_width * (y + 1), d) == 1 ||
                    disparity_grid_temp.at<short>(x + grid_width * (y + 1), d) == 1 ||
                    disparity_grid_temp.at<short>((x + 1) + grid_width * (y + 1), d) == 1)
                {
                    disparity_grid.at<short>(x + grid_width * y, curr_ind) = d;
                    curr_ind++;
                }
            }
            disparity_grid.at<short>(x + grid_width * y, 0) = curr_ind - 1;
        }
    }
}

void Elas::ComputeDisparity(const vector<Vec6f> &triangulate_points, const vector<Vec6f> &plane_params,
                            const Mat &disparity_grid, const Size &disparity_grid_size, const bool &right_image,
                            const vector<Vec3d> &triangulate_d, Mat &disparity)
{
    // pre-compute prior
    float two_sigma_squared = 2 * param_.sigma * param_.sigma;
    int *P = new int[param_.disp_max + 1];
    for (int delta_d = 0; delta_d < param_.disp_max + 1; delta_d++)
        P[delta_d] = (int)((-log(param_.gamma + exp(-delta_d * delta_d / two_sigma_squared)) + log(param_.gamma)) /
                           param_.beta);
    int plane_radius = (int)max((float)ceil(param_.sigma * param_.sradius), (float)2.0);

    int c1, c2, c3;
    float plane_a, plane_b, plane_c, plane_d;

    for (int i = 0; i < plane_params.size(); i++)
    {
        if (!right_image)
        {
            plane_a = plane_params[i][0];
            plane_b = plane_params[i][1];
            plane_c = plane_params[i][2];
            plane_d = plane_params[i][3];
        }
        else
        {
            plane_a = plane_params[i][3];
            plane_b = plane_params[i][4];
            plane_c = plane_params[i][5];
            plane_d = plane_params[i][0];
        }

        float tri_u[3], tri_v[3] = {triangulate_points[i][1], triangulate_points[i][3], triangulate_points[i][5]};
        if (right_image)
        {
            tri_u[0] = triangulate_points[i][0] - triangulate_d[i][0];
            tri_u[1] = triangulate_points[i][2] - triangulate_d[i][1];
            tri_u[2] = triangulate_points[i][4] - triangulate_d[i][2];
        }
        else
        {
            tri_u[0] = triangulate_points[i][0];
            tri_u[1] = triangulate_points[i][2];
            tri_u[2] = triangulate_points[i][4];
        }

        for (uint32_t j = 0; j < 3; j++)
        {
            for (uint32_t k = 0; k < j; k++)
            {
                if (tri_u[k] > tri_u[j])
                {
                    float tri_u_temp = tri_u[j];
                    tri_u[j] = tri_u[k];
                    tri_u[k] = tri_u_temp;
                    float tri_v_temp = tri_v[j];
                    tri_v[j] = tri_v[k];
                    tri_v[k] = tri_v_temp;
                }
            }
        }

        // rename corners
        float A_u = tri_u[0];
        float A_v = tri_v[0];
        float B_u = tri_u[1];
        float B_v = tri_v[1];
        float C_u = tri_u[2];
        float C_v = tri_v[2];

        // compute straight lines connecting triangle corners
        float AB_a = 0;
        float AC_a = 0;
        float BC_a = 0;
        if ((int32_t)(A_u) != (int32_t)(B_u))
            AB_a = (A_v - B_v) / (A_u - B_u);
        if ((int32_t)(A_u) != (int32_t)(C_u))
            AC_a = (A_v - C_v) / (A_u - C_u);
        if ((int32_t)(B_u) != (int32_t)(C_u))
            BC_a = (B_v - C_v) / (B_u - C_u);
        float AB_b = A_v - AB_a * A_u;
        float AC_b = A_v - AC_a * A_u;
        float BC_b = B_v - BC_a * B_u;

        // a plane is only valid if itself and its projection
        // into the other image is not too much slanted
        bool valid = fabs(plane_a) < 0.7 && fabs(plane_d) < 0.7;

        // first part (triangle corner A->B)
        if ((int32_t)(A_u) != (int32_t)(B_u))
        {
            for (int32_t u = max((int32_t)A_u, 0); u < min((int32_t)B_u, width_); u++)
            {
                int32_t v_1 = (uint32_t)(AC_a * (float)u + AC_b);
                int32_t v_2 = (uint32_t)(AB_a * (float)u + AB_b);
                if (v_1 <= height_ && v_2 <= height_)
                    for (int32_t v = min(v_1, v_2); v < max(v_1, v_2); v++)
                    {
                        findMatch(u, v, plane_a, plane_b, plane_c, disparity_grid, disparity_grid_size, P, plane_radius,
                                  valid, right_image, disparity);
                    }
            }
        }

        // second part (triangle corner B->C)
        if ((int32_t)(B_u) != (int32_t)(C_u))
        {
            for (int32_t u = max((int32_t)B_u, 0); u < min((int32_t)C_u, width_); u++)
            {
                int32_t v_1 = (uint32_t)(AC_a * (float)u + AC_b);
                int32_t v_2 = (uint32_t)(BC_a * (float)u + BC_b);

                for (int32_t v = min(v_1, v_2); v < max(v_1, v_2); v++)
                {
                    findMatch(u, v, plane_a, plane_b, plane_c, disparity_grid, disparity_grid_size, P, plane_radius,
                              valid, right_image, disparity);
                }
            }
        }
    }
}

void Elas::findMatch(const int32_t &u, const int32_t &v, const float &plane_a, const float &plane_b, const float &plane_c,
                     const Mat &disparity_grid, const Size &disparity_grid_size, int32_t *P, const int32_t &plane_radius,
                     const bool &valid, const bool &right_image, Mat &disparity)
{
    uint32_t grid_height = disparity_grid_size.height;
    uint32_t grid_width = disparity_grid_size.width;
    const int32_t window_size = 2;
    if (u < window_size || u >= width_ - window_size)
        return;

    int32_t sum = 0;
    const uchar *data_desc1, *data_desc2;
    if (right_image)
    {
        data_desc1 = descriptor_right_.ptr<uchar>(v * width_ + u);
    }
    else
    {
        data_desc1 = descriptor_left_.ptr<uchar>(v * width_ + u);
    }

    // for (int32_t i = 0; i < 16; i++)
    //     sum += abs((int)data_desc1[i] - 128);

    // if (sum < param_.match_texture)
    // return;

    // compute disparity, min disparity and max disparity of plane prior
    int32_t d_plane = (int32_t)(plane_a * (float)u + plane_b * (float)v + plane_c);
    int32_t d_plane_min = max(d_plane - plane_radius, 0);
    int32_t d_plane_max = min(d_plane + plane_radius, param_.disp_max);

    // get grid pointer
    int32_t grid_x = (int32_t)floor((float)u / (float)param_.grid_size);
    int32_t grid_y = (int32_t)floor((float)v / (float)param_.grid_size);

    int32_t num_grid = disparity_grid.at<short>(v * grid_width + u, 0);
    const short *d_grid = disparity_grid.ptr<short>(v * grid_width + u);
    int32_t u_warp, val;
    int32_t min_val = INT_MAX;
    int32_t min_d = -1;

    if (!right_image)
    {
        short d_curr;
        for (int32_t i = 0; i < num_grid; i++)
        {
            d_curr = d_grid[i + 1];
            if (d_curr < d_plane_min || d_curr > d_plane_max)
            {
                u_warp = u - d_curr;
                if (u_warp < window_size || u_warp >= width_ - window_size)
                    continue;
                data_desc2 = descriptor_right_.ptr<uchar>(v * width_ + u - d_curr);

                sum = 0;
                for (int i = 0; i < 16; ++i)
                {
                    sum += abs(data_desc1[i] - data_desc2[i]);
                }

                if (sum < min_val)
                {
                    min_val = sum;
                    min_d = d_curr;
                }

                // set disparity value
                if (min_d >= 0)
                    disparity.at<float>(v, u) = min_d; // MAP value (min neg-Log probability)
                else
                    disparity.at<float>(v, u) = -1; // invalid disparity
            }
        }

        for (d_curr = d_plane_min; d_curr <= d_plane_max; d_curr++)
        {
            sum = 0;
            u_warp = u - d_curr;
            if (u_warp < window_size || u_warp >= width_ - window_size)
                continue;
            data_desc2 = descriptor_right_.ptr<uchar>(v * width_ + u - d_curr);

            for (int i = 0; i < 16; ++i)
            {
                sum += abs(data_desc1[i] - data_desc2[i]);
            }

            sum += valid ? *(P + abs(d_curr - d_plane)) : 0;
            if (sum < min_val)
            {
                min_val = sum;
                min_d = d_curr;
            }
        }
        // set disparity value
        if (min_d >= 0)
            disparity.at<float>(v, u) = min_d; // MAP value (min neg-Log probability)
        else
            disparity.at<float>(v, u) = -1; // invalid disparity
    }

    // for (int32_t i = 0; i < num_grid; i++)
    // {
    //     short d_curr = d_grid[i + 1];

    //     if (right_image)
    //     {
    //         u_warp = u + d_curr;

    //         if (u_warp < window_size || u_warp >= width_ - window_size)
    //             continue;
    //         else
    //             data_desc2 = descriptor_left_.ptr<uchar>(v * width_ + u_warp);
    //     }
    //     else
    //     {
    //         u_warp = u - d_curr;
    //         if (u_warp < window_size || u_warp >= width_ - window_size)
    //             continue;
    //         else
    //             data_desc2 = descriptor_right_.ptr<uchar>(v * width_ + u_warp);
    //     }

    //     for (int i = 0; i < 16; ++i)
    //     {
    //         sum += abs(data_desc1[i] - data_desc2[i]);
    //     }

    //     sum += valid ? *(P + abs(d_curr - d_plane)) : 0;
    //     if (sum < min_val)
    //     {
    //         min_val = val;
    //         min_d = d_curr;
    //     }

    //     // set disparity value
    //     if (min_d >= 0)
    //         disparity.at<float>(v, u) = min_d; // MAP value (min neg-Log probability)
    //     else
    //         disparity.at<float>(v, u) = -1; // invalid disparity
    // }
}
