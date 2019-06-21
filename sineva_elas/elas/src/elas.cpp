#include <algorithm>
#include <math.h>

#include "../include/elas.h"
#include "../include/descriptor.h"
#include "boost/progress.hpp"
using namespace std;
using namespace cv;

bool Elas::process(const Mat &image_left, const Mat &image_right, Mat &disparity_left, Mat &disparity_right)
{

    // get width, height and bytes per line
    width_ = image_left.cols;
    height_ = image_left.rows;

    // compute descriptor
    cout << "Compute Descriptor" << endl;
    Descriptor descriptor_left(image_left, width_, height_, param_.subsampling);
    Descriptor descriptor_right(image_right, width_, height_, param_.subsampling);
    descriptor_left_ = descriptor_left.CreateDescriptor();
    descriptor_right_ = descriptor_right.CreateDescriptor();
    vector<Point3i> support_points = ComputeSupportMatches();

    cout << "Compute Triangulate" << endl;
    vector<Vec6f> triangulate_left, triangulate_right;
    triangulate_left = ComputeDelaunayTriangulation(support_points, false);
    triangulate_right = ComputeDelaunayTriangulation(support_points, true);

    cout << "Compute Disparity Planes" << endl;
    ComputeDisparityPlanes(support_points, triangulate_left);
    ComputeDisparityPlanes(support_points, triangulate_left);

    int grid_width = (int)ceil((float)width_ / (float)param_.grid_size);
    int grid_height = (int)ceil((float)height_ / (float)param_.grid_size);

    cout << "Compute Disparity Grid" << endl;
    Mat disparity_grid(grid_height * grid_width, param_.disp_max + 2, CV_16U, Scalar(0));

    cout << "Compute Disparity Grid Left" << endl;
    CreateGrid(support_points, disparity_grid, Size(grid_width, grid_height), false);
    cout << "Compute Disparity Grid Right" << endl;
    CreateGrid(support_points, disparity_grid, Size(grid_width, grid_height), true);

    ComputeDisparity(support_points, triangulate_left, disparity_grid, Size(grid_width, grid_height), false, disparity_left);

    return true;
}

vector<Point3i> Elas::ComputeSupportMatches()
{
    vector<Point3i> support_points;
    int candidate_stepsize = param_.candidate_stepsize;

    Mat D_can(height_ / candidate_stepsize, width_ / candidate_stepsize, CV_16S, -1);

    int d, d2;

    for (int u = candidate_stepsize; u < width_; u += candidate_stepsize)
    {
        for (int v = candidate_stepsize; v < height_; v += candidate_stepsize)
        {
            d = ComputeMatchingDisparity(u, v, false);

            if (d >= 0)
            {
                // find backwards
                d2 = ComputeMatchingDisparity(u - d, v, true);
                if (d2 >= 0 && abs(d - d2) <= param_.lr_threshold)
                {
                    D_can.at<short>(v / candidate_stepsize, u / candidate_stepsize) = d;
                }
            }
        }
    }
    for (int u = 0; u < D_can.cols; ++u)
    {
        for (int v = 0; v < D_can.rows; ++v)
        {
            short d = D_can.ptr<short>(v)[u];
            if (d >= 0)
            {
                support_points.push_back(Point3i(u * candidate_stepsize, v * candidate_stepsize, d));
            }
        }
    }

    return support_points;
}

int Elas::ComputeMatchingDisparity(const int &u, const int &v, const bool &right_image)
{
    if (u < 0 || u > width_ || v < 0 || v > height_)
        return -1;

    const int up_shift = (v - 1) * width_ + u;
    const int left_shift = v * width_ + u - 1;
    const int right_shift = v * width_ + u + 1;
    const int bottom_shift = (v + 1) * width_ + u;

    int disp_min_valid = max(param_.disp_min, 0);
    int disp_max_valid = param_.disp_max;

    if (right_image)
        disp_max_valid = min(param_.disp_max, width_ - u);
    else
        disp_max_valid = min(param_.disp_max, u);

    short *left_up, *left_left, *left_right, *left_bottom, *right_up, *right_left, *right_right, *right_bottom;
    int sum, min_energy = INT_MAX, sec_min_energy = INT_MAX, min_disparity = -1, sec_min_disparity = -1;

    for (int d = disp_min_valid; d <= disp_max_valid; ++d)
    {
        if (right_image)
        {
            left_up = descriptor_left_.ptr<short>(up_shift + d);
            left_left = descriptor_left_.ptr<short>(left_shift + d);
            left_right = descriptor_left_.ptr<short>(right_shift + d);
            left_bottom = descriptor_left_.ptr<short>(bottom_shift + d);

            right_up = descriptor_left_.ptr<short>(up_shift);
            right_left = descriptor_left_.ptr<short>(left_shift);
            right_right = descriptor_left_.ptr<short>(right_shift);
            right_bottom = descriptor_left_.ptr<short>(bottom_shift);
        }
        else
        {
            left_up = descriptor_left_.ptr<short>(up_shift);
            left_left = descriptor_left_.ptr<short>(left_shift);
            left_right = descriptor_left_.ptr<short>(right_shift);
            left_bottom = descriptor_left_.ptr<short>(bottom_shift);

            right_up = descriptor_left_.ptr<short>(up_shift - d);
            right_left = descriptor_left_.ptr<short>(left_shift - d);
            right_right = descriptor_left_.ptr<short>(right_shift - d);
            right_bottom = descriptor_left_.ptr<short>(bottom_shift - d);
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

    if (min_disparity >= 0 && sec_min_disparity >= 0 && (float)min_energy < param_.support_threshold * (float)sec_min_energy)
        return min_disparity;
    else
        return -1;
}

void Elas::RemoveInconsistentSupportPoints(Mat &D_can)
{
    int width = D_can.cols;
    int height = D_can.rows;
    int candidate_stepsize = param_.candidate_stepsize;

    for (int i = 0; i < width; i += candidate_stepsize)
    {
        for (int j = 0; j < height; j += candidate_stepsize)
        {
            short d = D_can.ptr<short>(j)[i];
            if (d > 0)
            {
                int support = 0;

                for (int d_neighbour_x = i - param_.incon_window_size * candidate_stepsize; d_neighbour_x <= i + param_.incon_window_size * candidate_stepsize; d_neighbour_x += candidate_stepsize)
                {
                    for (int d_neighbour_y = i - param_.incon_window_size * candidate_stepsize; d_neighbour_y <= i + param_.incon_window_size * candidate_stepsize; d_neighbour_y += candidate_stepsize)
                    {
                        if (d_neighbour_x >= 0 && d_neighbour_y >= 0 && d_neighbour_x < width && d_neighbour_y < height)
                        {
                            short d_neighbour = D_can.ptr<short>(d_neighbour_y)[d_neighbour_x];
                            if (d_neighbour >= 0 && abs(d - d_neighbour) <= param_.incon_threshold)
                                support++;
                        }
                    }
                }

                if (support < param_.incon_min_support)
                {
                    D_can.ptr<short>(j)[i] = -1;
                }
            }
        }
    }
}

vector<Vec6f> Elas::ComputeDelaunayTriangulation(const vector<Point3i> &support_points, const bool &right_image)
{
    Subdiv2D subdiv(Rect(0, 0, width_, height_));

    for (const auto point : support_points)
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
        if ((*it)[0] > width_ || (*it)[2] > width_ || (*it)[4] > width_ || (*it)[0] < 0 || (*it)[2] < 0 || (*it)[4] < 0 ||
            (*it)[1] > height_ || (*it)[3] > height_ || (*it)[5] > height_ || (*it)[1] < 0 || (*it)[3] < 0 || (*it)[5] < 0)
        {
            it = triangles.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // for (auto point : triangles)
    // {
    //     cout << point << endl;
    // }
    return triangles;
}

vector<Vec6f> Elas::ComputeDisparityPlanes(const vector<Point3i> &support_points, const vector<Vec6f> &triangulate_points)
{
    vector<Vec6f> result;
    for (auto tri_point : triangulate_points)
    {
        Mat A_left = (Mat_<double>(3, 3) << tri_point[0], tri_point[1], 1,
                      tri_point[2], tri_point[3], 1,
                      tri_point[4], tri_point[5], 1);
        double d1, d2, d3;
        for (auto point : support_points)
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
        if (solve(A_left, b_left, result_left) && solve(A_left, b_left, result_right))
        {
            MatIterator_<double> it_left = result_left.begin<double>();
            MatIterator_<double> it_right = result_right.begin<double>();
            result.push_back(Vec6f((float)(*it_left), (float)(*(it_left + 1)), (float)(*(it_left + 2)),
                                   (float)(*it_right), (float)(*(it_right + 1)), (float)(*(it_right + 2))));
        }
    }
    return result;
}

void Elas::CreateGrid(const vector<Point3i> &support_points, Mat &disparity_grid, const Size &disparity_grid_size, bool right_image)
{
    int grid_width = disparity_grid_size.width;
    int grid_height = disparity_grid_size.height;

    Mat disparity_grid_temp(grid_height * grid_width, param_.disp_max + 1, CV_16U, Scalar(0));

    for (auto point : support_points)
    {
        int x_curr = point.x, y_curr = point.y, d_curr = point.z;
        int d_min = max(d_curr - 1, 0), d_max = min(d_curr + 1, param_.disp_max);
        int x, y = floor((float)(y_curr / param_.grid_size));

        if (right_image)
            x = floor((float)((x_curr - d_curr) / param_.grid_size));
        else
            x = floor((float)(x_curr / param_.grid_size));

        if (x >= 0 && x < grid_width && y >= 0 && y < grid_height)
        {
            for (int d = d_min; d < d_max; ++d)
            {
                disparity_grid_temp.at<short>((x - 1) + grid_width * (y - 1), d) = 1;
                disparity_grid_temp.at<short>(x + grid_width * (y - 1), d) = 1;
                disparity_grid_temp.at<short>((x + 1) + grid_width * (y - 1), d) = 1;
                disparity_grid_temp.at<short>((x - 1) + grid_width * y, d) = 1;
                disparity_grid_temp.at<short>(x + grid_width * y, d) = 1;
                disparity_grid_temp.at<short>((x + 1) + grid_width * y, d) = 1;
                disparity_grid_temp.at<short>((x - 1) + grid_width * (y + 1), d) = 1;
                disparity_grid_temp.at<short>(x + grid_width * (y + 1), d) = 1;
                disparity_grid_temp.at<short>((x + 1) + grid_width * (y + 1), d) = 1;
            }
        }

        for (int x = 0; x < grid_width; x++)
        {
            for (int y = 0; y < grid_height; y++)
            {
                int curr_ind = 1;

                for (int d = 0; d <= param_.disp_max; d++)
                {
                    if (disparity_grid_temp.at<short>(x + grid_width * y, d) > 0)
                    {
                        disparity_grid.at<short>(x + grid_width * y, curr_ind) = d;

                        curr_ind++;
                    }
                }
                disparity_grid.at<short>(x + grid_width * y, 0) = curr_ind - 1;
            }
        }
    }
}

void Elas::ComputeDisparity(const vector<Point3i> &support_points, const vector<Vec6f> &triangulate_points, const Mat &disparity_grid, const Size &disparity_grid_size, bool right_image, Mat &disparity)
{
    // pre-compute prior
    float two_sigma_squared = 2 * param_.sigma * param_.sigma;
    int *P = new int[param_.disp_max + 1];
    for (int delta_d = 0; delta_d < param_.disp_max + 1; delta_d++)
        P[delta_d] = (int)((-log(param_.gamma + exp(-delta_d * delta_d / two_sigma_squared)) + log(param_.gamma)) / param_.beta);
    int plane_radius = (int)max((float)ceil(param_.sigma * param_.sradius), (float)2.0);
}
