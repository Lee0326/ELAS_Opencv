#include <algorithm>
#include <math.h>

#include "elas.h"
#include "descriptor.h"
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
        plane_param_right = ComputeDisparityPlanes(triangulate_right, triangulate_left_d);
    }
    int grid_width = (int)ceil((float)width_ / (float)param_.grid_size);
    int grid_height = (int)ceil((float)height_ / (float)param_.grid_size);

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
        ComputeDisparity(triangulate_left, plane_param_left, disparity_grid_left, Size(grid_width, grid_height), false, triangulate_left_d, disparity_left);
        cout << "Compute Disparity Right" << endl;
        ComputeDisparity(triangulate_right, plane_param_right, disparity_grid_right, Size(grid_width, grid_height), false, triangulate_right_d, disparity_right);
    }

    return true;
}

vector<Point3i> Elas::ComputeSupportMatches()
{
    vector<Point3i> support_points_;
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
                support_points_.push_back(Point3i(u * candidate_stepsize, v * candidate_stepsize, d));
            }
        }
    }

    return support_points_;
}

int Elas::ComputeMatchingDisparity(const int &u, const int &v, const bool &right_image)
{
    if (u < 0 || u > width_ || v < 0 || v > height_)
        return -1;

    const int up_shift = (v - 2) * width_ + u;
    const int left_shift = v * width_ + u - 2;
    const int right_shift = v * width_ + u + 2;
    const int bottom_shift = (v + 2) * width_ + u;

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
        if (solve(A_left, b_left, result_left) && solve(A_left, b_left, result_right))
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
        int x, y = floor((float)(y_curr / param_.grid_size));

        if (right_image)
            x = floor((float)((x_curr - d_curr) / param_.grid_size));
        else
            x = floor((float)(x_curr / param_.grid_size));

        if (x >= 0 && x < grid_width - 1 && y >= 0 && y < grid_height - 1)
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

void Elas::ComputeDisparity(const vector<Vec6f> triangulate_points, const vector<Vec6f> &plane_params, const Mat &disparity_grid, const Size &disparity_grid_size, const bool &right_image,
                            const vector<Vec3d> &triangulate_d, Mat &disparity)
{
    // pre-compute prior
    float two_sigma_squared = 2 * param_.sigma * param_.sigma;
    int *P = new int[param_.disp_max + 1];
    for (int delta_d = 0; delta_d < param_.disp_max + 1; delta_d++)
        P[delta_d] = (int)((-log(param_.gamma + exp(-delta_d * delta_d / two_sigma_squared)) + log(param_.gamma)) / param_.beta);
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
        if (!right_image)
        {
            tri_u[0] = triangulate_points[i][0];
            tri_u[1] = triangulate_points[i][2];
            tri_u[2] = triangulate_points[i][4];
        }
        else
        {
            tri_u[0] = triangulate_points[i][0] - triangulate_d[i][0];
            tri_u[1] = triangulate_points[i][2] - triangulate_d[i][1];
            tri_u[2] = triangulate_points[i][4] - triangulate_d[i][2];
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
                if (!param_.subsampling || u % 2 == 0)
                {
                    int32_t v_1 = (uint32_t)(AC_a * (float)u + AC_b);
                    int32_t v_2 = (uint32_t)(AB_a * (float)u + AB_b);
                    for (int32_t v = min(v_1, v_2); v < max(v_1, v_2); v++)
                        if (!param_.subsampling || v % 2 == 0)
                        {
                            findMatch(u, v, plane_a, plane_b, plane_c, disparity_grid, disparity_grid_size, P, plane_radius, valid, right_image, disparity);
                        }
                }
            }
        }

        // second part (triangle corner B->C)
        if ((int32_t)(B_u) != (int32_t)(C_u))
        {
            for (int32_t u = max((int32_t)B_u, 0); u < min((int32_t)C_u, width_); u++)
            {
                if (!param_.subsampling || u % 2 == 0)
                {
                    int32_t v_1 = (uint32_t)(AC_a * (float)u + AC_b);
                    int32_t v_2 = (uint32_t)(BC_a * (float)u + BC_b);
                    for (int32_t v = min(v_1, v_2); v < max(v_1, v_2); v++)
                        if (!param_.subsampling || v % 2 == 0)
                        {
                            findMatch(u, v, plane_a, plane_b, plane_c, disparity_grid, disparity_grid_size, P, plane_radius, valid, right_image, disparity);
                        }
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
    const short *data_desc1, *data_desc2;
    if (right_image)
    {
        data_desc1 = descriptor_right_.ptr<short>(v * width_ + u);
        data_desc2 = descriptor_left_.ptr<short>(v * width_ + u);
    }
    else
    {
        data_desc1 = descriptor_left_.ptr<short>(v * width_ + u);
        data_desc2 = descriptor_right_.ptr<short>(v * width_ + u);
    }

    for (int32_t i = 0; i < 16; i++)
        sum += abs(data_desc1[i] - 128);
    if (sum < param_.match_texture)
        return;

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

    for (int32_t i = 0; i < num_grid; i++)
    {
        short d_curr = d_grid[i + 1];

        if (right_image)
        {
            u_warp = u + d_curr;

            if (u_warp < window_size || u_warp >= width_ - window_size)
                continue;
            else
                data_desc2 = descriptor_left_.ptr<short>(v * width_ + u_warp);
        }
        else
        {
            u_warp = u - d_curr;
            if (u_warp < window_size || u_warp >= width_ - window_size)
                continue;
            else
                data_desc2 = descriptor_right_.ptr<short>(v * width_ + u_warp);
        }

        for (int i = 0; i < 16; ++i)
        {
            sum += abs(data_desc1[i] - data_desc2[i]);
        }

        sum += valid ? *(P + abs(d_curr - d_plane)) : 0;
        if (sum < min_val)
        {
            min_val = val;
            min_d = d_curr;
        }

        // set disparity value
        if (min_d >= 0)
            disparity.at<float>(v, u) = min_d; // MAP value (min neg-Log probability)
        else
            disparity.at<float>(v, u) = -1; // invalid disparity
    }
}