// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/sepfm.hpp"
#include "opencv2/opencv_modules.hpp"
#include <vector>
#include <numeric>

namespace cv {
namespace separableFundamentalMatrix {

using namespace cv;

struct line_info
{
    std::vector<int> matching_indexes;
    Point3d line_eq_abc;
    Point3d line_eq_abc_norm;
    Point2d bottom_left_edge_point;
    Point2d top_right_edge_point;
    double max_distance;
    int line_index;
};


class top_line
{
public:
    bool empty() { return num_inliers == 0; }
    int num_inliers;
    std::vector<Point2d> line_points_1;
    std::vector<Point2d> line_points_2;
    int line1_index;
    int line2_index;
    std::vector<int> inlier_selected_index;
    std::vector<Point2d> selected_line_points1;
    std::vector<Point2d> selected_line_points2;
    double max_dist;
    double min_dist;
    double homg_err;
};

}    
}

#endif
