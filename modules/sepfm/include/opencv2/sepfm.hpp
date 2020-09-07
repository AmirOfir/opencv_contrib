// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_SEPFM_H_
#define _OPENCV_SEPFM_H_


#include <opencv2/core.hpp>

/**
@defgroup sepfm Separable Fundamental Matrix finder
*/

#define DEFAULT_HOUGH_RESCALE -1

namespace cv {
namespace separableFundamentalMatrix {
//! @addtogroup sepfm
//! @{



Mat CV_EXPORTS_W findSeparableFundamentalMat(InputArray pts1, InputArray pts2, int im_size_h_org, int im_size_w_org,
    float inlier_ratio = 0.4, int inlier_threshold = 3,
    double hough_rescale = DEFAULT_HOUGH_RESCALE, int num_matching_pts_to_use = 150, int pixel_res = 4, int min_hough_points = 4,
    int theta_res = 180, float max_distance_pts_line = 3, int top_line_retries = 2, int min_shared_points = 4);    


//! @}
}
}

#endif
