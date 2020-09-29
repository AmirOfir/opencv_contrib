// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_MATCHING_LINES_H_
#define _OPENCV_MATCHING_LINES_H_

#include "precomp.hpp"

namespace cv { 
namespace separableFundamentalMatrix {

using namespace cv;
using namespace std;

vector<line_info> getHoughLines(Mat pts, 
    int im_size_w, int im_size_h, int min_hough_points,
    int pixel_res, int theta_res, double max_distance, int num_matching_pts_to_use);
    
vector<top_line> getTopMatchingLines(
    InputArray _ptsImg1, InputArray _ptsImg2, 
    const vector<line_info> &lineInfosImg1, const vector<line_info> &lineInfosImg2, 
    const vector<Point3i> &sharedPoints, int minSharedPoints, double inlierRatio);

Mat createHeatmap(InputArray ptsImg1, InputArray ptsImg2, 
    const vector<line_info> &lineInfosImg1, const vector<line_info> &lineInfosImg2);

}
}


#endif // !_OPENCV_MATCHING_LINES_H_
