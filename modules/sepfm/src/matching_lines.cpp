// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "precomp.hpp"
#include "matching_lines.hpp"
#include "SFM_finder.hpp"
#include "np_cv_imp.hpp"
#include "line_geometry.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

Mat cv::separableFundamentalMatrix::createHeatmap(InputArray ptsImg1, InputArray ptsImg2, const vector<line_info> &lineInfosImg1, const vector<line_info> &lineInfosImg2)
{
    /*
    pts_lines         = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))

    set_ix_1=set_shared_index_1(pts_lines)
    [set_ix_1(x['matching_index'],x['line_index']) for x in lines_info_img1]
    set_ix_2=set_shared_index_2(pts_lines)
    [set_ix_2(x['matching_index'][0],x['line_index']) for x in lines_info_img2 if len(x['matching_index'][0])>0]

    */

    // Create heatmap
    const int heatmapSize[] = { (int)lineInfosImg1.size(), (int)lineInfosImg2.size(), ptsImg1.size().height };
    Mat heatmap = Mat::zeros(3, heatmapSize, CV_8U);

    // Fill by first lines
    for (auto &lineInfo : lineInfosImg1)
    {
        for (const int point_index : lineInfo.matching_indexes)
        {
            for (int i = 0; i < heatmapSize[1]; i++)
            {
                heatmap.at<uchar>(lineInfo.line_index, i, point_index) = 1;
            }
        }
    }

    // Fill by second lines
    for (auto &lineInfo : lineInfosImg2)
    {
        for (const int point_index : lineInfo.matching_indexes)
        {
            for (int i = 0; i < heatmapSize[0]; i++)
            {
                heatmap.at<uchar>(i, lineInfo.line_index, point_index) += 1;
            }
        }
    }

    return heatmap;
}

template <typename _Tp>
vector<Point_<_Tp>> projectPointsOnLineByInices(InputArray _pts, const line_info &lineInfo, vector<int> indices)
{
    vector<Point_<_Tp>> filteredPts;
    filteredPts = byIndices<_Tp>(_pts, indices);
    return projectPointsOnLine(lineInfo.bottom_left_edge_point, lineInfo.top_right_edge_point, filteredPts);
}

template <typename _Tp>
vector<int> uniqueIntersectedPoints(const vector<Point_<_Tp>> &matchingPoints1, const vector<Point_<_Tp>> &matchingPoints2)
{
    vector<size_t> uniqueIdx1 = index_unique(vector<Point>(matchingPoints1.begin(), matchingPoints1.end()));
    vector<size_t> uniqueIdx2 = index_unique(vector<Point>(matchingPoints2.begin(), matchingPoints2.end()));

    vector<int> unique_idx;
    intersect1d(uniqueIdx1.begin(), uniqueIdx1.end(), uniqueIdx2.begin(), uniqueIdx2.end(), back_inserter(unique_idx));

    return unique_idx;
}

array<top_line,2> topTwoLinesWithMaxAngle(const vector<line_info> &lineInfosImg1, vector<top_line> &topLines)
{
    sort(topLines.begin(), topLines.end(),
        [](const top_line &line1, const top_line &line2) { return line1.min_dist > line2.min_dist; });

    const top_line firstLine = topLines[0];

    double maxAngle = -180;
    int maxAngleIx;
    
    auto firstLineEq = lineInfosImg1[firstLine.line1_index].line_eq_abc_norm;
        
    for (int i = 1; i < topLines.size(); i++)
    {
        auto lineEq = lineInfosImg1[topLines[i].line1_index].line_eq_abc_norm;
        double angle = std::acos(std::min<double>((lineEq.x*firstLineEq.x) + (lineEq.y*firstLineEq.y), 1)) * 180 / CV_PI;
        if (std::min(angle, 180 - angle) > maxAngle)
        {
            maxAngle = std::min(angle, 180 - angle);
            maxAngleIx = i;
        }
    }
    
    return { move(firstLine), move(topLines[maxAngleIx]) };
}

bool createTopLine(InputArray _ptsImg1, InputArray _ptsImg2, const vector<Point3i> &num_shared_points_vote, 
    const vector<line_info> &lineInfosImg1, const vector<line_info> &lineInfosImg2, int n, int num_line_ransac_iterations, top_line *curr)
{
    int line1_index = num_shared_points_vote[n].y;
    int line2_index = num_shared_points_vote[n].z;

    vector<int> arr_idx;
    intersect1d(
        lineInfosImg1[line1_index].matching_indexes.begin(), lineInfosImg1[line1_index].matching_indexes.end(), 
        lineInfosImg2[line2_index].matching_indexes.begin(), lineInfosImg2[line2_index].matching_indexes.end(), 
        back_inserter(arr_idx));

    vector<Point2d> matchingPoints1 = projectPointsOnLineByInices<double>(_ptsImg1, lineInfosImg1[line1_index], arr_idx);
    vector<Point2d> matchingPoints2 = projectPointsOnLineByInices<double>(_ptsImg2, lineInfosImg2[line2_index], arr_idx);

    vector<int> uniqueIdx = uniqueIntersectedPoints(matchingPoints1, matchingPoints2);

    // We need at least four unique points
    if (uniqueIdx.size() < 4)
        return false;

    // Filter
    matchingPoints1 = byIndices<double>(matchingPoints1, uniqueIdx);
    matchingPoints2 = byIndices<double>(matchingPoints2, uniqueIdx);

    // Find inliers, inlier_idx_homography - index of inliers of all the line points
    auto matchingPoints = VecMatchingPoints<double>(matchingPoints1, matchingPoints2);
    auto lineInliersErrors = lineInliersRansac(num_line_ransac_iterations, matchingPoints);

    if (lineInliersErrors.inlierCount < 4)
        return false;

    // Find the model with the longest interval
    LineInliersModelResult lineInliersResult;
    vector<Point2d> inlierPoints1, inlierPoints2;
    IntervalEndpointsResult intervalEndpointsResult;
    IntervalMidPointResult intervalMidPointResult;
    intervalEndpointsResult.distance = 0;
    intervalMidPointResult.minDistance = 0;
    for (size_t i = 0; i < lineInliersErrors.fittestModelsErrors.size(); i++)
    {
        auto currLineInliersResult = modelInliers(lineInliersErrors.fittestModelsErrors[i]);

        auto currInlierPoints1 = byIndices<double>(matchingPoints1, currLineInliersResult.inlierIndexes);
        auto currInlierPoints2 = byIndices<double>(matchingPoints2, currLineInliersResult.inlierIndexes);
                
        auto currEndpoints = intervalEndpoints(currInlierPoints1);
        auto currMidPointResult = intervalPointClosestToCenter(currInlierPoints1, currEndpoints.firstIdx, currEndpoints.secondIdx);

        if (currMidPointResult.minDistance > intervalMidPointResult.minDistance)
        {
            lineInliersResult = currLineInliersResult;
            inlierPoints1 = currInlierPoints1;
            inlierPoints2 = currInlierPoints2;
            intervalEndpointsResult = currEndpoints;
            intervalMidPointResult = currMidPointResult;
        }
    }
    
    curr->num_inliers = (int)lineInliersResult.inlierIndexes.size();
    curr->line_points_1 = inlierPoints1;
    curr->line_points_2 = inlierPoints2;
    curr->line1_index = line1_index;
    curr->line2_index = line2_index;
    curr->inlier_selected_index = { intervalEndpointsResult.firstIdx, intervalEndpointsResult.secondIdx, intervalMidPointResult.midPointIdx };
    curr->selected_line_points1 = byIndices<double>(inlierPoints1, curr->inlier_selected_index);
    curr->selected_line_points2 = byIndices<double>(inlierPoints2, curr->inlier_selected_index);
    curr->max_dist = intervalEndpointsResult.distance;
    curr->min_dist = intervalMidPointResult.minDistance;
    curr->homg_err = lineInliersResult.meanError;

    return true;
}

vector<top_line> cv::separableFundamentalMatrix::getTopMatchingLines(
    InputArray _ptsImg1, InputArray _ptsImg2, 
    const vector<line_info> &lineInfosImg1, const vector<line_info> &lineInfosImg2, 
    const vector<Point3i> &sharedPoints, int minSharedPoints, double inlierRatio)
{
    // For each matching points on the matching lines,
    // project the shared points to be exactly on the line
    // start with the lines that shared the highest number of points, so we can do top-N
    // return a list index by the lines (k,j) with the projected points themself
    int num_line_ransac_iterations = int((log(0.01) / log(1 - pow(inlierRatio, 3)))) + 1;

    // Go over the top lines with the most number of shared points, project the points, store by the matching indices of the pair of lines
    int num_sorted_lines = min((int)sharedPoints.size(), 50);
    top_line topLines[50];
    bool topLinesPresent[50];

    cv::parallel_for_(Range(0, num_sorted_lines), [&](const Range& range) {
        for (size_t n = range.start; n < range.end; n++)
        {
            bool success = createTopLine(_ptsImg1, _ptsImg2, sharedPoints, 
                lineInfosImg1, lineInfosImg2, n, num_line_ransac_iterations, &(topLines[n]));
            topLinesPresent[n] = success;
        }
    });

    vector<top_line> nonEmptyTopLines;
    for (size_t i = 0; i < num_sorted_lines; i++)
    {
        if (topLinesPresent[i])
            nonEmptyTopLines.push_back(topLines[i]);
    }

    if (nonEmptyTopLines.size() < 2)
        return {};
    
    auto topTwoLines = topTwoLinesWithMaxAngle(lineInfosImg1, nonEmptyTopLines);

    return { topTwoLines[0], topTwoLines[1] };
}

template <typename _Tp>
vector<_Tp> findIntersectionPoints(_Tp rho, _Tp theta, const int im_size_w, const int im_size_h)
{
    _Tp a = cos(theta);
    _Tp b = sin(theta);
    _Tp x_0 = a != 0 ? rho / a : -1000;
    _Tp x_1 = a != 0 ? (rho - (b * im_size_w)) / a : -1000;
    _Tp y_0 = b != 0 ? rho / b : -1000;
    _Tp y_1 = b != 0 ? (rho - (a * im_size_h)) / b : -1000;

    vector<_Tp> ret;
    if (x_0 >= 0 && x_0 < im_size_h)
    {
        ret.push_back(x_0);
        ret.push_back(0);
    }
    if (y_0 >= 0 && y_0 < im_size_w)
    {
        ret.push_back(0);
        ret.push_back(y_0);
    }
    if (x_1 >= 0 && x_1 < im_size_h)
    {
        ret.push_back(x_1);
        ret.push_back(_Tp(im_size_w));
    }
    if (y_1 >= 0 && y_1 < im_size_w)
    {
        ret.push_back(_Tp(im_size_h));
        ret.push_back(y_1);
    }

    return ret;
}

template <typename _Tp>
line_info createLineInfo(Mat pts, const vector<_Tp> &points_intersection, _Tp max_distance, int line_index)
{
    CV_Assert(points_intersection.size() == 4);

    Point3_<_Tp> pt1(points_intersection[0], points_intersection[1], 1);
    Point3_<_Tp> pt2(points_intersection[2], points_intersection[3], 1);
    Point3_<_Tp> line_eq = pt1.cross(pt2);

    if (abs(line_eq.z) >= FLT_EPSILON)
    {
        // divide by z
        line_eq /= line_eq.z;
    }
    // too small to divide, solve with least square
    else
    {
        _Tp a[4] = { points_intersection[0],1,
                        points_intersection[2],1 };
        Mat A(2, 2, CV_8U, &a);
        vector<_Tp> B{ points_intersection[1], points_intersection[2] };
        vector<_Tp> x;
        solve(A, B, x);
        line_eq.x = x[0];
        line_eq.y = -1;
        line_eq.z = 0;
    }

    vector<int> matching_indexes;
    {
        _Tp scale = sqrt((line_eq.x * line_eq.x) + (line_eq.y * line_eq.y));
        auto d = MatrixVectorMul<_Tp>(pts, line_eq, 1.f / scale, true);
        matching_indexes = index_if(d.begin(), d.end(), [&](_Tp f) {return f < max_distance; });
    }

    auto lineEqNormDivider = sqrt(pow(line_eq.x, 2) + pow(line_eq.y, 2)) + FLT_EPSILON;

    line_info ret;
    ret.matching_indexes = matching_indexes;
    ret.line_eq_abc = line_eq;
    ret.line_eq_abc_norm = line_eq / lineEqNormDivider;
    ret.bottom_left_edge_point = Point2d(points_intersection[0], points_intersection[1]);
    ret.top_right_edge_point = Point2d(points_intersection[2], points_intersection[3]);
    ret.max_distance = max_distance;
    ret.line_index = line_index;
    return ret;
}

vector<line_info>  cv::separableFundamentalMatrix::getHoughLines(Mat pts, 
    int im_size_w, int im_size_h, int min_hough_points,
    int pixel_res, int theta_res, double max_distance, int num_matching_pts_to_use)
{
    vector<line_info> lineInfos;


    num_matching_pts_to_use = min(pts.size().height, num_matching_pts_to_use);

    Mat ptsRounded;
    pts.convertTo(ptsRounded, CV_32S);

    // Fill map
    Mat bw_img = Mat::zeros(im_size_h, im_size_w, CV_8U);
    for (int addedCount = 0; addedCount < num_matching_pts_to_use; ++addedCount)
    {
        int x0 = ptsRounded.at<int>(addedCount, 1), x1 = ptsRounded.at<int>(addedCount, 0);
        bw_img.at<uint8_t>(x0, x1) = (unsigned short)255;
    }

    vector<Vec2f> houghLines;
    cv::HoughLines(bw_img, houghLines, pixel_res, CV_PI / theta_res, min_hough_points);

    int lineIndex = 0;
    for (auto l : houghLines)
    {
        double rho = l[0], theta = l[1];
        auto p_intersect = findIntersectionPoints(rho, theta, im_size_w, im_size_h);
        if (p_intersect.size() == 4)
        {
            lineInfos.push_back(createLineInfo(pts, p_intersect, max_distance, lineIndex));
            ++lineIndex;
        }
    }

    return lineInfos;
}
