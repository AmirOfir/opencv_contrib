// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_LINE_GEOMETRY_H_
#define _OPENCV_LINE_GEOMETRY_H_

#include "precomp.hpp"

namespace cv {
namespace separableFundamentalMatrix {
using namespace cv;
using namespace std;

struct IntervalEndpointsResult
{
    double distance;
    int firstIdx;
    int secondIdx;
};

/** lineEndpoints
@param a input vector of points
@param distance - the eucalidean distance between the end points.
@param firstIdx - the first point location.
@param secondIdx - the second point location.
*/
template <typename _Tp>
IntervalEndpointsResult intervalEndpoints(const vector<Point_<_Tp>> &points)
{
    CV_Assert(points.size() > 2);
    int aIdx = 0, bIdx = 1;
    double maxDist = norm(points[0] - points[1]);
    for (size_t i = 0; i < points.size() - 1; i++)
    {
        for (size_t j = i + 1; j < points.size(); j++)
        {
            double currDist = norm(points[i] - points[j]);

            if (currDist > maxDist)
            {
                aIdx = i;
                bIdx = j;
                maxDist = currDist;
            }
        }
    }
            
    IntervalEndpointsResult ret;
    ret.distance = maxDist;
    ret.firstIdx = aIdx;
    ret.secondIdx = bIdx;
    return ret;
}


struct IntervalMidPointResult
{
    int midPointIdx;
    double minDistance;
};

template <typename _Tp>
IntervalMidPointResult intervalPointClosestToCenter(const vector<Point_<_Tp>> &points, int endpoint1Idx, int endpoint2Idx)
{
    CV_Assert(points.size() > 2);

    Point_<_Tp> start = points[endpoint1Idx], end = points[endpoint2Idx];
    double minDist = norm(end - start);
    double medianIdx = 0;
            
    for (size_t i = 0; i < points.size(); i++)
    {
        if (i != endpoint1Idx && i != endpoint2Idx)
        {
            double currDist = abs(norm(points[i] - start) - norm(points[i] - end));
            if (currDist < minDist)
            {
                minDist = currDist;
                medianIdx = i;
            }
        }
    }

    // The min distance from the median point to the endpoints
    minDist = min(norm(points[medianIdx] - start), norm(points[medianIdx] - end));

    IntervalMidPointResult ret;
    ret.midPointIdx = medianIdx;
    ret.minDistance = minDist;
    return ret;
}

}
}

#endif