// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_MATCHING_POINTS_H_
#define _OPENCV_MATCHING_POINTS_H_

#include "precomp.hpp"
#include "np_cv_imp.hpp"

namespace cv { 
namespace separableFundamentalMatrix {
using namespace cv;
using namespace std;
    
template <typename _Tp>
struct MatchingPoints
{
    Point_<_Tp> left;
    Point_<_Tp> right;
};

template <typename _Tp>
class VecMatchingPoints {
private:
    vector<Point_<_Tp>> _left;
    vector<Point_<_Tp>> _right;
public:
    VecMatchingPoints() {}
    VecMatchingPoints(const VecMatchingPoints &cpy) : _left(cpy._left), _right(cpy._right) {}
    VecMatchingPoints(const vector<Point_<_Tp>> &left, const vector<Point_<_Tp>> &right) : _left(left), _right(right)
    {
        CV_Assert(left.size() == right.size());
    }
        
    size_t size() const { return _left.size(); }
        
    Mat leftMat() const 
    {
        Mat ret = pointVectorToMat(_left);
        return ret;
    }

    Mat rightMat() const
    {
        Mat ret = pointVectorToMat(_right);
        return ret;
    }

    const MatchingPoints<_Tp> operator[](int index) const
    {
        MatchingPoints<_Tp> ret;
        ret.left = _left[index];
        ret.right = _right[index];
        return ret;
    }

     VecMatchingPoints<_Tp> subset(const vector<size_t> &indices) const
    {
        VecMatchingPoints<_Tp> ret;
        for (auto ix : indices)
        {
            ret._left.push_back(_left[ix]);
            ret._right.push_back(_right[ix]);
        }
        return ret;
    }

    VecMatchingPoints<_Tp> randomSample(uint sizeOfsample) const
    {
        VecMatchingPoints<_Tp> ret;

        std::vector<int> v(size()) ; // vector with N ints.
        iota (v.begin(), v.end(), 0); // Fill with 0, 1, ..., 99.
        random_shuffle(v.begin(), v.end());
            
        for (auto i = 0; i < sizeOfsample; i++)
        {
            int ix = v[i];
            ret._left.push_back(_left[ix]);
            ret._right.push_back(_right[ix]);
        }

        return ret;
    }
};

}
}


#endif // !_OPENCV_MATCHING_POINTS_H_
