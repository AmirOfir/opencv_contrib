// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_SFM_FINDER_H_
#define _OPENCV_SFM_FINDER_H_

#include "precomp.hpp"

#include "np_cv_imp.hpp"
#include "matching_points.hpp"
#include "line_homography.hpp"
#include "pointset_registrator.hpp"

using namespace cv;
using namespace std;

namespace cv { 
namespace separableFundamentalMatrix {
    

class SFMEstimatorCallback CV_FINAL : public PointSetRegistrator::Callback
{
    private:
    Mat fixed1;
    Mat fixed2;
public:
    ~SFMEstimatorCallback()
    {
        fixed1.release();
        fixed2.release();
    }
    void setFixedMatrices(InputArray _m1, InputArray _m2);
    bool checkSubset(InputArray _ms1, InputArray _ms2, int count) const CV_OVERRIDE;
    int runKernel(InputArray _m1, InputArray _m2, OutputArray _model) const CV_OVERRIDE;
    void computeError(InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err) const CV_OVERRIDE;
};

class SeparableFundamentalMatFindCommand
{
private:
    int imSizeHOrg;
    int imSizeWOrg;
    float inlierRatio; 
    int inlierThreashold; 
    float houghRescale; 
    int numMatchingPtsToUse; 
    int pixelRes;
    int minHoughPints; 
    int thetaRes;
    float maxDistancePtsLine;
    int topLineRetries;
    int minSharedPoints;
    bool isExecuting;
    Mat points1;
    Mat points2;
    int nPoints;
            
public:
    SeparableFundamentalMatFindCommand(InputArray _points1, InputArray _points2, int _imSizeHOrg, int _imSizeWOrg,
        float _inlierRatio, int _inlierThreashold, float _houghRescale, int _numMatchingPtsToUse, int _pixelRes,
        int _minHoughPints, int _thetaRes, float _maxDistancePtsLine, int _topLineRetries, int _minSharedPoints);
            
    ~SeparableFundamentalMatFindCommand()
    {
        points1.release();
        points2.release();
    }

    Mat FindMat();

    Mat TransformResultMat(Mat mat);

    static vector<top_line> FindMatchingLines(const int im_size_h_org, const int im_size_w_org, cv::InputArray pts1, cv::InputArray pts2,
        const int top_line_retries, float hough_rescale, float max_distance_pts_line, int min_hough_points, int pixel_res,
        int theta_res, int num_matching_pts_to_use, int min_shared_points, float inlier_ratio);

};
        
}
}

#endif // !_OPENCV_SFM_FINDER_H_