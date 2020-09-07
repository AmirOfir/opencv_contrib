// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_SFM_FINDER_H_
#define _OPENCV_SFM_FINDER_H_

#include "precomp.hpp"
#include "matching_lines.hpp"
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
public:
    Mat points1;
    Mat points2;
    int nPoints;
           
    SeparableFundamentalMatFindCommand(InputArray _points1, InputArray _points2, int _imSizeHOrg, int _imSizeWOrg,
        float _inlierRatio, int _inlierThreashold, float _houghRescale, int _numMatchingPtsToUse, int _pixelRes,
        int _minHoughPints, int _thetaRes, float _maxDistancePtsLine, int _topLineRetries, int _minSharedPoints);
            
    ~SeparableFundamentalMatFindCommand()
    {
        points1.release();
        points2.release();
    }
    
    vector<top_line> FindMatchingLines();

    vector<Mat> FindMat(const vector<top_line> &topMathingLines);

    int CountInliers(Mat f);

    Mat FindMatForInliers(Mat mat);

    Mat TransformResultMat(Mat mat);

};
        
}
}

#endif // !_OPENCV_SFM_FINDER_H_