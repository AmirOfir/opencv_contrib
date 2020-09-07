// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "SFM_finder.hpp"
#include "pointset_registrator.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

template <typename _Tp>
bool haveCollinearPoints( const Mat& m, int count )
{
    int j, k, i = count-1;
    const Point_<_Tp>* ptr = m.ptr<Point_<_Tp>>();

    // check that the i-th selected point does not belong
    // to a line connecting some previously selected points
    // also checks that points are not too close to each other
    for( j = 0; j < i; j++ )
    {
        _Tp dx1 = ptr[j].x - ptr[i].x;
        _Tp dy1 = ptr[j].y - ptr[i].y;
        for( k = 0; k < j; k++ )
        {
            _Tp dx2 = ptr[k].x - ptr[i].x;
            _Tp dy2 = ptr[k].y - ptr[i].y;
            if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                return true;
        }
    }
    return false;
}

void SFMEstimatorCallback::setFixedMatrices(InputArray _m1, InputArray _m2)
{
    _m1.getMat().convertTo(fixed1, CV_64F);
    _m2.getMat().convertTo(fixed2, CV_64F);
    //cout << _m1.getMat() << endl << fixed1 << endl << _m2.getMat() << fixed2 << endl;
}

bool SFMEstimatorCallback::checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
{
    Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
    return !haveCollinearPoints<double>(ms1, count) && !haveCollinearPoints<double>(ms2, count);
}

int SFMEstimatorCallback::runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
{
    Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    cv::vconcat(fixed1, m1, m1);
    cv::vconcat(fixed2, m2, m2);

    //cout << m1 << endl << m2 << endl;
    Mat F = cv::findFundamentalMat(m1, m2, FM_8POINT);
    
    if (!F.empty() && F.data[0] != NULL)
    {
        if( _model.empty() )
            _model.create(3, 3, CV_64F);
        
        F.convertTo(_model, F.type());
        return true;
    }

    return false;
}

void SFMEstimatorCallback::computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
{
    Mat __m1 = _m1.getMat(), __m2 = _m2.getMat(), __model = _model.getMat();
    int i, count = __m1.checkVector(2);
    _err.create(count, 1, CV_32F);
    float* err = _err.getMat().ptr<float>();
    
    const Point2d* m1 = __m1.ptr<Point2d>();
    const Point2d* m2 = __m2.ptr<Point2d>();
    const double* F = __model.ptr<double>();
    
    for( i = 0; i < count; i++ )
    {
        double a, b, c, d1, d2, s1, s2;

        a = F[0]*m1[i].x + F[1]*m1[i].y + F[2];
        b = F[3]*m1[i].x + F[4]*m1[i].y + F[5];
        c = F[6]*m1[i].x + F[7]*m1[i].y + F[8];

        s2 = 1./(a*a + b*b);
        d2 = m2[i].x*a + m2[i].y*b + c;

        a = F[0]*m2[i].x + F[3]*m2[i].y + F[6];
        b = F[1]*m2[i].x + F[4]*m2[i].y + F[7];
        c = F[2]*m2[i].x + F[5]*m2[i].y + F[8];

        s1 = 1./(a*a + b*b);
        d1 = m1[i].x*a + m1[i].y*b + c;

        err[i] =(float) 0.5 * (sqrt(d1*d1*s1) + sqrt(d2*d2*s2));
        //err[i] = (double)std::max(d1*d1*s1, d2*d2*s2);
    }
}
  

SeparableFundamentalMatFindCommand::SeparableFundamentalMatFindCommand(InputArray _points1, InputArray _points2, int _imSizeHOrg, 
    int _imSizeWOrg, float _inlierRatio, int _inlierThreashold, double _houghRescale, int _numMatchingPtsToUse, int _pixelRes,
    int _minHoughPints, int _thetaRes, float _maxDistancePtsLine, int _topLineRetries, int _minSharedPoints)
    :imSizeHOrg(_imSizeHOrg), imSizeWOrg(_imSizeWOrg), inlierRatio(_inlierRatio), inlierThreashold(_inlierThreashold),
    houghRescale(_houghRescale),  numMatchingPtsToUse(_numMatchingPtsToUse), pixelRes(_pixelRes),
    minHoughPints(_minHoughPints), thetaRes(_thetaRes),  maxDistancePtsLine(_maxDistancePtsLine), topLineRetries(_topLineRetries), 
    minSharedPoints(_minSharedPoints)
{
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);
    
    nPoints = points1.rows;
    if (nPoints == 2 && points1.cols > 2)
    {
        points1 = points1.reshape(2, points1.cols);
        nPoints = points1.rows;
    }
    if (points2.rows == 2 && points2.cols == nPoints)
        points2 = points2.reshape(2, nPoints);
        
    int pts1Count = _points1.isVector() ? _points1.getMat().size().width : _points1.getMat().size().height;
    if (DEFAULT_HOUGH_RESCALE == houghRescale)
        houghRescale = float(2 * pts1Count) / imSizeHOrg;
    else if (houghRescale > 1) // Only subsample
        houghRescale = 1;
}



vector<top_line> SeparableFundamentalMatFindCommand::FindMatchingLines()
{
    houghRescale = houghRescale * 2; // for the first time
    maxDistancePtsLine = maxDistancePtsLine * 0.5;

    Mat pts1Org = points1;
    Mat pts2Org = points2;
    Mat pts1, pts2;
            
    vector<top_line> topMatchingLines;
    // we sample a small subset of features to use in the hough transform, if our sample is too sparse, increase it
    for (auto i = 0; i < this->topLineRetries && topMatchingLines.size() < 2; i++)
    {
        // rescale points and image size for fast line detection
        houghRescale = houghRescale * 0.5;
        maxDistancePtsLine = maxDistancePtsLine * 2;
        pts1 = houghRescale * pts1Org;
        pts2 = houghRescale * pts2Org;
        auto im_size_h = int(round(imSizeHOrg * houghRescale)) + 3;
        auto im_size_w = int(round(imSizeWOrg * houghRescale)) + 3;

        auto linesImg1 = getHoughLines(pts1, im_size_w, im_size_h, minHoughPints, pixelRes, thetaRes, maxDistancePtsLine, numMatchingPtsToUse);
        auto linesImg2 = getHoughLines(pts2, im_size_w, im_size_h, minHoughPints, pixelRes, thetaRes, maxDistancePtsLine, numMatchingPtsToUse);
        
        if (linesImg1.size() && linesImg2.size())
        {
            topMatchingLines =
                getTopMatchingLines(pts1, pts2, linesImg1, linesImg2, minSharedPoints, inlierRatio);
        }
    }

    if (topMatchingLines.size())
    {
        points1 = pts1;
        points2 = pts2;
    }

    return topMatchingLines;
}

vector<Mat> SeparableFundamentalMatFindCommand::FindMat(const vector<top_line> &topMatchingLines)
{
    vector<Mat> ret;

    // We don't have at least one line
    if (!topMatchingLines.size()) return ret;
    
    int maxIterations = 1896;// int((log(0.01) / log(1 - pow(inlierRatio, 5)))) + 1;

    Ptr<SFMEstimatorCallback> cb = makePtr<SFMEstimatorCallback>();
    int result;

    for (auto &topLine : topMatchingLines)
    {
        Mat mask;
        Mat f = Mat(3, 3, CV_64F);

        Mat line_x1n = Mat(topLine.selected_line_points1);
        Mat line_x2n = Mat(topLine.selected_line_points2);
        cb->setFixedMatrices(line_x1n, line_x2n);
        result = createRANSACPointSetRegistrator(cb, 5, 3., 0.99, maxIterations)->run(points1, points2, f, mask);

        if (result > 0)
        {
            ret.push_back(f);
        }
    }
    return ret;
}

int SeparableFundamentalMatFindCommand::CountInliers(Mat f)
{
    SFMEstimatorCallback c;
    Mat err;
    c.computeError(points1, points2, f, err);
    float *eptr = err.ptr<float>();

    // Count inliers
    int inlierCount = 0;
    for (size_t i = 0; i < err.rows; i++)
    {
        if (eptr[i] < inlierThreashold)
        {
            ++inlierCount;
        }
    }

    return inlierCount;
}

Mat SeparableFundamentalMatFindCommand::FindMatForInliers(Mat mat)
{
    SFMEstimatorCallback c;
    Mat err;
    c.computeError(points1, points2, mat, err);
    float *eptr = err.ptr<float>();

    // Count inliers
    int inlierCount = 0;
    for (size_t i = 0; i < err.rows; i++)
    {
        if (eptr[i] < inlierThreashold)
        {
            ++inlierCount;
        }
    }

    // Create matrices with the inliers
    Mat p1;
    Mat p2;

    for (size_t i = 0; i < err.rows; i++)
    {
        if (eptr[i] < inlierThreashold)
        {
            p1.push_back(points1.row(i));
            p2.push_back(points2.row(i));
        }
    }
    points1 = p1;
    points2 = p2;

    // Compute fundamental matrix
    Mat f = cv::findFundamentalMat(p1, p2, noArray(), FM_8POINT);
    return f;
}

Mat SeparableFundamentalMatFindCommand::TransformResultMat(Mat mat)
{
    Mat diag = Mat::zeros(3, 3, CV_64F);
    diag.at<double>(0, 0) = houghRescale;
    diag.at<double>(1, 1) = houghRescale;
    diag.at<double>(2, 2) = 1;

    Mat ret = mat * diag;
    ret = diag * ret;
    return ret;
}

// pts1 is Mat of shape(X,2)
// pts2 is Mat of shape(X,2)
Mat cv::separableFundamentalMatrix::findSeparableFundamentalMat(InputArray _points1, InputArray _points2, int _imSizeHOrg, int _imSizeWOrg,
        float _inlierRatio, int _inlierThreashold, double _houghRescale, int _numMatchingPtsToUse, int _pixelRes,
        int _minHoughPints, int _thetaRes, float _maxDistancePtsLine, int _topLineRetries, int _minSharedPoints)
{
    SeparableFundamentalMatFindCommand command(_points1, _points2, _imSizeHOrg, _imSizeWOrg, _inlierRatio, _inlierThreashold, _houghRescale,
        _numMatchingPtsToUse, _pixelRes, _minHoughPints, _thetaRes, _maxDistancePtsLine, _topLineRetries, _minSharedPoints);

    auto topMatchingLines = command.FindMatchingLines();
    
    vector<Mat> matrices = command.FindMat(topMatchingLines);
    int bestInliersCount = 0;
    Mat bestInliersMat;
    for (auto m: matrices)
    {
        int inliersCount = command.CountInliers(m);
        if (bestInliersCount < inliersCount)
        {
            bestInliersCount = inliersCount;
            bestInliersMat = m;
        }
    }

    bestInliersMat = command.FindMatForInliers(bestInliersMat);

    Mat f;
    f = command.TransformResultMat(bestInliersMat);

    return f;
}