// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef _OPENCV_LINE_HOMOGRAPHY_H_
#define _OPENCV_LINE_HOMOGRAPHY_H_

#include "precomp.hpp"
#include "np_cv_imp.hpp"
#include "matching_points.hpp"

namespace cv {
namespace separableFundamentalMatrix {
/*
    Find 1D Homography

        input : kx4, k is at least 3
        output: The 2x2 (1d) line homography between the points
*/
template <typename _Tp>
Mat findLineHomography(const VecMatchingPoints<_Tp> &matchingPoints)
{
    int numPoints = (int)matchingPoints.size();
    Mat A = Mat::zeros(numPoints, 4, traits::Type<_Tp>::value);

    for (int n = 0; n < numPoints; n++)
    {
        MatchingPoints<_Tp> point = matchingPoints[n];
                
        A.at<_Tp>(n, 0) = point.left.x * point.right.y;
        A.at<_Tp>(n, 1) = point.left.y * point.right.y;
        A.at<_Tp>(n, 2) = - point.left.x * point.right.x;
        A.at<_Tp>(n, 3) = - point.left.y * point.right.x;
    }
    cv::SVD svd(A, SVD::Flags::FULL_UV);
    
    return svd.vt.row(3).reshape(0,2);
}

template <typename _Tp>
void normalizeCoordinatesByLastCol(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    CV_Assert(traits::Type<_Tp>::value == src.type());

    Mat lastCol = src.col(src.cols - 1) + 1e-10; // avoid divid by zero
    _Tp *data = (_Tp *)lastCol.data;
    for (size_t i = 0; i < lastCol.rows; i++)
    {
        data[i] = ((_Tp)1) / data[i];
    }

    Mat ret = matrixVectorElementwiseMultiplication<_Tp>(_src, lastCol);
    _dst.assign(ret);
}
        
/*
    Find the homography error for each matching points
    Returns the error for each data point - the difference between the pixel and its projection
*/
template <typename _Tp>
vector<double> lineHomographyError(Mat model, const VecMatchingPoints<_Tp> &data)
{
    const int lastCol = 1;
    CV_Assert(traits::Type<_Tp>::value == model.type());
    vector<double> ret(data.size(), 100);

    Mat left = data.leftMat();
    Mat right = data.rightMat();
            
    try {
        // Project the points from the left side using the computer homography
        Mat right_H = model * left.t();

        // Project the points from the right side using the inverse of the homography
        Mat left_H = model.inv() * right.t();

        // Normalization
        normalizeCoordinatesByLastCol<_Tp>(right_H.t(), right_H);
        normalizeCoordinatesByLastCol<_Tp>(left_H.t(), left_H);

        // Calculate the error
        for (size_t row = 0; row < data.size(); row++)
        {
            //double error = 0;
            double rightError[2];
            double leftError[2];
                    
            for (size_t col = 0; col < 2; col++)
            {
                rightError[col] = right.at<_Tp>(row,col) - (right_H.at<_Tp>(row, col) * right.at<_Tp>(row, lastCol));
                leftError[col] = left.at<_Tp>(row,col) - (left_H.at<_Tp>(row, col) * left.at<_Tp>(row, lastCol));
            }
            double error = pow(rightError[0], 2) + pow(leftError[0], 2) + pow(rightError[1], 2) + pow(leftError[1], 2);
            error = sqrt(error);
            ret[row] = error;
        }
                
    }
    catch (...) {

    }
    return ret;
}

struct LineInliersModelResult
{
    vector<int> inlierIndexes;
    double meanError;
};

LineInliersModelResult modelInliers(const vector<double> &modelErrors, double inlierTh = 0.35);

struct LineInliersRansacResult
{
    int inlierCount;
    vector<vector<double>> fittestModelsErrors;
};

inline unsigned nChoosek( unsigned n, unsigned k )
{
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}
template <typename _Tp>
LineInliersRansacResult lineInliersRansac(int numIterations, const VecMatchingPoints<_Tp> &matchingPoints, double inlierTh = 0.35)
{
    const int k = 3;
    LineInliersRansacResult result;
    result.inlierCount = 0;

    vector<VecMatchingPoints<_Tp>> samples;
    int nMax = nChoosek(matchingPoints.size(), k);
    if (nMax < numIterations + 50)
    {
        numIterations = nMax;
        size_t max = matchingPoints.size();
        for (size_t a = 0; a < max-2; a++)
        {
            for (size_t b = a+1; b < max-1; b++)
            {
                for (size_t c = b+1; c < max; c++)
                {
                    samples.push_back(matchingPoints.subset(vector<size_t>({ a, b, c })));
                }
            }
        }
    }
    else
    {
        for (size_t i = 0; i < numIterations; i++)
        {
            auto sample = matchingPoints.randomSample(k);
            samples.push_back(sample);
        }
    }

    for (int i = 0; i < numIterations; i++)
    {
        auto sample = samples[i];
        Mat sampleModel = findLineHomography(sample);
        
        vector<double> modelErrors = lineHomographyError(sampleModel, matchingPoints);
        int modelInliers = 0;
        for (auto err : modelErrors)
        {
            if (err < inlierTh)
                ++modelInliers;
        }

        if (modelInliers > result.inlierCount)
        {
            result.fittestModelsErrors.clear();
            result.inlierCount = modelInliers;
        }

        if (modelInliers == result.inlierCount)
        {
            result.fittestModelsErrors.push_back(move(modelErrors));
        }
    }

    return result;

    /*def ransac_get_line_inliers(n_iters,line1_pts,line2_pts,inlier_th=0.35):
        data           = np.concatenate((line1_pts,line2_pts),axis=1)
        random_samples = [random.sample(list(np.arange(len(data))), k=3) for _ in range(n_iters)]
        data_samples   = [data[x, :] for x in random_samples]
        model_samples  = [line_homography(x) for x in data_samples]
        model_errs     = [homography_err(data,model_s) for model_s in model_samples]
        model_inliers  = [np.sum(model_err<inlier_th) for model_err in model_errs]
        best_idx_ransac= np.argmax(model_inliers)
        inliers_idx    = np.arange(len(data))[model_errs[best_idx_ransac]<inlier_th]
        return inliers_idx,np.mean(model_errs[best_idx_ransac][inliers_idx])*/
}
}
}

#endif
