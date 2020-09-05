// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_POINTSET_REGISTRATOR_H_
#define _OPENCV_POINTSET_REGISTRATOR_H_

#include "precomp.hpp"

namespace cv {

class CV_EXPORTS PointSetRegistrator : public Algorithm
{
public:
    class CV_EXPORTS Callback
    {
    public:
        virtual ~Callback() {}
        virtual int runKernel(InputArray m1, InputArray m2, OutputArray model) const = 0;
        virtual void computeError(InputArray m1, InputArray m2, InputArray model, OutputArray err) const = 0;
        virtual bool checkSubset(InputArray, InputArray, int) const { return true; }
    };

    virtual void setCallback(const Ptr<PointSetRegistrator::Callback>& cb) = 0;
    virtual bool run(InputArray m1, InputArray m2, OutputArray model, OutputArray mask) const = 0;
};

Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
    int _modelPoints, double _threshold=0, double _confidence=0.99, int _maxIters=1000);

}

#endif // !_OPENCV_POINTSET_REGISTRATOR_H_

