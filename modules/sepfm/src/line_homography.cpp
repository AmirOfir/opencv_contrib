// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "line_homography.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

namespace cv {
namespace separableFundamentalMatrix {

void FlattenToMat(const vector<Vec4f> &data, OutputArray _dst)
{
    CV_Assert(data.size());
    Mat mat(data[0]);
    for (size_t i = 1; i < data.size(); i++)
    {
        mat.push_back(data[i]);
    }
    _dst.assign(mat);
}

}
}