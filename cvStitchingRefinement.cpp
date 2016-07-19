//
//  cvStitchingRefinement.cpp
//  HiResPano
//
//  Created by safdarne on 6/29/16.
//  Copyright © 2016 Adobe. All rights reserved.
//

#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/mat.hpp"

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace cv;



cv::Mat refineStitchingParameters(std::vector<cv::Mat> _storedImages, std::vector<std::vector<cv::KeyPoint>> _storedKeypoints, std::vector<cv::Mat> _storedDescriptors)
{
    Mat result;
    Mat I0  = _storedImages[1];
    
    
    return result;
}