//
//  CvEstimator.cpp
//  Opencv-directRotationComp
//
//  Created by safdarne on 6/2/16.
//  Copyright © 2016 Adobe. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <ctime>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/mat.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>


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


#include "mvg/rotation3_est.h"
#include "math/CubicSolver.h"
#include "mvg/robust_cost_function.h"
#include "mvg/Progress_Monitor.h"
#include "mvg/message_reporter.h"
#include <utility/iterator_adaptor.h>

#include <stdio.h>
#include <string>



using namespace cv;
using namespace cv::detail;


class CvEstimator
{
public:
    CvEstimator();
    ~CvEstimator();
private:
    cv::Mat _prevImage;
    cv::Mat _currentImage;
    std::vector<cv::KeyPoint> _prevKeypoints, _currentKeypoints;
    cv::UMat _prevDescriptors, _currentDescriptors;
    std::vector<float> _RFromImageRobust;
    Ptr<Feature2D> _featureDetector;
    cv::Mat _K;
    Mat _RFromSensor;
    std::vector<cv::Mat> _storedImages;
    std::vector<std::vector<cv::KeyPoint>> _storedKeypoints;
    std::vector<cv::UMat> _storedDescriptors;
    std::vector<cv::Mat> _refinedRotationArray;
    cv::Mat H;
    cv::Mat initH;
    std::vector<cv::detail::MatchesInfo> _pairwiseMatches;
    std::vector<ImageFeatures> _features;
    std::vector<CameraParams> _cameras;
    
    float _focal;
    int _width;
    int _height;
    
    double xCanvasF;
    double yCanvasF;
    std::vector<float> gH;
    float roll;
    float pitch;
    float yaw;
    
    void putImageInCanvas (cv::Mat &image, double xCanvasF, double yCanvasF, cv::Mat &initH, bool resizeFlag);
    std::vector< std::vector<cv::Point>>  overlayFrame(cv::Mat &imgA, cv::Mat &WarpImg, cv::Mat &WarpImgAccu, cv::Mat accuH, int mode);
    void overlayFrameWithMask(cv::Mat &imgA, cv::Mat &imgB, cv::Mat &maskA, cv::Mat &maskB, std::vector<cv::Point> &corners, cv::Mat &overlaid);
    cv::Mat computeRotationMatrixFromEulerAngles(float roll, float pitch, float yaw);
    void computeHomographyFromRotation(cv::Mat R, float focal, cv::Point2d principalPoint, cv::Mat &H, cv::Mat &K);
    void findRotationFromKeypointMatches(std::vector<cv::Point2f> &cur, std::vector<cv::Point2f> &prev, cv::Mat &K, cv::Mat &R);
    void robustKeypointMatching(int width, int height, cv::Mat &HFromSensor, std::vector<cv::KeyPoint> &prevKeypoints, std::vector<cv::KeyPoint> &currentKeypoints, cv::UMat &prevDescriptors, cv::UMat &currentDescriptors, std::vector< cv::DMatch > &good_matches, std::vector<cv::Point2f> &prev, std::vector<cv::Point2f> &cur, cv::Mat &Mask);
    cv::Vec3d estimateEulerAngles(std::vector<cv::Point2f> &cur, std::vector<cv::Point2f> &prev, float focal, int height, int width);
    void drawFinalMatches(cv::Mat &prevImage, std::vector<cv::KeyPoint> &prevKeypoints, cv::Mat &currentImage, std::vector<cv::KeyPoint> &currentKeypoints, std::vector< cv::DMatch > &good_matches, cv::Mat &Mask, cv::Mat &result);
    void copyBAResultToCamerArray(int num_images, std::vector<CameraParams> &cameras);
    void copyRotationToCameraArray(std::vector<Mat> &currentRotationUsedToInitBA, std::vector<CameraParams>& cameras);
    
public:
    void saveNextFrame(cv::Mat &nextFrame);
    void procFrame(cv::Mat &nextFrame, bool flag);
    void initialize(int width, int height);
    std::vector<float> getRotation();
    void setRotationFromSensor(std::vector<float> &rot);
    void restart();
    std::vector<cv::Mat> refinedStitching(std::vector<Mat> &currentRotationUsedToInitBA, const std::vector<std::vector<int>> &closestFrames);
    float getFocalLength();
    
};


CvEstimator::CvEstimator():_focal(515)
{
    // In Class Pano (display module), the index 0 is reserved for camera live feed, so, start from index 1 for compatibility.
    std::vector<cv::KeyPoint> temp;
    _storedImages.push_back(cv::Mat());
    _storedKeypoints.push_back(temp);
    _storedDescriptors.push_back(cv::UMat());
}

CvEstimator::~CvEstimator()
{
}

float CvEstimator::getFocalLength()
{
    return _focal;
}

void CvEstimator::putImageInCanvas (cv::Mat &image, double xCanvasF, double yCanvasF, cv::Mat &initH, bool resizeFlag)
{
    float gH2[9]={1,0, static_cast<float>((xCanvasF/2 - 0.5) * image.cols), 0,1, static_cast<float>((yCanvasF/2 - 0.5) * image.rows), 0,0,1};
    cv::Mat temp = cv::Mat(3, 3, CV_32FC1, gH2);
    temp.copyTo(initH);
    cv::Mat tempImage;
    warpPerspective(image, tempImage, initH, cv::Size(xCanvasF * image.cols, yCanvasF * image.rows));
    if (resizeFlag)
        resize(tempImage, image, cv::Size(image.cols, image.rows));
    else
        tempImage.copyTo(image);
}


std::vector< std::vector<cv::Point>>  CvEstimator::overlayFrame(cv::Mat &imgA, cv::Mat &WarpImg, cv::Mat &WarpImgAccu, cv::Mat accuH, int mode) {
    // First finds a mask showing the area for which the new warped frame has values, and then,
    // replaces the values on the overlay ("WarpImgAccu") with the values from this new frame ("WarpImg")
    cv::Mat mask(WarpImg.rows, WarpImg.cols, CV_8UC1, cv::Scalar(0));
    
    cv::Point P1(1,1);
    cv::Point P2(1, imgA.rows-1-1);
    cv::Point P3(imgA.cols-1-1,imgA.rows-1-1);
    cv::Point P4(imgA.cols-1-1,1);
    
    std::vector< std::vector<cv::Point>> co_ordinates;
    co_ordinates.push_back(std::vector<cv::Point>());
    std::vector<cv::Point2f> vec;
    vec.push_back(P1);
    vec.push_back(P2);
    vec.push_back(P3);
    vec.push_back(P4);
    perspectiveTransform(vec, vec, accuH);
    co_ordinates[0].push_back(vec[0]);
    co_ordinates[0].push_back(vec[1]);
    co_ordinates[0].push_back(vec[2]);
    co_ordinates[0].push_back(vec[3]);
    drawContours(mask,co_ordinates,0, cv::Scalar(255),CV_FILLED, 8);
    
    switch (mode) {
        case 0:
            WarpImg.copyTo(WarpImgAccu, mask);
            break;
        case 1:
            WarpImg.copyTo(WarpImgAccu);
            break;
    }
    //drawContours(WarpImgAccu,co_ordinates,0, Scalar(255), 1, 8);
    return co_ordinates;
}

void CvEstimator::overlayFrameWithMask(cv::Mat &imgA, cv::Mat &imgB, cv::Mat &maskA, cv::Mat &maskB, std::vector<cv::Point> &corners, cv::Mat &overlaid) {
    //cv::detail::Blender temp;
    //cv::Ptr<cv::detail::Blender> blender;
    /*
     int blend_type = cv::detail::FEATHER;
     float blend_strength = 50;
     
     std::vector<cv::Size> sizes(2);
     sizes[0].width = imgA.cols;
     sizes[0].height = imgA.rows;
     sizes[1].width = imgB.cols;
     sizes[1].height = imgB.rows;
     
     cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
     float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
     
     blender = Blender::createDefault(blend_type);
     
     cv::detail::blender::FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
     fb->setSharpness(1.f/blend_width);
     
     blender->prepare(corners, sizes);
     
     
     cv::Mat imgA_s, imgB_s;
     
     cvtColor(imgA,imgA_s, cv::COLOR_RGBA2BGR);
     cvtColor(imgB,imgB_s, cv::COLOR_RGBA2BGR);
     
     
     
     imgA_s.convertTo(imgA_s, CV_16S);
     imgB_s.convertTo(imgB_s, CV_16S);
     
     
     
     //std::cout << imgA_s.type() << std::endl;
     //std::cout << imgA_s.channels() << std::endl;
     
     blender->feed(imgA_s, maskA, corners[0]);
     blender->feed(imgB_s, maskB, corners[1]);
     
     
     cv::Mat result_mask;
     blender->blend(overlaid, result_mask);
     overlaid.convertTo(overlaid, CV_8U);*/
}



cv::Mat CvEstimator::computeRotationMatrixFromEulerAngles(float roll, float pitch, float yaw) {
    cv::Mat R = cv::Mat::zeros(3, 3, CV_64FC1);
    
    cv::Mat XX = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Mat YY = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Mat ZZ = cv::Mat::zeros(3, 3, CV_64FC1);
    
    XX.at<double>(0,0) = 1;
    XX.at<double>(1,1) = cos(roll);
    XX.at<double>(1,2) = -sin(roll);
    XX.at<double>(2,1) = sin(roll);
    XX.at<double>(2,2) = cos(roll);
    
    YY.at<double>(1,1) = 1;
    YY.at<double>(0,0) = cos(pitch);
    YY.at<double>(0,2) = sin(pitch);
    YY.at<double>(2,0) = -sin(pitch);
    YY.at<double>(2,2) = cos(pitch);
    
    ZZ.at<double>(2,2) = 1;
    ZZ.at<double>(0,0) = cos(yaw);
    ZZ.at<double>(0,1) = -sin(yaw);
    ZZ.at<double>(1,0) = sin(yaw);
    ZZ.at<double>(1,1) = cos(yaw);
    
    R = ZZ * YY * XX;
    return R;
}


void CvEstimator::computeHomographyFromRotation(cv::Mat R, float focal, cv::Point2d principalPoint, cv::Mat &H, cv::Mat &K) {
    K = cv::Mat::zeros(3, 3, CV_64FC1);
    K.at<double>(0,0) = focal;
    K.at<double>(1,1) = focal;
    K.at<double>(2,2) = 1.0;
    
    K.at<double>(0,2) = principalPoint.x;
    K.at<double>(1,2) = principalPoint.y;
    
    cv::Mat temp2 = K * R * K.inv();
    temp2.copyTo(H);
}


void CvEstimator::findRotationFromKeypointMatches(std::vector<cv::Point2f> &cur, std::vector<cv::Point2f> &prev, cv::Mat &K, cv::Mat &R) {
    
    cv::Mat F = findFundamentalMat(cur, prev, cv::FM_RANSAC, 3., 0.97);
    
    cv::Mat E = K.t() * F * K;
    
    
    cv::SVD svd = cv::SVD(E);
    cv::Matx33d W(0,-1,0,   //HZ 9.13
                  1,0,0,
                  0,0,1);
    cv::Matx33d Winv(0,1,0,
                     -1,0,0,
                     0,0,1);
    
    
    R = svd.u * cv::Mat(W) * svd.vt; //HZ 9.19
    ////Mat t = svd.u.col(2); //u3
    
    /*
     P1 = Matx34d(R(0,0),    R(0,1), R(0,2), t(0),
     R(1,0),    R(1,1), R(1,2), t(1),
     R(2,0),    R(2,1), R(2,2), t(2));
     */
}


void CvEstimator::robustKeypointMatching(int width, int height, Mat &HFromSensor, std::vector<KeyPoint> &prevKeypoints, std::vector<KeyPoint> &currentKeypoints, cv::UMat &prevDescriptors, cv::UMat &currentDescriptors, std::vector< DMatch > &good_matches, std::vector<Point2f> &prev, std::vector<Point2f> &cur, Mat &Mask) {
    
    
    std::vector< DMatch > wellDistributed;
    std::vector<DMatch> matches, sortedMatches;
    std::vector<Point2f> curTransformed;
    
    Ptr<DescriptorMatcher>  descriptorMatcher = DescriptorMatcher::create(String("BruteForce"));
    descriptorMatcher->match(prevDescriptors, currentDescriptors, matches, Mat());
    
    if (matches.size() > 0)
    {
        
        Mask = Mat::zeros(Mask.rows, Mask.cols, Mask.type());
        
        
        // -- Rejection of keypoints matches based on device motion sensor result and distribution of keypoints. First, keypoints are sorted based on the match strength. Then, take well distributed keypoints, and then, check if the keypoints transformed under estimated transformation from motion sensor match well with matched keypoints.
        Mat index;
        int nbMatch=int(matches.size());
        
        Mat tab(nbMatch, 1, CV_32F);
        for (int i = 0; i < nbMatch; i++) {
            tab.at<float>(i, 0) = matches[i].distance;
        }
        sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
        
        for (int i = 0; i < nbMatch; i++) {
            sortedMatches.push_back(matches[index.at<int>(i, 0)]);
        }
        
        matches = sortedMatches;
        sortedMatches.clear();
        
        
        for (int i = 0; i < matches.size(); i++){
            Point2d currCoord = prevKeypoints[matches[i].queryIdx].pt;
            int p = 5; // fixme, neighborhoood should be larger than 0, and dependent on resolution
            int xCorner = min(max(0.0, round(currCoord.x) - p), double(height) - 1);
            int yCorner = min(max(0.0, round(currCoord.y) - p), double(width) - 1);
            int w = min(double(height - xCorner), double(2*p));
            int h = min(double(width - yCorner), double(2*p));
            
            Mat roi = Mask(cv::Rect(xCorner, yCorner, w, h));
            if (Mask.at<char>(round(currCoord.y), round(currCoord.x)) == 0) { // Only push in, if there is not a keypoint nearby already
                prev.push_back(currCoord);
                cur.push_back(currentKeypoints[matches[i].trainIdx].pt);
                wellDistributed.push_back(matches[i]);
                roi.setTo(Scalar(1));
            }
            //else
            //printf("Thrown away %d\n", Mask.at<char>(round(currCoord.y), round(currCoord.x)));
        }
        
        // If there is no motion sensor estimation, avoid filtering by copying the source keypoints to the destination keypoints
        //std::cout << HFromSensor << std::endl;
        if (HFromSensor.rows > 0) //// fixme
            perspectiveTransform(cur, curTransformed, HFromSensor);
        else
            curTransformed = prev;
        
        for (int i = 0; i < prev.size(); i++){
            if (norm(prev[i] - curTransformed[i]) < 50) //fixme
                good_matches.push_back(wellDistributed[i]);
            //else
            //printf("Thrown away 2\n");
        }
        prev.clear();
        cur.clear();
        for( int i = 0; i < good_matches.size(); i++ ){
            //-- Get the keypoints from the good matches
            prev.push_back(prevKeypoints[good_matches[i].queryIdx].pt );
            cur.push_back(currentKeypoints[good_matches[i].trainIdx].pt );
        }
    }
    else
        std::cout << "Could not match" << std::endl;
}

cv::Vec3d CvEstimator::estimateEulerAngles(std::vector<cv::Point2f> &cur, std::vector<cv::Point2f> &prev, float focal, int height, int width) {
    //-- Using Essential matrix, find roll, pitch, and yaw from matching keypoitns
    // http://stackoverflow.com/questions/31447128/camera-pose-estimation-how-do-i-interpret-rotation-and-translation-matrices
    Mat t;
    Mat R2;
    
    Point2d principalPoint = Point2d(height/2,width/2);
    Mat E = findEssentialMat(cur, prev, focal, principalPoint, RANSAC);
    int inliers = recoverPose(E, cur, prev, R2, t, focal, principalPoint);
    Mat mtxR, mtxQ;
    Mat Qx, Qy, Qz;
    Vec3d angles = RQDecomp3x3(R2, mtxR, mtxQ, Qx, Qy, Qz);
    return angles;
}


void CvEstimator::drawFinalMatches(cv::Mat &prevImage, std::vector<cv::KeyPoint> &prevKeypoints, cv::Mat &currentImage, std::vector<cv::KeyPoint> &currentKeypoints, std::vector< cv::DMatch > &good_matches, cv::Mat &Mask, cv::Mat &result) {
    cv::Mat matrix;
    prevImage.convertTo(matrix, CV_32FC1, 1/255.0);
    Mask.convertTo(Mask, CV_32FC1);
    
    multiply(matrix, Mask/2 + 0.5, matrix);
    matrix.convertTo(matrix, CV_8UC1, 255);
    
    drawMatches(matrix, prevKeypoints, currentImage, currentKeypoints, good_matches, result, cv::Scalar(0,255,0));
}

void CvEstimator::saveNextFrame(Mat &nextFrame)
{
    nextFrame.copyTo(_prevImage);
    cv::cvtColor(_prevImage, _prevImage, CV_BGR2GRAY);
    
    
    _featureDetector ->detect(_prevImage, _prevKeypoints, Mat());
    _featureDetector ->compute(_prevImage, _prevKeypoints, _prevDescriptors);
    
    _storedImages.push_back(nextFrame);
    _storedKeypoints.push_back(_prevKeypoints);
    _storedDescriptors.push_back(_prevDescriptors);
    
}

template<typename Iter_T>
long double vectorNorm(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0L));
}


void CvEstimator::procFrame(Mat &currentImage, bool flag)
{
    bool success = true;
    float normOfDiff = 0;
    
    cv::cvtColor(currentImage, currentImage, CV_BGR2GRAY);
    
    if (flag)
    {
        if (_prevImage.rows > 0)
        {
            _featureDetector ->detect(currentImage, _currentKeypoints, Mat());
            
            if (_currentKeypoints.size() > 20)
            {
                Mat Mask = Mat::zeros(_height, _width, CV_8UC1);
                Mat HFromSensor;
                Mat RFromImageRobust;
                std::vector< DMatch > good_matches, wellDistributed;
                std::vector<DMatch> matches, sortedMatches;
                std::vector<Point2f> prev, cur, curTransformed;
                std::vector<float> x1, x2, y1, y2;
                
                // Describe keypoints
                _featureDetector ->compute(currentImage, _currentKeypoints, _currentDescriptors);
                
                // Compute homograpjhy from rotaion matrix via the motion sensor information, to be used to rectify the keypoint matches
                computeHomographyFromRotation(_RFromSensor,  _focal, cv::Point2d(_width / 2, _height / 2) , HFromSensor, _K);
                
                
                // Match keypoints and rectify them using sensor information
                robustKeypointMatching(_width, _height, HFromSensor, _prevKeypoints, _currentKeypoints, _prevDescriptors, _currentDescriptors, good_matches, prev, cur, Mask);
                
                // Use matched keypoint coordinates to calculate focal length and rotation matrix
                for (int i = 0; i < cur.size(); i++) {
                    x1.push_back(prev[i].x);
                    y1.push_back(prev[i].y);
                    x2.push_back(cur[i].x);
                    y2.push_back(cur[i].y);
                }
                
                adobe_agt::mvg::rotation3_fl_est_2view_2point_ransac<float> rotation_fl_solver(x1.size(),
                                                                                               (&x1.front()),
                                                                                               (&y1.front()),
                                                                                               (&x2.front()),
                                                                                               (&y2.front()),
                                                                                               _focal, _width / 2, _height / 2, 1,
                                                                                               _focal, _width / 2, _height / 2, 1,
                                                                                               1,
                                                                                               std::size_t(2000),
                                                                                               double(.999),
                                                                                               rand
                                                                                               );
                
                
                if (! rotation_fl_solver.is_failed())
                {
                    //_focal = 0.9 * _focal + 0.1 * (rotation_fl_solver.focal_length() * _focal);
                    //std::cout << rotation_fl_solver.focal_length() * _focal <<  std::endl;
                    int jj = 0;
                    
                    for (auto it = rotation_fl_solver.rotation_begin(); it < rotation_fl_solver.rotation_end(); it++)
                    {
                        _RFromImageRobust[jj++] = *it;
                    }
                }
                
                
                // Only use rotation_fl_solver results if they are close to what the motion sensor says
                for (int jj = 0; jj <9 ; jj++)
                {
                    float temp  = _RFromImageRobust[jj] - _RFromSensor.at<double>(floor(jj / 3), (jj % 3));
                    normOfDiff += temp*temp;
                }
                normOfDiff = sqrt(normOfDiff);
                
                if (rotation_fl_solver.is_failed() || (normOfDiff > 1.1))
                {
                    //std::cout << "failed! :(      ,      " << normOfDiff <<  std::endl;
                    success = false;
                }
            }
            else
            {
                //std::cout << "Not enough matches" << std::endl;
                success = false;
            }
        }
        else
        {
            // std::cout << "Previous image not found" << std::endl;
            success = false;
        }
    }
    else
        success = false;
    
    
    
    
    
    
    ////success = false; /////!!!!!! fixme, set to false to force only motion sensor to be used
    if (!success)
    {
        for (int jj = 0; jj <9 ; jj++)
        {
            _RFromImageRobust[jj] = _RFromSensor.at<double>(floor(jj / 3), (jj % 3));
        }
    }
    //else
    //std::cout << "Success! :)      ,      " << normOfDiff <<  std::endl;
    
}

void CvEstimator::initialize(int width, int height)
{
    _featureDetector = ORB::create(1000, 1.2f, 3, 31, 0, 2, ORB::HARRIS_SCORE, 31, 10);
    _width = width;
    _height = height;
    
    _RFromSensor = Mat::zeros(3, 3, CV_64FC1);
    for (int i = 0; i < 9; i++){
        _RFromImageRobust.push_back(0);
        _RFromSensor.at<double>(floor(i / 3), i % 3) = 0;
    }
    _RFromImageRobust[0] = 1;
    _RFromImageRobust[4] = 1;
    _RFromImageRobust[8] = 1;
    
    
    
    _K = cv::Mat::zeros(3, 3, CV_64FC1);
    _K.at<double>(0,0) = _focal;
    _K.at<double>(1,1) = _focal;
    _K.at<double>(2,2) = 1.0;
    
    _K.at<double>(0,2) = _width / 2;
    _K.at<double>(1,2) = _height / 2;
    
}

std::vector<float> CvEstimator::getRotation()
{
    //_RFromImageRobust has rotation estimated from image features between the last reference frame and the current frame,
    // it should be converted first to a global rotation by compositing the rotations calculated up to now together
    return _RFromImageRobust;
}

void CvEstimator::setRotationFromSensor(std::vector<float> &rot)
{
    for (int i = 0; i < 9; i++ )
        _RFromSensor.at<double>(floor(i / 3), i % 3) = rot[i];
}

void CvEstimator::restart()
{
    _storedImages.clear();
    _storedKeypoints.clear();
    _storedDescriptors.clear();
    
    // In Class Pano (display module), the index 0 is reserved for camera live feed, so, start from index 1 for compatibility.
    std::vector<cv::KeyPoint> temp;
    _storedImages.push_back(cv::Mat());
    _storedKeypoints.push_back(temp);
    _storedDescriptors.push_back(cv::UMat());
    
    _pairwiseMatches.clear();
    _features.clear();
}
///////////////////////////////////////////////////////////////////////////////////////////////
void CvEstimator::copyRotationToCameraArray(std::vector<Mat> &currentRotationUsedToInitBA, std::vector<CameraParams>& cameras)
{
    for (size_t i = 0; i < currentRotationUsedToInitBA.size() - 1; ++i)
    {
        Mat R;
        
        // Use rotation matrix computed from previous stages (which is corrected by the device sensor) as the initial condition for the bundle adjustment
        // The way rotation matrices are used for display is different from what calculated in bundle adjustment, some flipping and rotation is needed to make them consistent
        currentRotationUsedToInitBA[i + 1].copyTo(R);
        
        // Flip x in input images
        for (int j = 0; j < 3; j++) {
            R.at<float>(j,0) = -1 * R.at<float>(j,0);
        }
        
        // This is equal to 180 degree rotation relative to the y axis to bring back the pixels in 3d space to the front of camera after fliping in z direction
        for (int j = 0; j < 3; j++) {
            R.at<float>(0, j) = -1 * R.at<float>(0, j);
            R.at<float>(2, j) = -1 * R.at<float>(2, j);
        }
        R = R.inv();
        
        // Flip in z direction for consistency
        for (int j = 0; j < 3; j++) {
            R.at<float>(2, j) = -1 * R.at<float>(2, j);
        }
        
        if (i >= cameras.size())
            cameras.resize(i + 1);
        
        R.convertTo(R, CV_32FC1);
        
        //std::cout << cameras[i].R << std::endl;
        R.copyTo(cameras[i].R);
        //std::cout << cameras[i].R << std::endl;
        //std::cout << "-----------------" << std::endl;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////
void CvEstimator::copyBAResultToCamerArray(int num_images, std::vector<CameraParams> &cameras)
{
    
    for (int i = 1; i <= num_images; i++) { // Index 0 is reserved for live camera feed, first one is also set to identity
        if (i >= _refinedRotationArray.size())
            _refinedRotationArray.resize(i + 1);
        
        cv::Mat rotMat;
        cameras[(i - 1)].R.copyTo(rotMat);
        
        // Flip in z direction for consistency
        for (int j = 0; j < 3; j++) {
            rotMat.at<float>(2, j) = -1 * rotMat.at<float>(2, j);
        }
        rotMat = rotMat.inv();
        
        // This is equal to 180 degree rotation relative to the y axis to bring back the pixels in 3d space to the front of camera after fliping in z direction
        for (int j = 0; j < 3; j++) {
            rotMat.at<float>(0, j) = -1 * rotMat.at<float>(0, j);
            rotMat.at<float>(2, j) = -1 * rotMat.at<float>(2, j);
        }
        
        // Flip x in input images
        for (int j = 0; j < 3; j++) {
            rotMat.at<float>(j,0) = -1 * rotMat.at<float>(j,0);
        }
        if ( i > 1)
            rotMat = _refinedRotationArray[1].inv() * rotMat;
        
        rotMat.copyTo(_refinedRotationArray[i]);
    }
    
    _refinedRotationArray[1] = Mat::eye(3, 3, _refinedRotationArray[1].type());
}
///////////////////////////////////////////////////////////////////////////////////////////////
std::vector<cv::Mat> CvEstimator::refinedStitching(std::vector<Mat> &currentRotationUsedToInitBA, const std::vector<std::vector<int>> &closestFrames)
{
    int MAXMATCHES = 100;
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Refinement started" << std::endl;
    int num_images = int(_storedImages.size())  - 1;
    
    HomographyBasedEstimator estimator;
    std::vector<CameraParams> cameras;
    
    Ptr<detail::BundleAdjusterBase> adjuster;
    adjuster = new detail::BundleAdjusterRay();
    float conf_thresh = 1.f;
    adjuster->setConfThresh(conf_thresh);
    std::string ba_refine_mask = "xxxxx";
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    
    
    float match_conf = 0.5f;
    BestOf2NearestMatcher matcher(false, match_conf);
    
    if ((num_images *  num_images) > _pairwiseMatches.size()) // There is new unprocessed frame, match keypoints and do BA, otherwise, only do BA
    {
        clock_t begin = clock();
        // If pairwise matches not formed yet, start from scratch, otherwise incrementally update it with each new frame captured
        if (_pairwiseMatches.size() < 9) // Do not do incremental for the first 3 frames
        {
            for (int i = 0; i < num_images; i++)
            {
                ImageFeatures tempFeatures;
                
                tempFeatures.img_idx = i;
                tempFeatures.img_size = Size(_width, _height);
                tempFeatures.keypoints = _storedKeypoints[i+1];
                tempFeatures.descriptors = _storedDescriptors[i+1];
                _features.push_back(tempFeatures);
                
                _cameras.push_back(CameraParams()); // Push empty camera param, it is going to be replaced with sensor data estimation
            }
            
            matcher(_features, _pairwiseMatches);
            matcher.collectGarbage();
        }
        else
        {
            // Match the last frame with all the previous frames in its neighborhood
            Ptr<DescriptorMatcher>  descriptorMatcher = DescriptorMatcher::create(String("BruteForce"));
            
            // For better performance, initialize camera matrix with the data from motion sensor
            copyRotationToCameraArray(currentRotationUsedToInitBA, cameras);
            
            for (int i = 0; i < num_images - 1; i++)
            {
                // Check all frames, see of they are in the neighborhood of the current frame which is indexed "num_images", and match them if they are adjacent
                std::vector<DMatch> matches;
                std::vector<Point2f> prev, cur;
                cv::detail::MatchesInfo tempMatchesInfo;
                std::vector<uchar> inliers_mask;
                bool frameShouldBeProcessed = false;
                
                tempMatchesInfo.src_img_idx = i;
                tempMatchesInfo.dst_img_idx = num_images - 1;
               
                
                if ( (i + 1) == (closestFrames[num_images][0]) || (i + 1) == (closestFrames[num_images][1]) || (i + 1) == (closestFrames[num_images][2]) || (i + 1) == (closestFrames[num_images][3]) ) // check against 4 nearest neighbors
                    frameShouldBeProcessed = true;
                
                if ( frameShouldBeProcessed )
                {
                    // Compute homography from sensor data
                    Mat HFromSensor, K;
                    Mat R = Mat::zeros(3, 3, CV_64FC1);
                    Mat Mask = Mat::zeros(_height, _width, CV_8UC1);
                    
                    R = cameras[ num_images - 1 ].R * cameras[ i ].R.inv(); // Remember index 0 is reserved for camera frame
                    R.convertTo(R, CV_64F);
                    computeHomographyFromRotation( R, _focal, cv::Point2d(_width / 2, _height / 2), HFromSensor, K );
                    
                    // Match keypoints
                     descriptorMatcher -> match( _storedDescriptors[i+1], _storedDescriptors[num_images], matches, Mat() );
                    std::vector<DMatch> sortedMatches;
                    Mat index;
                    int nbMatch=int(matches.size());
                    
                    // Only store a maximum of 100 strongest matches (MAXMATCHES)
                    Mat tab(nbMatch, 1, CV_32F);
                    for (int i = 0; i < nbMatch; i++) {
                        tab.at<float>(i, 0) = matches[i].distance;
                    }
                    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
                    for (int i = 0; i < min(nbMatch, MAXMATCHES); i++) {
                        sortedMatches.push_back(matches[index.at<int>(i, 0)]);
                    }
                    matches = sortedMatches;
                    sortedMatches.clear();
                    
                    //robustKeypointMatching(_width, _height, HFromSensor,  _storedKeypoints[i+1], _storedKeypoints[num_images],  _storedDescriptors[i+1], _storedDescriptors[num_images], matches, prev, cur, Mask);
                    
                    // Extract coordinates of matches
                    for (int j = 0; j < matches.size(); j++)
                    {
                        Point2f prevTemp, curTemp;
                        int queryIdx = matches[j].queryIdx;
                        int trainIdx = matches[j].trainIdx;
                        prevTemp = _storedKeypoints[i+1][queryIdx].pt;
                        curTemp = _storedKeypoints[num_images][trainIdx].pt;
        
                        prev.push_back(prevTemp);
                        cur.push_back(curTemp);
                    }
                    
                    // Rectify keypoints using sensor information
                    std::vector<Point2f> rectifiedPrev, rectifiedCur, curTransformed, curTransformed2;
                    perspectiveTransform(cur, curTransformed, HFromSensor);
                    
                    /*
                    for (int kk = 0; kk < cur.size(); kk++)
                        printf("[%.0f,%.0f] -> [%.0f,%.0f]       -> [%.0f,%.0f]        -> [%.0f,%.0f]  \n", prev[kk].x, prev[kk].y, cur[kk].x, cur[kk].y, curTransformed[kk].x, curTransformed[kk].y, 0.0, 0.0);
                    */
                    
                    float rejectionThr = 50.0; //fixme, this threshold is dependent on resolution
                    int inlierCount = 0;
                    for (int j = 0; j < matches.size(); j++)
                    {
                        if (norm ( prev[j] - curTransformed[j] ) < rejectionThr)
                            {
                            rectifiedPrev.push_back(prev[j]); // Only to be used for finding a better homography, not in pairwaise matches
                            rectifiedCur.push_back(cur[j]);
                            inliers_mask.push_back(uint(1));
                            inlierCount++;
                            }
                        else
                            inliers_mask.push_back(uint(0));
                    }
                    
                    Mat HFromMatches = Mat::eye(3, 3, CV_32F);
                    if ( rectifiedCur.size() > 10 )
                    {
                        HFromMatches = findHomography(rectifiedCur, rectifiedPrev, CV_RANSAC, 3, noArray(), 1000, 0.995) / 2;
                        HFromMatches.copyTo(tempMatchesInfo.H);
                    }
                    else
                        HFromSensor.copyTo(tempMatchesInfo.H);

                    // To match the direction OpenCV bundle adjustment does the job:
                    Mat tempH = tempMatchesInfo.H.inv();
                    tempH.copyTo(tempMatchesInfo.H);
                    
                    /*
                    // Test error values
                    float err1=0;
                    float err2 = 0;

                    perspectiveTransform(rectifiedCur, curTransformed, HFromSensor);
                    perspectiveTransform(rectifiedCur, curTransformed2, HFromMatches);

                    
                    err1 = norm(rectifiedPrev, curTransformed);
                    err2 = norm(rectifiedPrev, curTransformed2);
                    
                    for (int kk = 0; kk < cur.size(); kk++)
                        printf("[%.0f,%.0f] -> [%.0f,%.0f]       -> [%.0f,%.0f]        -> [%.0f,%.0f]  \n", prev[kk].x, prev[kk].y, cur[kk].x, cur[kk].y, curTransformed[kk].x, curTransformed[kk].y, curTransformed2[kk].x, curTransformed2[kk].y);
                    
                    std::cout << std::endl << err1/cur.size() << "," << err2/cur.size() << std::endl;
                
                    std::cout << HFromSensor << std::endl;
                    
                    std::cout << HFromMatches << std::endl;
                    */
                    tempMatchesInfo.matches = matches;
                    tempMatchesInfo.num_inliers = inlierCount;
                    tempMatchesInfo.inliers_mask = inliers_mask;
                    tempMatchesInfo.confidence = inlierCount/MAXMATCHES;////fixme
                }
                
                
                std::vector<cv::detail::MatchesInfo>::iterator it;
                
                it = _pairwiseMatches.begin();
                _pairwiseMatches.insert(it +  (i+1) * num_images - 1, tempMatchesInfo);
                
                //  After mathing i to j and pushing it, push j to i matches
                tempMatchesInfo.src_img_idx = num_images - 1;
                tempMatchesInfo.dst_img_idx = i;
                
                // push in the inverse match
                if (frameShouldBeProcessed)
                {
                    Mat temp = tempMatchesInfo.H.inv();
                    temp.copyTo(tempMatchesInfo.H);
                    
                    for (int k = 0; k < matches.size(); k++)
                    {
                        // Reverse matches
                        matches[k].queryIdx = tempMatchesInfo.matches[k].trainIdx;
                        matches[k].trainIdx = tempMatchesInfo.matches[k].queryIdx;
                    }
                    tempMatchesInfo.matches = matches;
                }
                _pairwiseMatches.push_back(tempMatchesInfo);
            }
            
            //self-match
            cv::detail::MatchesInfo tempMatchesInfo;
            tempMatchesInfo.src_img_idx = -1;
            tempMatchesInfo.dst_img_idx = -1;
            
            std::vector<cv::detail::MatchesInfo>::iterator it;
            it = _pairwiseMatches.begin();
            _pairwiseMatches.insert(it +  num_images * num_images - 1, tempMatchesInfo);
            
            // push features of the last frame
            int i = num_images - 1;
            ImageFeatures tempFeatures;
            tempFeatures.img_idx = i;
            tempFeatures.img_size = Size(_width, _height);
            tempFeatures.keypoints = _storedKeypoints[i+1];
            tempFeatures.descriptors = _storedDescriptors[i+1];
            _features.push_back(tempFeatures);
            
            _cameras.push_back(CameraParams());
        }
        

        
        copyRotationToCameraArray(currentRotationUsedToInitBA, _cameras); //fixme
        
        for (int i = 0; i < _cameras.size(); i++) {
            _cameras[i].R.convertTo(_cameras[i].R, CV_32FC1);
            _cameras[i].focal = _focal;
            _cameras[i].ppx = _width / 2;
            _cameras[i].ppy = _height / 2;
        }
        
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Augmenting matches: " << elapsed_secs << std::endl;
        
    }
    else
    { // Not a new frame, "refinement" button  pressed again

        clock_t begin = clock();
        std::vector<cv::detail::MatchesInfo> pairwiseMatches;
        matcher(_features, pairwiseMatches); // Most of randomness comes from here, if a good match is found, there won't be any change by repeating the refinement (tested if _cameras overwritten by motion sensor data)
        /*
        // Check incremental computations versus the OpenCV bundle adjustment
        std::cout << _pairwiseMatches[11].H << std::endl;
        std::cout << pairwiseMatches[11].H << std::endl;
        */
        /* // Filter out matches that we know from sensor data are false matches
        //std::cout << "--- Neighbors" << std::endl;
        for (int i = 0; i < num_images; i++)
        {

            for (int j = 0; j < num_images; j++)
            {
                if (i != j)
                {
                    cv::detail::MatchesInfo tempMatchesInfo;
                    bool frameShouldBeProcessed = false;
                    
                    tempMatchesInfo.src_img_idx = i;
                    tempMatchesInfo.dst_img_idx = j;
                    tempMatchesInfo.confidence = 0.9;////fixme
                    
                    // Check if frames i and j are neighbors
                    for (int k = 1; k < closestFrames[j + 1].size(); k++)
                        if ( (i + 1) == (closestFrames[j + 1][k]))
                        {
                                frameShouldBeProcessed = true;
                                //std::cout << i << "," << j << std::endl;
                        }
                    
                    for (int k = 1; k < closestFrames[i + 1].size(); k++)
                        if ( (j + 1) == (closestFrames[i + 1][k]))
                        {
                            frameShouldBeProcessed = true;
                            //std::cout << i << "," << j << std::endl;
                        }
                    
                    
                    if ( !frameShouldBeProcessed )
                    {
                        tempMatchesInfo.src_img_idx = i;
                        tempMatchesInfo.dst_img_idx = j;
                        pairwiseMatches[i * num_images + j] = tempMatchesInfo;
                        
                        tempMatchesInfo.src_img_idx = j;
                        tempMatchesInfo.dst_img_idx = i;
                        pairwiseMatches[j * num_images + i] = tempMatchesInfo;
                        
                    }
                }
            }
        }
        //std::cout << "---" << std::endl;
        */
        _pairwiseMatches = pairwiseMatches;
        
        matcher.collectGarbage();
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Re-match: "<< elapsed_secs << std::endl;
        /*
        //Basing the results on the sensor ratation data is much more reliable than using this estimation here to estimate rotation from scratch
        estimator(_features, _pairwiseMatches, _cameras); // This part introduces some randomness, redo this at each press of "refinement" button to be able to correct mistakes from previous stages
        
        for (int i = 0; i < _cameras.size(); i++) {
            _cameras[i].R.convertTo(_cameras[i].R, CV_32FC1);
        }
        */
        
        copyRotationToCameraArray(currentRotationUsedToInitBA, _cameras); //fixme
        
        for (int i = 0; i < _cameras.size(); i++) {
            _cameras[i].R.convertTo(_cameras[i].R, CV_32FC1);
            _cameras[i].focal = _focal;
            _cameras[i].ppx = _width / 2;
            _cameras[i].ppy = _height / 2;
        }
         
    }

    clock_t begin = clock();
    // estimator(_features, _pairwiseMatches, _cameras); // This part introduces some randomness, redo this at each press of "refinement" button to be able to correct mistakes from previous stages
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(_features, _pairwiseMatches, _cameras);
    _focal = _cameras[0].focal;
    copyBAResultToCamerArray(num_images, _cameras);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "BA main part: "<< elapsed_secs << std::endl;
    std::cout << "Estimated focal: "<< _focal << std::endl;
    
    return _refinedRotationArray;
}
///////////////////////////////////////////////////////////////////////////////////////////////

// ===============================================================================================================
// An interface to objective c.
// ===============================================================================================================
#include "cvEstimatorInterface.h"

void *cvEstimator_create()
{
    return (void*) new CvEstimator;
}


void cvEstimator_initialize(void *_cvEstimator, int width, int height)
{
    ((CvEstimator*)_cvEstimator)->initialize(width, height);
}



std::vector<float> cvEstimator_getRotation(void *_cvEstimator)
{
    return ((CvEstimator*)_cvEstimator)->getRotation();
}


float cvEstimator_getFocalLength(void *_cvEstimator)
{
    return ((CvEstimator*)_cvEstimator)->getFocalLength();
}




void cvEstimator_setRotationFromSensor(void *_cvEstimator, std::vector<float> &rot)
{
    ((CvEstimator*)_cvEstimator)->setRotationFromSensor(rot);
}


void cvEstimator_restart(void *_cvEstimator)
{
    ((CvEstimator*)_cvEstimator)->restart();
}


std::vector<std::vector <float>> cvEstimator_refinedStitching(void *_cvEstimator, std::vector<std::vector <float>> currentRotationUsedToInitBA, std::vector<std::vector<int>> closestFrames)
{
    // Convert from vector to Mat
    std::vector<Mat> currentRotation;
    currentRotation.push_back(Mat(3, 3, CV_32F));
    
    for (int i = 1; i < currentRotationUsedToInitBA.size(); i++)
    {
        currentRotation.push_back(Mat(3, 3, CV_32F));
        for (int j = 0; j < 9; j++) {
            currentRotation[i].at<float>(floor(j / 3), (j % 3)) = currentRotationUsedToInitBA[i][j];
        }
    }
    
    
    std::vector<cv::Mat> refinedRotationArray = ((CvEstimator*)_cvEstimator)->refinedStitching(currentRotation, closestFrames);
    
    // Convert from Mat to vector
    std::vector<std::vector <float>> refinedRotation(refinedRotationArray.size());
    for (int i = 1; i < refinedRotationArray.size(); i++)
    {
        for (int j = 0; j < 9; j++) {
            refinedRotation[i].push_back(refinedRotationArray[i].at<float>(floor(j / 3), (j % 3)));
        }
    }
    return refinedRotation;
}


void cvEstimator_saveNextFrame(void *_cvEstimator, Mat& nextFrame)
{
    ((CvEstimator*)_cvEstimator)->saveNextFrame(nextFrame);
}


void cvEstimator_procFrame(void *_cvEstimator, Mat& nextFrame, bool flag)
{
    ((CvEstimator*)_cvEstimator)->procFrame(nextFrame, flag);
}
