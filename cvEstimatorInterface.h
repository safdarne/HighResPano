//
//  CvEstimator.hpp
//  Opencv-directRotationComp
//
//  Created by safdarne on 6/2/16.
//  Copyright © 2016 Adobe. All rights reserved.
//

#ifndef CvEstimator_hpp
#define CvEstimator_hpp



#ifdef __cplusplus



extern "C" {
#endif
    
    void *cvEstimator_create();
    void cvEstimator_saveNextFrame(void *_cvEstimator, cv::Mat &nextFrame);
    void cvEstimator_procFrame(void *_cvEstimator, cv::Mat &nextFrame, bool flag);
    void cvEstimator_initialize(void *_cvEstimator, int width, int height);
    std::vector<float> cvEstimator_getRotation(void *_cvEstimator);
    void cvEstimator_setRotationFromSensor(void *_cvEstimator, std::vector<float> &rot);
    void cvEstimator_restart(void *_cvEstimator);
    std::vector<std::vector <float>>  cvEstimator_refinedStitching(void *_cvEstimator, std::vector<std::vector <float>> currentRotationUsedToInitBA, std::vector<std::vector<int>> _closestFrames);
    float cvEstimator_getFocalLength(void *_cvEstimator);


    
#ifdef __cplusplus
}
#endif





#endif /* CvEstimator_hpp */
