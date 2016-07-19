//
//  livePano.h
//  panoOGL
//
//  Created by gaoyuan on 4/28/14.
//  Copyright (c) 2014 gaoyuan. All rights reserved.
//

#ifndef __panoOGL__livePanoInterface__
#define __panoOGL__livePanoInterface__

#ifdef __cplusplus
extern "C" {
#endif
    
    void *pano_create(); // returns panoContext
    void pano_destroy(void *panoContext);
    void pano_initializeOpenGLTexture(void *panoContext, unsigned int imageTexture, unsigned int w, unsigned int h, unsigned int computeResReductionLevel);   // scaleForComputeResolution==2 means compute textures are in resolution w/4 x h/4
    void pano_clear(void *panoContext);
    void pano_step(void *panoContext);
    void pano_render(void *panoContext, float screenWidth, float screenHeight);
    void pano_getViewParams(void *panoContext, float viewLTRB[4], float
                               imageLTRBInClipPlane[4]);
    void pano_changeWarpMode(void *panoContext);////
    void pano_updateMotionData(void * panoContext, float roll, float pitch, float yaw, std::vector<float> fusedRotation);////
    void pano_setTexture(void *panoContext, unsigned int texture, unsigned int w, unsigned int h, bool nextReady, bool displayMode);
    void pano_restart(void *panoContext);
    void pano_setRotationFromImage(void *panoContext, std::vector<float> rotMat);
    void pano_setRefinedRotations(void *panoContext, std::vector<std::vector <float>> refinedRotations, float focal);
    void pano_setFocalLength(void *panoContext, float focal);
    std::vector<std::vector<float>>  pano_getCurrentRotations(void *panoContext);
    
    typedef struct {
        float brightness;

        float panoSigma;

        
        //ROI
        int xMin;
        int yMin;
        int xAdd;
        int yAdd;
        
    } PanoParams;
    
    void pano_getParams(void *panoContext,       PanoParams *sep);
    void pano_setParams(void *panoContext, const PanoParams *sep);

    void pano_setModeSelected(void *panoContext, int modeIdx);
    

#ifdef __cplusplus
}
#endif

#endif /* defined(__livePanoInterface__) */
