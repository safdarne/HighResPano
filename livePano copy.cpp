//
//  livePano.cpp
//  sharpenOGL
//
//  Created by gaoyuan on 4/28/14.
//  Copyright (c) 2014 gaoyuan. All rights reserved.
//

#include <iostream>

#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>


#include <fstream>
#include <sstream>
#include <string>

#include <math.h>
#include <map>
#include <vector>
#include <queue>

#import <CoreMedia/CMBufferQueue.h>

using namespace std;

// ==============================================================================
static float l = -1.f,   r =  1.0f;
static float b = -1.f,   t =  1.f;
static float n =  0.1f,  f =  100.f;

float gComputeMVP[16] = {
    2.0f/(r-l),    0.0f,          0.0f,         0.0f,
    0.0f,          2.0f/(t-b),    0.0f,         0.0f,
    0.0f,          0.0f,         -2.0f/(f-n),   0.0f,
    -(r+l)/(r-l),  -(t+b)/(t-b),  -(f+n)/(f-n),  1.0f
};

static const GLfloat gComputeQuadVertexData[] =
{
    -1.0f, -1.0f, -10.f,       0.0f, 0.0f,
    1.0f, -1.0f, -10.f,       1.0f, 0.0f,
    -1.0f,  1.0f, -10.f,       0.0f, 1.0f,
    -1.0f,  1.0f, -10.f,       0.0f, 1.0f,
    1.0f, -1.0f, -10.f,       1.0f, 0.0f,
    1.0f,  1.0f, -10.f,       1.0f, 1.0f
};

const int SharpenVertexAttribPosition  = 0;
const int SharpenVertexAttribTexCoord0 = 3;
static const GLbitfield _glClearBits = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
enum  _modeSelected {Blur,Denoise,Sharpen,Brightness,Contrast,Direc_denoise};

#define PI 3.1415926
//for new denoising algorithm
struct Params
{
    float sigma;
    int _hps;
    int _hss;
    int _ps;
    int _nStep;
    float _fStrength;
};

#define TABMAX 30
#define TABPRECISION 1000


class ProgramUniforms
{
public:
    GLint              program;
    map<string, GLint> uniformMap;
    
    ProgramUniforms() : program(0) {}
    ~ProgramUniforms() {
        if (program) {
            glDeleteProgram(program);
            program = 0;
        }
    }
};

class TextureBuffer {
public:
    GLuint texture;
    GLuint frameBuffer;
    GLenum internalformat;
    GLenum format;
    GLenum type;
    int    w,h;
    
    TextureBuffer() : texture(0), frameBuffer(0) {}
    void release() {
        if(texture)
        {
            glDeleteTextures(1, &texture);
            texture = 0;
        }
        if(frameBuffer)
        {
            glDeleteFramebuffers(1, &frameBuffer);
            frameBuffer = 0;
        }
        
    }
};


class livePano
{
public:
    livePano();
    ~livePano();
    
public:
    
    
    bool    _imagePanZoomUpdate;
    
    
    float   _xMin;
    float   _yMin;
    float   _xAdd;
    float   _yAdd;
    float   _imageZoomScale;
    int _texturesCount;
    
    float _roll;
    float _yaw;
    float _pitch;
    float _fov;
    
    float _pitchArray[32];
    float _rollArray[32];
    float _yawArray[32];
    
    
    //for new denoising algorithm
    Params param;
    static const int tabLength = TABMAX*TABPRECISION;
    float fpTab[tabLength];
private:
    float  _screenWidth, _screenHeight;
public:
    float   _imageRectInClipPlane[4];
    float   _viewRect[4];
protected:
    GLfloat _drawQuadVertexData[sizeof(gComputeQuadVertexData)];
    GLfloat _drawMVP[16];
    
    GLuint _drawVertexArray;
    GLuint _drawVertexBuffer;
    
    GLuint _computeVertexArray;
    GLuint _computeVertexBuffer;
    
    
    
    //// Mori
    GLuint _drawSphereVertexArray;
    GLuint _panoVertexBuffer;
    GLuint _panoSphereVertexBuffer;
    
    
    
protected:
    // source image: original
    GLuint        _imageTextureLive; // This will always have the latest image, which is moving on the fly and not stitched yet
    GLuint        _imageTextureArray[32];
    int           _imageWidthOriginal, _imageHeightOriginal;
    
    // source image: reduced
    TextureBuffer _imageTextureReduced;
    
    
    // input image texture: original or reduced
    GLuint        _imageTexture;
    int           _imageWidth, _imageHeight;
public:
    float         _imageReductionScale;
    
public:
    
    // the following 3 buffers are set to implement the sharpen algorithm
    TextureBuffer _inputBuffer;
    TextureBuffer _outputBuffer;

    
    TextureBuffer _overlayBuffer;
    int           _currentTargetBuffer;
    
    // Mori
    GLfloat *m_VertexData;
    GLint _m_Stacks;
    GLint _m_Slices;
    bool _warpMode;
    
protected:
    GLint  _defaultFrameBuffer;
    GLint _defaultTexture;
    GLint  _overlayFrameBuffer;
    GLint  _defaultViewport[4];
    bool _initialized;
    
    map<string, ProgramUniforms>  _programs;
    
public:
    void initializeOpenGL(GLuint imageTexture, int w, int h, unsigned int computeResReductionLevel);
    void destroyOpenGL();
    void updateView(float screenWidth, float screenHeight);
    void render(float screenWidth, float screenHeight);
    void step();
    void changeWarpMode();
    void updateMotionData(float roll, float pitch, float yaw);
    void setTexture(GLuint imageTexture, unsigned int w, unsigned int h, bool nextReady);////
    void initialAllTextures();
    
    
public:
    
    void switchBetween5Modes();
    
    void computeImageBrightness();
    void computeImageBrightness2();
    
    float kSigmaClipping(vector<float> & dataset, vector<bool> &judge);//for filtering out the uniform patch , used for noise estimation.
    void SetParams(const float sigma0);
    
    float computeSlope(const float x0, const float y0, const float x1, const float y1);
    bool isLocatedInOrNot(const float brushSlope, const float patchSlope, const float offset);
    
    void initSelection(GLuint maskTexture);
    void initialInputBuffer();
    void updateInputBuffer();
    GLuint outputFrameBuffer();

    void drawImageSelection();
    
    void findSphereVertices(GLint m_Stacks, GLint m_Slices, GLfloat m_Scale); ////
    void myDrawImage(bool warpMode); ////
    void copyBuffer2();
    void copyBufferToDisplay();
    
    void prepareProjectionMatrix();
    
    void generateRenderToTexture(GLint internalformat, GLenum format, GLenum type, TextureBuffer &tb, int w, int h, bool linearInterp);
    
protected:
    void updateTexturesAndFrameBuffers();
    void createTexturesAndFrameBuffers();
    GLint compileShaders(const GLchar vShaderText[], const GLchar fShaderText[]);
    
    
    void loadShaders_sphericalWarp();
    void loadShaders_sphericalWarpTemp();
    void loadShaders_copyBuffer();
    void loadShaders_initialBuffer();
    void loadShaders_myDrawImage();
    void loadShaders_sphericalWarpMultiple();
    
    float gauss(float x, float sigma);
};

#define png_infopp_NULL (png_infopp)NULL

livePano::livePano() :


_screenWidth(-1),
_screenHeight(-1),
_currentTargetBuffer(0),
_imageTexture(0),
_drawVertexArray(0),
_drawSphereVertexArray(0),
_drawVertexBuffer(0),
_computeVertexArray(0),
_computeVertexBuffer(0),
_defaultFrameBuffer(-1),
_defaultTexture(-1),
_overlayFrameBuffer(-1),
_xMin(0.),
_yMin(0.),
_xAdd(200.),
_yAdd(200.),

_m_Stacks(64),
_m_Slices(64),
_warpMode(false),
_fov(.56),
_initialized(false),
_texturesCount(1)  // Start from 1. The index 0 will always store the current texture from camera feed
{
}

livePano::~livePano()
{
    destroyOpenGL();
}

///////////////////////////////////////////////////////////////////////////////
// CORE



//// Mori

void livePano::findSphereVertices(GLint m_Stacks, GLint m_Slices, GLfloat m_Scale) {
    
    GLfloat *vPtr = m_VertexData = (GLfloat*)malloc(sizeof(GLfloat) * 5 * ((m_Slices*2+2) * (m_Stacks)));
    // Normals
    GLfloat *nPtr = (GLfloat*)malloc(sizeof(GLfloat) * 5 * ((m_Slices*2+2) * (m_Stacks)));
    GLfloat *tPtr = NULL;
    tPtr = (GLfloat*)malloc(sizeof(GLfloat) * 2 * ((m_Slices*2+2) * (m_Stacks)));
    unsigned int phiIdx, thetaIdx;
    
    
    
    // Latitude
    for(phiIdx = 0; phiIdx < m_Stacks; phiIdx++){
        //starts at -pi/2 goes to pi/2
        //the first circle
        float phi0 = M_PI * ((float)(phiIdx+0) * (1.0/(float)(m_Stacks)) - 0.5);
        //second one
        float phi1 = M_PI * ((float)(phiIdx+1) * (1.0/(float)(m_Stacks)) - 0.5);
        float cosPhi0 = cos(phi0);
        float sinPhi0 = sin(phi0);
        float cosPhi1 = cos(phi1);
        float sinPhi1 = sin(phi1);
        float cosTheta, sinTheta;
        
        
        //longitude
        for(thetaIdx = 0; thetaIdx < m_Slices; thetaIdx++){
            float theta = -2.0*M_PI * ((float)thetaIdx) * (1.0/(float)(m_Slices - 1));
            cosTheta = cos(theta+M_PI*.5);
            sinTheta = sin(theta+M_PI*.5);
            //get x-y-x of the first vertex of stack
            vPtr[0] = m_Scale*cosPhi0 * cosTheta;
            vPtr[1] = m_Scale*sinPhi0;
            vPtr[2] = m_Scale*(cosPhi0 * sinTheta);
            //// Mori
            vPtr[3] = (float)thetaIdx/(float)(m_Slices);
            vPtr[4] = (float)phiIdx/(float)(m_Stacks);
            
            //the same but for the vertex immediately above the previous one.
            vPtr[3+2] = m_Scale*cosPhi1 * cosTheta;
            vPtr[4+2] = m_Scale*sinPhi1;
            vPtr[5+2] = m_Scale*(cosPhi1 * sinTheta);
            //// Mori
            vPtr[8] = (float)(thetaIdx)/(float)(m_Slices);
            vPtr[9] = (float)(phiIdx+1)/(float)(m_Stacks);
            
            
            nPtr[0] = cosPhi0 * cosTheta;
            nPtr[1] = sinPhi0;
            nPtr[2] = cosPhi0 * sinTheta;
            nPtr[3] = cosPhi1 * cosTheta;
            nPtr[4] = sinPhi1;
            nPtr[5] = cosPhi1 * sinTheta;
            if(tPtr!=NULL){
                GLfloat texX = (float)thetaIdx * (1.0f/(float)(m_Slices-1));
                tPtr[0] = 1.0-texX;
                tPtr[1] = (float)(phiIdx + 0) * (1.0f/(float)(m_Stacks));
                tPtr[2] = 1.0-texX;
                tPtr[3] = (float)(phiIdx + 1) * (1.0f/(float)(m_Stacks));
            }
            vPtr += 2*5;////2*3;
            nPtr += 2*3;
            if(tPtr != NULL) tPtr += 2*2;
        }
        
        //Degenerate triangle to connect stacks and maintain winding order
        vPtr[0] = vPtr[5] = vPtr[-5];
        vPtr[1] = vPtr[6] = vPtr[-4];
        vPtr[2] = vPtr[7] = vPtr[-3];
        
        nPtr[0] = nPtr[3] = nPtr[-3];
        nPtr[1] = nPtr[4] = nPtr[-2];
        nPtr[2] = nPtr[5] = nPtr[-1];
        if(tPtr != NULL){
            tPtr[0] = tPtr[2] = tPtr[-2];
            tPtr[1] = tPtr[3] = tPtr[-1];
        }
    }
}


void livePano::step()
{
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &_defaultFrameBuffer);
    glGetIntegerv(GL_VIEWPORT, _defaultViewport);
    
    glDisable(GL_DEPTH_TEST);
    glViewport(_defaultViewport[0],_defaultViewport[1],_defaultViewport[2] / 2,_defaultViewport[3] / 2);
}

void livePano::render(float screenWidth, float screenHeight)
{
    
    // update view if necessary
    if(_imagePanZoomUpdate || _screenWidth!=screenWidth || _screenHeight!=screenHeight)
        updateView(screenWidth, screenHeight);
    
    //glEnable(GL_BLEND);
    
    computeImageBrightness();
    //copyBuffer2();
    //computeImageBrightness2();
    //copyBufferToDisplay();
    
    //myDrawImage(_warpMode);
    
    //glBlendFuncSeparate(_outputBuffer.frameBuffer, _defaultFrameBuffer, 0.5, 0.5);
    //glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
}

void livePano::changeWarpMode()
{
    _warpMode = !_warpMode;
}

void livePano::updateMotionData(float roll, float pitch, float yaw)
{
    _roll = roll;
    _pitch = pitch;
    _yaw = yaw;
    //printf("%.2f, %.2f,%.2f \n", _roll, _pitch, _yaw);
}

void livePano::setTexture(GLuint imageTexture, unsigned int w, unsigned int h, bool nextReady)
{
    
    _imageTextureArray[0] = imageTexture; // This will always have the latest image, which is moving on the fly and not stitched yet
    _imageWidth  = _imageWidthOriginal  = w;
    _imageHeight = _imageHeightOriginal = h;

    bool linearInterp = true;
    
    _pitchArray[0] = _pitch;
    _rollArray[0] = _roll;
    _yawArray[0] = _yaw;
    
    
    if (nextReady)
    {
        // Next texture to be stitched is ready
        
        glGenTextures(1, &_imageTextureArray[_texturesCount]);
        glBindTexture(GL_TEXTURE_2D, _imageTextureArray[_texturesCount]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        
        
        GLuint temp;
        glGenFramebuffers(1, &temp);
        glBindFramebuffer(GL_FRAMEBUFFER, temp);
        glClear(_glClearBits);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _imageTextureArray[_texturesCount], 0);
        
        
        glBindFramebuffer(GL_FRAMEBUFFER, temp);
        glClear(_glClearBits);
        glViewport(0,0, _imageWidth,_imageHeight);
        ProgramUniforms &pu = _programs["copyBuffer"];
        glUseProgram(pu.program);
        
        glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);
        glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, imageTexture);
        glUniform1i(pu.uniformMap["imageTexture"], 0);
        
        glBindVertexArrayOES(_drawVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3*2); 
        
        //glDeleteTextures(1, &temp);
        //temp = 0;
        
        //glReadBufferNV(GL_BACK);
        //glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, w, h, 0);
        //_imageTextureArray[_texturesCount] = imageTexture;
        
        _pitchArray[_texturesCount] = _pitch;
        _rollArray[_texturesCount] = _roll;
        _yawArray[_texturesCount] = _yaw;
        _texturesCount++;
    }
    
    initialInputBuffer();
}
////////////////////////////////////////////////////////////////////////////
#define VERTEX_SHADER_SOURCE(S) \
"//#extension GL_OES_standard_derivatives : enable\n" \
#S

#define FRAGMENT_SHADER_SOURCE(S) \
"#extension GL_OES_standard_derivatives : enable\n" \
"uniform highp vec2 imageWH;\n" \
#S


const GLchar gVertexShaderText[] = VERTEX_SHADER_SOURCE
(
 attribute vec4 position;
 attribute vec2 uv0vert;
 
 varying highp vec2 uv0;   // output
 uniform mat4  mvpMatrix;
 uniform highp float height;
 
 void main()
 {
     uv0  = uv0vert;
     gl_Position = mvpMatrix * position;
 }
);


void livePano::loadShaders_sphericalWarp()
{
    const GLchar fShaderText[] = FRAGMENT_SHADER_SOURCE
    (
     uniform sampler2D imageTexture0;
     uniform sampler2D imageTexture1;
     uniform sampler2D imageTexture2;
     uniform sampler2D imageTexture3;
     uniform sampler2D imageTexture4;
     varying highp vec2 uv0;
     uniform highp vec4  roll;
     uniform highp vec4  pitch;
     uniform highp vec4  yaw;
     uniform highp float fov;
     
     uniform highp vec4 a11;
     uniform highp vec4 a12;
     uniform highp vec4 a13;
     uniform highp vec4 a21;
     uniform highp vec4 a22;
     uniform highp vec4 a23;
     uniform int number;
     
     bool isOnLeft(highp float x1, highp float x2, highp float y1, highp float y2, highp float xx, highp float yy){
         if (abs(x2 - x1) > 10e-3)
         {
             highp float m = (y2 - y1) / (x2 - x1);
             
             if ((yy - (m * (xx - x1) + y1) > 0.))
                 if (x1 < x2)
                     return true;
                 else
                     return false;
                 else
                     if (x1 < x2)
                         return false;
                     else
                         return true;
         }
         else
             if (xx < x1)
                 if (y1 < y2)
                     return true;
                 else
                     return false;
                 else
                     if (y1 < y2)
                         return false;
                     else
                         return true;
     }
     
     
     void main()
     {
         highp vec2 pos;
         highp vec4 newCol;
         highp vec4 col;
         
         highp float f = .560;
         highp float fovX = .480;
         highp float fovY = .640;//fovX * 640./480.;
         highp float s = 1.;
         
         highp float pitch2 ;
         highp float roll2;
         highp float yaw2 ;
         highp float cosYaw;
         highp float sinYaw;
         
         highp vec4 newColAverage = vec4(0., 0., 0., 0.);
         
         highp float theta = (uv0.x - 0.5) * 1. * 3.1415;
         highp float phi = (uv0.y - .5) * 3.1415;
         /*
          highp float pxx = (cos(phi) * sin(theta));
          highp float pyy = (sin(phi));
          highp float pzz = (cos(phi) * cos(theta));
          */
         
         int count = 0;
         
         for (int jj = 0; jj < number; jj++) {
             
             highp float a11_2;
             highp float a12_2;
             highp float a13_2;
             highp float a21_2;
             highp float a22_2;
             highp float a23_2;
             
             if (jj == 0) {
                 pitch2 = pitch.x;
                 roll2 = roll.x;
                 yaw2 = yaw.x;
                 a11_2 = a11.x;
                 a12_2 = a12.x;
                 a13_2 = a13.x;
                 a21_2 = a21.x;
                 a22_2 = a22.x;
                 a23_2 = a23.x;
             }
             if (jj == 1) {
                 pitch2 = pitch.y;
                 roll2 = roll.y;
                 yaw2 = yaw.y;
                 a11_2 = a11.y;
                 a12_2 = a12.y;
                 a13_2 = a13.y;
                 a21_2 = a21.y;
                 a22_2 = a22.y;
                 a23_2 = a23.y;
             }
             if (jj == 2) {
                 pitch2 = pitch.z;
                 roll2 = roll.z;
                 yaw2 = yaw.z;
                 a11_2 = a11.z;
                 a12_2 = a12.z;
                 a13_2 = a13.z;
                 a21_2 = a21.z;
                 a22_2 = a22.z;
                 a23_2 = a23.z;
             }
             if (jj == 3) {
                 pitch2 = pitch.w;
                 roll2 = roll.w;
                 yaw2 = yaw.w;
                 a11_2 = a11.w;
                 a12_2 = a12.w;
                 a13_2 = a13.w;
                 a21_2 = a21.w;
                 a22_2 = a22.w;
                 a23_2 = a23.w;
             }
             
             cosYaw = 1.;
             sinYaw = 0.;
             
             //cosYaw = cos(yaw2);
             //sinYaw = sin(-yaw2);
             
             
             highp float xp = theta + roll2; //////
             highp float yp = phi - pitch2; /////
             
             
             // Cylinerical
             highp float pxx = 1. * tan(xp);
             //highp float pyy = tan(yp) * f * sqrt(1. + tan(xp) * tan(xp));//sqrt(pxx*pxx + 1. * 1.)*tan(phi - pitch2);
             highp float pyy = (yp) * 1. * sqrt(1. + tan(xp) * tan(xp));//sqrt(pxx*pxx + 1. * 1.)*tan(phi - pitch2);
             highp float pzz = 1.;
             
             /*
             // Spherical
             highp float pxx = f * tan(xp / s);
             highp float pyy = tan(yp / s) * f * sqrt(1. + tan(xp / s) * tan(xp / s));//sqrt(pxx*pxx + 1. * 1.)*tan(phi - pitch2);
             highp float pzz = 0.;
             */
             
             // Polar (longitude - latitude)
             //highp float pxx = f * tan(xp);
             //highp float pyy = tan(yp) * f * sqrt(1. + tan(xp) * tan(xp));//sqrt(pxx*pxx + 1. * 1.)*tan(phi - pitch2);
             /*
             highp float pxx = s* cos(phi) * sin(theta);
             highp float pyy = s* sin(phi) ;
             highp float pzz = s*cos(phi) * cos(theta);*/
             /*
             highp float pxx = s* cos(phi) * sin(theta);
             highp float pyy = s* sin(phi) ;
             highp float pzz = s*cos(phi) * cos(theta);
             */
             bool flag = false;

             highp float xCorners[4];
             highp float yCorners[4];
             /*
             highp float vrpxx = f * (a11_2 * pxx      +       a12_2 * pyy     +       a13_2 * pzz) + 0.;
             highp float vrpyy = f * (a21_2 * pxx      +       a22_2 * pyy     +       a23_2 * pzz) + 0.;
             */
             highp float vrpxx = f * pxx;
             highp float vrpyy = f * pyy;
             
             /*
              highp float xxRotated = cosYaw * (pxx - roll2) + sinYaw * (pyy - pitch2);
              highp float yyRotated = -sinYaw * (pxx - roll2) + cosYaw * (pyy - pitch2);
              */
             highp float xxRotated = cosYaw * (pxx) - sinYaw * (pyy);
             highp float yyRotated = sinYaw * (pxx ) + cosYaw * (pyy);
             
             newCol = vec4(0., 1., 0., 0.);
             
             
             if ((xp < 3.1415 / 2.) && (xp > -3.1415 / 2.) )
             {
                 
                 
                 /*
                  xCorners[0] = cosYaw * (-fovX / 2.) - sinYaw * (-fovY / 2.) + roll2;
                  xCorners[1] = cosYaw * (fovX / 2.) - sinYaw * (-fovY / 2.) + roll2;
                  xCorners[2] = cosYaw * (fovX / 2.) - sinYaw * (fovY / 2.) + roll2;
                  xCorners[3] = cosYaw * (-fovX / 2.) - sinYaw * (fovY / 2.) + roll2;
                  
                  yCorners[0] = sinYaw * (-fovX / 2.) + cosYaw * (-fovY / 2.) + pitch2;
                  yCorners[1] = sinYaw * (fovX / 2.) + cosYaw * (-fovY / 2.) + pitch2;
                  yCorners[2] = sinYaw * (fovX / 2.) + cosYaw * (fovY / 2.) + pitch2;
                  yCorners[3] = sinYaw * (-fovX / 2.) + cosYaw * (fovY / 2.) + pitch2;
                  
                  if (isOnLeft(xCorners[0], xCorners[1], yCorners[0], yCorners[1], vrpxx, vrpyy)) {
                  if (isOnLeft(xCorners[1], xCorners[2], yCorners[1], yCorners[2], vrpxx, vrpyy)) {
                  if (isOnLeft(xCorners[2], xCorners[3], yCorners[2], yCorners[3], vrpxx, vrpyy)) {
                  if (isOnLeft(xCorners[3], xCorners[0], yCorners[3], yCorners[0], vrpxx, vrpyy)) {
                  flag = true;
                  }
                  }
                  }
                  }
                  */
                 
                 
                 
                 xCorners[0] = -fovX / 2.;//cosYaw * (-fovX / 2.) - sinYaw * (-fovY / 2.);// + roll2;
                 xCorners[1] = fovX / 2.;//cosYaw * (fovX / 2.) - sinYaw * (-fovY / 2.);// + roll2;
                 xCorners[2] = fovX / 2.;//cosYaw * (fovX / 2.) - sinYaw * (fovY / 2.);// + roll2;
                 xCorners[3] = -fovX / 2.;//cosYaw * (-fovX / 2.) - sinYaw * (fovY / 2.);// + roll2;
                 
                 yCorners[0] = -fovY / 2.;//sinYaw * (-fovX / 2.) + cosYaw * (-fovY / 2.);// + pitch2;
                 yCorners[1] = -fovY / 2.;//sinYaw * (fovX / 2.) + cosYaw * (-fovY / 2.);// + pitch2;
                 yCorners[2] = fovY / 2.;//sinYaw * (fovX / 2.) + cosYaw * (fovY / 2.);//+ pitch2;
                 yCorners[3] = fovY / 2.;//sinYaw * (-fovX / 2.) + cosYaw * (fovY / 2.);// + pitch2;
                 
                 if (isOnLeft(xCorners[0], xCorners[1], yCorners[0], yCorners[1], vrpxx, vrpyy)) {
                     if (isOnLeft(xCorners[1], xCorners[2], yCorners[1], yCorners[2], vrpxx, vrpyy)) {
                         if (isOnLeft(xCorners[2], xCorners[3], yCorners[2], yCorners[3], vrpxx, vrpyy)) {
                             if (isOnLeft(xCorners[3], xCorners[0], yCorners[3], yCorners[0], vrpxx, vrpyy)) {
                                 flag = true;
                             }
                         }
                     }
                 }
             }
             
             if (flag)
             {
                 pos = vec2(vrpxx / fovX + 0.5, vrpyy / fovY + 0.5);
                 //pos = vec2(vrpxx / 1. + 0.5, vrpyy / 1. + 0.5);
                 //pos = vec2(xxRotated / fovX + 0.5, yyRotated / fovY + 0.5);
                 //col = texture2D(imageTexture, pos);//// Mori
                 
                 if (jj == 0) {
                     col = texture2D(imageTexture0, pos);
                 } else if (jj == 1) {
                     col = texture2D(imageTexture1, pos);
                 } else if (jj == 2) {
                     col = texture2D(imageTexture2, pos);
                 } else /* if (ndx == 3) */ {
                     col = texture2D(imageTexture3, pos);
                 }
                 
                 newCol.x = (col.x > 1.)? 1.:col.x;
                 newCol.y = (col.y > 1.)? 1.:col.y;
                 newCol.z = (col.z > 1.)? 1.:col.z;
                 newCol.w = 1.;
                 newColAverage += newCol;
                 count++;
                 
             }
         }
         if (count > 0) // at lease one texture existed here
             gl_FragColor = vec4(newColAverage.x / float(count), newColAverage.y / float(count), newColAverage.z / float(count), 1.);  //fixme -> for more than 4 images, the average does not work well
         else
             gl_FragColor = texture2D(imageTexture4, vec2(uv0.x, uv0.y));
     }
     );
    
    // Store the progrm, compute uniform locations
    ProgramUniforms &pu = (_programs["sphericalWarp"] = ProgramUniforms());
    pu.program = compileShaders(gVertexShaderText, fShaderText);
    
    
    pu.uniformMap["imageTexture0"] = glGetUniformLocation(pu.program, "imageTexture0");
    pu.uniformMap["imageTexture1"] = glGetUniformLocation(pu.program, "imageTexture1");
    pu.uniformMap["imageTexture2"] = glGetUniformLocation(pu.program, "imageTexture2");
    pu.uniformMap["imageTexture3"] = glGetUniformLocation(pu.program, "imageTexture3");
    pu.uniformMap["imageTexture4"] = glGetUniformLocation(pu.program, "imageTexture4");
    
    pu.uniformMap["mvpMatrix"]    = glGetUniformLocation(pu.program, "mvpMatrix");   //////!!!!! needs to be here for the vertex shader to map to uv0
    pu.uniformMap["imageWH"]   = glGetUniformLocation(pu.program, "imageWH");
    
    pu.uniformMap["heigth"]      = glGetUniformLocation(pu.program, "heigth"); //// Mori
    pu.uniformMap["roll"]      = glGetUniformLocation(pu.program, "roll"); //// Mori
    pu.uniformMap["pitch"]      = glGetUniformLocation(pu.program, "pitch"); //// Mori
    pu.uniformMap["yaw"]      = glGetUniformLocation(pu.program, "yaw"); //// Mori
    pu.uniformMap["fov"]      = glGetUniformLocation(pu.program, "fov"); //// Mori
    pu.uniformMap["a11"]      = glGetUniformLocation(pu.program, "a11"); //// Mori
    pu.uniformMap["a12"]      = glGetUniformLocation(pu.program, "a12"); //// Mori
    pu.uniformMap["a13"]      = glGetUniformLocation(pu.program, "a13"); //// Mori
    pu.uniformMap["a21"]      = glGetUniformLocation(pu.program, "a21"); //// Mori
    pu.uniformMap["a22"]      = glGetUniformLocation(pu.program, "a22"); //// Mori
    pu.uniformMap["a23"]      = glGetUniformLocation(pu.program, "a33"); //// Mori
    pu.uniformMap["prevTexture"]      = glGetUniformLocation(pu.program, "prevTexture"); //// Mori
    pu.uniformMap["number"]      = glGetUniformLocation(pu.program, "number"); //// Mori
}


void livePano::loadShaders_sphericalWarpTemp()
{
    const GLchar fShaderText[] = FRAGMENT_SHADER_SOURCE
    (
     uniform sampler2D imageTexture0;
     uniform sampler2D imageTexture1;
     uniform sampler2D imageTexture2;
     uniform sampler2D imageTexture3;
     uniform sampler2D imageTexture4;
     varying highp vec2 uv0;
     uniform highp vec4  roll;
     uniform highp vec4  pitch;
     uniform highp vec4  yaw;
     uniform highp float fov;
     
     uniform highp vec4 a11;
     uniform highp vec4 a12;
     uniform highp vec4 a13;
     uniform highp vec4 a21;
     uniform highp vec4 a22;
     uniform highp vec4 a23;
     uniform int number;
     
     bool isOnLeft(highp float x1, highp float x2, highp float y1, highp float y2, highp float xx, highp float yy){
         if (abs(x2 - x1) > 10e-3)
         {
             highp float m = (y2 - y1) / (x2 - x1);
             
             if ((yy - (m * (xx - x1) + y1) > 0.))
                 if (x1 < x2)
                     return true;
                 else
                     return false;
                 else
                     if (x1 < x2)
                         return false;
                     else
                         return true;
         }
         else
             if (xx < x1)
                 if (y1 < y2)
                     return true;
                 else
                     return false;
                 else
                     if (y1 < y2)
                         return false;
                     else
                         return true;
     }
     
     
     void main()
     {
         highp vec2 pos;
         highp vec4 newCol;
         highp vec4 col;
         
         highp float f = .4;
         highp float fovX = 480;
         highp float fovY = 640;
         
         
         highp float pitch2 ;
         highp float roll2;
         highp float yaw2 ;
         highp float cosYaw;
         highp float sinYaw;
         
         highp vec4 newColAverage = vec4(0., 0., 0., 0.);
  
         highp float theta = (uv0.x - 0.5) * 1. * 3.1415;
         highp float phi = (uv0.y - 0.5) * 3.1415;
         /*
         highp float pxx = (cos(phi) * sin(theta));
         highp float pyy = (sin(phi));
         highp float pzz = (cos(phi) * cos(theta));
         */
         
         int count = 0;
         
         for (int jj = 0; jj < number; jj++) {
         
             highp float a11_2;
             highp float a12_2;
             highp float a13_2;
             highp float a21_2;
             highp float a22_2;
             highp float a23_2;
             
             if (jj == 0) {
                 pitch2 = pitch.x;
                 roll2 = roll.x;
                 yaw2 = yaw.x;
                 a11_2 = a11.x;
                 a12_2 = a12.x;
                 a13_2 = a13.x;
                 a21_2 = a21.x;
                 a22_2 = a22.x;
                 a23_2 = a23.x;
             }
             if (jj == 1) {
                 pitch2 = pitch.y;
                 roll2 = roll.y;
                 yaw2 = yaw.y;
                 a11_2 = a11.y;
                 a12_2 = a12.y;
                 a13_2 = a13.y;
                 a21_2 = a21.y;
                 a22_2 = a22.y;
                 a23_2 = a23.y;
             }
             if (jj == 2) {
                 pitch2 = pitch.z;
                 roll2 = roll.z;
                 yaw2 = yaw.z;
                 a11_2 = a11.z;
                 a12_2 = a12.z;
                 a13_2 = a13.z;
                 a21_2 = a21.z;
                 a22_2 = a22.z;
                 a23_2 = a23.z;
             }
             if (jj == 3) {
                 pitch2 = pitch.w;
                 roll2 = roll.w;
                 yaw2 = yaw.w;
                 a11_2 = a11.w;
                 a12_2 = a12.w;
                 a13_2 = a13.w;
                 a21_2 = a21.w;
                 a22_2 = a22.w;
                 a23_2 = a23.w;
             }

             cosYaw = 1.;
             sinYaw = 0.;
             
             //cosYaw = cos(yaw2);
             //sinYaw = sin(-yaw2);
             
            
             /*
             highp float pxx = cos(phi) * sin(theta);
             highp float pyy = sin(phi) ;
             highp float pzz = cos(phi) * cos(theta);
             */
             
             
             highp float xp = theta - roll2;
             highp float yp = phi - pitch2;
             
             highp float pxx = f * tan(xp);
             highp float pyy = tan(yp) * f * sqrt(1. + tan(xp) * tan(xp));//sqrt(pxx*pxx + 1. * 1.)*tan(phi - pitch2);
             highp float pzz = 1.;
             
             
             bool flag = false;
             
             highp float xCorners[4];
             highp float yCorners[4];
             
             /*
             highp float vrpxx = f * ( cos(yaw2 ) * pxx     +       -sin(yaw2 ) * pyy         +        0. * pzz) + 0.;
             highp float vrpyy = f * ( sin(yaw2) * pxx     +       cos(yaw2) * pyy         +        0. * pzz) + 0.;
             highp float vrpzz = f * ( 0.     +       0.         +        1. * pzz) + 0.;
             

             vrpxx =  ( cos(roll2 ) * vrpxx     +       0. * vrpyy         +        -sin(roll2 )  * vrpzz) + 0.;
             vrpyy =  ( 0. * vrpxx    +       1.  * vrpyy       +        0. * vrpzz) + 0.;
             vrpzz =  ( sin(roll2) * vrpxx     +       0. * vrpyy         +        cos(roll2) * vrpzz) + 0.;
             
             vrpxx =  ( 1. * vrpxx     +       0.          +        0.) + 0.;
             vrpyy =  ( 0. * vrpxx    +       cos(pitch2)  * vrpyy       +        -sin(pitch2) * vrpzz) + 0.;
             vrpzz =  ( 0. * vrpxx     +       sin(pitch2) * vrpyy         +        cos(pitch2) * vrpzz) + 0.;
             */
             
             highp float vrpxx = f * (a11_2 * pxx      +       a12_2 * pyy     +       a13_2 * pzz) + 0.;
             highp float vrpyy = f * (a21_2 * pxx      +       a22_2 * pyy     +       a23_2 * pzz) + 0.;
             
             
             /*
              highp float xxRotated = cosYaw * (pxx - roll2) + sinYaw * (pyy - pitch2);
              highp float yyRotated = -sinYaw * (pxx - roll2) + cosYaw * (pyy - pitch2);
              */
             highp float xxRotated = cosYaw * (pxx) - sinYaw * (pyy);
             highp float yyRotated = sinYaw * (pxx ) + cosYaw * (pyy);
             
             newCol = vec4(0., 1., 0., 0.);
             
             
            if (true)//(vrpxx < 3.1415 / 2.) && (vrpxx > -3.1415 / 2.) )
            {
             

                 /*
                 xCorners[0] = cosYaw * (-fovX / 2.) - sinYaw * (-fovY / 2.) + roll2;
                 xCorners[1] = cosYaw * (fovX / 2.) - sinYaw * (-fovY / 2.) + roll2;
                 xCorners[2] = cosYaw * (fovX / 2.) - sinYaw * (fovY / 2.) + roll2;
                 xCorners[3] = cosYaw * (-fovX / 2.) - sinYaw * (fovY / 2.) + roll2;
                 
                 yCorners[0] = sinYaw * (-fovX / 2.) + cosYaw * (-fovY / 2.) + pitch2;
                 yCorners[1] = sinYaw * (fovX / 2.) + cosYaw * (-fovY / 2.) + pitch2;
                 yCorners[2] = sinYaw * (fovX / 2.) + cosYaw * (fovY / 2.) + pitch2;
                 yCorners[3] = sinYaw * (-fovX / 2.) + cosYaw * (fovY / 2.) + pitch2;
                 
                 if (isOnLeft(xCorners[0], xCorners[1], yCorners[0], yCorners[1], vrpxx, vrpyy)) {
                     if (isOnLeft(xCorners[1], xCorners[2], yCorners[1], yCorners[2], vrpxx, vrpyy)) {
                         if (isOnLeft(xCorners[2], xCorners[3], yCorners[2], yCorners[3], vrpxx, vrpyy)) {
                             if (isOnLeft(xCorners[3], xCorners[0], yCorners[3], yCorners[0], vrpxx, vrpyy)) {
                                 flag = true;
                             }
                         }
                     }
                 }
                 */
                  

                  
                xCorners[0] = -fovX / 2.;//cosYaw * (-fovX / 2.) - sinYaw * (-fovY / 2.);// + roll2;
                 xCorners[1] = fovX / 2.;//cosYaw * (fovX / 2.) - sinYaw * (-fovY / 2.);// + roll2;
                 xCorners[2] = fovX / 2.;//cosYaw * (fovX / 2.) - sinYaw * (fovY / 2.);// + roll2;
                 xCorners[3] = -fovX / 2.;//cosYaw * (-fovX / 2.) - sinYaw * (fovY / 2.);// + roll2;
                 
                 yCorners[0] = -fovY / 2.;//sinYaw * (-fovX / 2.) + cosYaw * (-fovY / 2.);// + pitch2;
                 yCorners[1] = -fovY / 2.;//sinYaw * (fovX / 2.) + cosYaw * (-fovY / 2.);// + pitch2;
                 yCorners[2] = fovY / 2.;//sinYaw * (fovX / 2.) + cosYaw * (fovY / 2.);//+ pitch2;
                 yCorners[3] = fovY / 2.;//sinYaw * (-fovX / 2.) + cosYaw * (fovY / 2.);// + pitch2;
                 
                 if (isOnLeft(xCorners[0], xCorners[1], yCorners[0], yCorners[1], vrpxx, vrpyy)) {
                     if (isOnLeft(xCorners[1], xCorners[2], yCorners[1], yCorners[2], vrpxx, vrpyy)) {
                         if (isOnLeft(xCorners[2], xCorners[3], yCorners[2], yCorners[3], vrpxx, vrpyy)) {
                             if (isOnLeft(xCorners[3], xCorners[0], yCorners[3], yCorners[0], vrpxx, vrpyy)) {
                                 flag = true;
                             }
                         }
                     }
                 }
            }
             
             if (flag)
             {
                 //pos = vec2(pxx / fovX + 0.5, pyy / fovY + 0.5);
                 pos = vec2(vrpxx / fovX + 0.5, vrpyy / fovY + 0.5);
                 //pos = vec2(xxRotated / fovX + 0.5, yyRotated / fovY + 0.5);
                 //col = texture2D(imageTexture, pos);//// Mori
                 
                 if (jj == 0) {
                     col = texture2D(imageTexture0, pos);
                 } else if (jj == 1) {
                     col = texture2D(imageTexture1, pos);
                 } else if (jj == 2) {
                     col = texture2D(imageTexture2, pos);
                 } else /* if (ndx == 3) */ {
                     col = texture2D(imageTexture3, pos);
                 }

                 newCol.x = (col.x > 1.)? 1.:col.x;
                 newCol.y = (col.y > 1.)? 1.:col.y;
                 newCol.z = (col.z > 1.)? 1.:col.z;
                 newCol.w = 1.;
                 newColAverage += newCol;
                 count++;
                 
             }
         }
         if (count > 0) // at lease one texture existed here
             gl_FragColor = vec4(newColAverage.x / float(count), newColAverage.y / float(count), newColAverage.z / float(count), 1.);  //fixme -> for more than 4 images, the average does not work well
         else
             gl_FragColor = texture2D(imageTexture4, vec2(uv0.x, uv0.y));
     }
     );
    
    // Store the progrm, compute uniform locations
    ProgramUniforms &pu = (_programs["sphericalWarp"] = ProgramUniforms());
    pu.program = compileShaders(gVertexShaderText, fShaderText);
    
    
    pu.uniformMap["imageTexture0"] = glGetUniformLocation(pu.program, "imageTexture0");
    pu.uniformMap["imageTexture1"] = glGetUniformLocation(pu.program, "imageTexture1");
    pu.uniformMap["imageTexture2"] = glGetUniformLocation(pu.program, "imageTexture2");
    pu.uniformMap["imageTexture3"] = glGetUniformLocation(pu.program, "imageTexture3");
    pu.uniformMap["imageTexture4"] = glGetUniformLocation(pu.program, "imageTexture4");
    
    pu.uniformMap["mvpMatrix"]    = glGetUniformLocation(pu.program, "mvpMatrix");   //////!!!!! needs to be here for the vertex shader to map to uv0
    pu.uniformMap["imageWH"]   = glGetUniformLocation(pu.program, "imageWH");
    
    pu.uniformMap["heigth"]      = glGetUniformLocation(pu.program, "heigth"); //// Mori
    pu.uniformMap["roll"]      = glGetUniformLocation(pu.program, "roll"); //// Mori
    pu.uniformMap["pitch"]      = glGetUniformLocation(pu.program, "pitch"); //// Mori
    pu.uniformMap["yaw"]      = glGetUniformLocation(pu.program, "yaw"); //// Mori
    pu.uniformMap["fov"]      = glGetUniformLocation(pu.program, "fov"); //// Mori
    pu.uniformMap["a11"]      = glGetUniformLocation(pu.program, "a11"); //// Mori
    pu.uniformMap["a12"]      = glGetUniformLocation(pu.program, "a12"); //// Mori
    pu.uniformMap["a13"]      = glGetUniformLocation(pu.program, "a13"); //// Mori
    pu.uniformMap["a21"]      = glGetUniformLocation(pu.program, "a21"); //// Mori
    pu.uniformMap["a22"]      = glGetUniformLocation(pu.program, "a22"); //// Mori
    pu.uniformMap["a23"]      = glGetUniformLocation(pu.program, "a33"); //// Mori
    pu.uniformMap["prevTexture"]      = glGetUniformLocation(pu.program, "prevTexture"); //// Mori
    pu.uniformMap["number"]      = glGetUniformLocation(pu.program, "number"); //// Mori
}



void livePano::loadShaders_myDrawImage()
{
    const GLchar fShaderText[] = FRAGMENT_SHADER_SOURCE
    (
     uniform sampler2D imageTexture;
     varying highp vec2 uv0;
     void main()
     {
         highp vec4 col = texture2D(imageTexture, uv0);
         gl_FragColor = col;
     }
     );
    
    // Store the progrm, compute uniform locations
    ProgramUniforms &pu = (_programs["myDrawImage"] = ProgramUniforms());
    pu.program = compileShaders(gVertexShaderText, fShaderText);
    
    pu.uniformMap["mvpMatrix"]    = glGetUniformLocation(pu.program, "mvpMatrix");
    pu.uniformMap["imageTexture"] = glGetUniformLocation(pu.program, "imageTexture");
    pu.uniformMap["imageWH"]      = glGetUniformLocation(pu.program, "imageWH");
}


void livePano::loadShaders_initialBuffer()
{
    const GLchar fShaderText[] = FRAGMENT_SHADER_SOURCE
    (
     varying highp vec2 uv0;
     void main()
     {
         highp vec4 col = vec4(0.2,0.2,0.,0.);
         gl_FragColor = col;
     }
     );
    
    // Store the progrm, compute uniform locations
    ProgramUniforms &pu = (_programs["initialBuffer"] = ProgramUniforms());
    pu.program = compileShaders(gVertexShaderText, fShaderText);
    
    pu.uniformMap["mvpMatrix"]    = glGetUniformLocation(pu.program, "mvpMatrix");
    pu.uniformMap["imageWH"]      = glGetUniformLocation(pu.program, "imageWH");
    
}


void livePano::loadShaders_copyBuffer()
{
    const GLchar fShaderText[] = FRAGMENT_SHADER_SOURCE
    (
     uniform sampler2D imageTexture;
     varying highp vec2 uv0;
     void main()
     {
         highp vec4 y = texture2D(imageTexture, vec2(uv0.x, uv0.y));
         gl_FragColor = y;
     }
     );
    
    // Store the progrm, compute uniform locations
    ProgramUniforms &pu = (_programs["copyBuffer"] = ProgramUniforms());
    pu.program = compileShaders(gVertexShaderText, fShaderText);
    pu.uniformMap["imageTexture"] = glGetUniformLocation(pu.program, "imageTexture");
}


void livePano::initialInputBuffer()
{
    // Set input buffer with frame from camera
    glBindFramebuffer(GL_FRAMEBUFFER, _inputBuffer.frameBuffer);
    glClear(_glClearBits);
    glViewport(0,0, _inputBuffer.w, _inputBuffer.h);
    
    ProgramUniforms &pu = _programs["copyBuffer"];
    glUseProgram(pu.program);
    
    glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, gComputeMVP);
    glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _imageTextureArray[0]);
    
    glUniform1i(pu.uniformMap["imageTexture"], 0);
    
    glBindVertexArrayOES(_computeVertexArray);
    glDrawArrays(GL_TRIANGLES, 0, 3*2);
    
    
    
}

void livePano::computeImageBrightness()
{
    //printf("%.2f, %.2f,%.2f \n", _roll * 180. / 3.14, _pitch * 180. / 3.14, _yaw * 180. / 3.14);
    /*
    #define A11(i) (cos(_yawArray[i]) * cos(_rollArray[i]))
    #define A12(i) (cos(_pitchArray[i]) * sin(_yawArray[i]) - sin(_pitchArray[i]) * sin(_rollArray[i]) * cos(_yawArray[i]))
    #define A13(i) (sin(_pitchArray[i]) * sin(_yawArray[i]) + cos(_pitchArray[i]) * cos(_yawArray[i]) * sin(_rollArray[i]))
    #define A21(i) (-cos(_rollArray[i]) * sin(_yawArray[i]))
    #define A22(i) (cos(_pitchArray[i]) * cos(_yawArray[i]) + sin(_pitchArray[i]) * sin(_yawArray[i]) * sin(_rollArray[i]))
    #define A23(i) (sin(_pitchArray[i]) * cos(_yawArray[i]) - cos(_pitchArray[i]) * sin(_yawArray[i]) * sin(_rollArray[i]))
    */
    
    /*
     #define A11(i) (cos(_yawArray[i]) * cos(_rollArray[i]))
     #define A12(i) (-sin(_pitchArray[i]) * cos(_yawArray[i]) - sin(_pitchArray[i]) * sin(_rollArray[i]) * cos(_yawArray[i]))
     #define A13(i) (sin(_pitchArray[i]) * sin(_yawArray[i]) + cos(_pitchArray[i]) * cos(_yawArray[i]) * sin(_rollArray[i]))
     #define A21(i) (-cos(_rollArray[i]) * sin(_yawArray[i]))
     #define A22(i) (cos(_pitchArray[i]) * cos(_yawArray[i]) + sin(_pitchArray[i]) * sin(_yawArray[i]) * sin(_rollArray[i]))
     #define A23(i) (sin(_pitchArray[i]) * cos(_yawArray[i]) - cos(_pitchArray[i]) * sin(_yawArray[i]) * sin(_rollArray[i]))
    */
    
#define A11(i) (cos(-_yawArray[i]) * cos(-_rollArray[i]))
#define A12(i) (-sin(-_yawArray[i]) * cos(-_pitchArray[i]) - cos(-_yawArray[i]) * sin(-_rollArray[i]) * sin(-_pitchArray[i]))
#define A13(i) (sin(-_yawArray[i]) * sin(-_pitchArray[i]) - cos(-_yawArray[i]) * sin(-_rollArray[i]) * cos(-_pitchArray[i]))
#define A21(i) (sin(-_yawArray[i]) * cos(-_rollArray[i]))
#define A22(i) (cos(-_yawArray[i]) * cos(-_pitchArray[i]) - sin(-_yawArray[i]) * sin(-_rollArray[i]) * sin(-_pitchArray[i]))
#define A23(i) (-cos(-_yawArray[i]) * sin(-_pitchArray[i]) - sin(-_yawArray[i]) * sin(-_rollArray[i]) * cos(-_pitchArray[i]))
    
    
     // Set overlay to zero
     glBindFramebuffer(GL_FRAMEBUFFER, _overlayBuffer.frameBuffer);
     glClear(_glClearBits);
     glViewport(0,0, _overlayBuffer.w, _overlayBuffer.h);
     ProgramUniforms &pu = _programs["initialBuffer"];
     
     glUseProgram(pu.program);
     
     glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);
     glUniform2f(pu.uniformMap["imageWH"], (float) _overlayBuffer.w, (float) _overlayBuffer.h);
     
     glBindVertexArrayOES(_drawVertexArray);
     glDrawArrays(GL_TRIANGLES, 0, 3*2);
     
    //printf("%.2f\n", _pitchArray[0]);
    
    for (int j = 0; j < (1 + (_texturesCount - 1) / 4); j++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, _outputBuffer.frameBuffer);
        glViewport(0, 0, _outputBuffer.w, _outputBuffer.h);
        
        ProgramUniforms &pu = _programs["sphericalWarp"];
        glUseProgram(pu.program);
        
        glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);
        glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
        glUniform4f(pu.uniformMap["roll"], _rollArray[j*4+0], _rollArray[j*4 + 1], _rollArray[j*4+ 2], _rollArray[j*4 + 3]);
        glUniform4f(pu.uniformMap["pitch"], _pitchArray[j*4+0], _pitchArray[j*4 + 1] , _pitchArray[j*4 + 2] , _pitchArray[j*4 + 3] );
        glUniform4f(pu.uniformMap["yaw"], _yawArray[j*4+0] , _yawArray[j*4 + 1] , _yawArray[j*4 + 2] , _yawArray[j*4 + 3]);
        glUniform1f(pu.uniformMap["fov"], _fov);
        glUniform4f(pu.uniformMap["a11"], A11(j*4 + 0), A11(j*4 + 1), A11(j*4 + 2), A11(j*4 + 3));
        glUniform4f(pu.uniformMap["a12"], A12(j*4 + 0), A12(j*4 + 1), A12(j*4 + 2), A12(j*4 + 3));
        glUniform4f(pu.uniformMap["a13"], A13(j*4 + 0), A13(j*4 + 1), A13(j*4 + 2), A13(j*4 + 3));
        glUniform4f(pu.uniformMap["a21"], A21(j*4 + 0), A21(j*4 + 1), A21(j*4 + 2), A21(j*4 + 3));
        glUniform4f(pu.uniformMap["a22"], A22(j*4 + 0), A22(j*4 + 1), A22(j*4 + 2), A22(j*4 + 3));
        glUniform4f(pu.uniformMap["a23"], A23(j*4 + 0), A23(j*4 + 1), A23(j*4 + 2), A23(j*4 + 3));
        
        float  phi = 0;//3.1415/4;
        float theta = 0;
         float pxx = cos(phi) * sin(theta);
         float pyy = sin(phi) ;
         float pzz = cos(phi) * cos(theta);
        
        printf("%.2f \n", A11(j*4 + 0) * pxx +  A12(j*4 + 0) * pyy + A13(j*4 + 0) * pzz);
        
        
        
        
        
        int thisIterationTextureCount;
        if (_texturesCount >  (j+1)*4)
            thisIterationTextureCount = 4;  // This means 4 textures to be processed
        else
            thisIterationTextureCount = (_texturesCount) % 4;
            
        if (thisIterationTextureCount == 0)
            thisIterationTextureCount = 4;

        glUniform1i(pu.uniformMap["number"], thisIterationTextureCount);
         
        //glUniform1i(pu.uniformMap["number"], min(_texturesCount-1, 3));
        
        if (_texturesCount > (j*4+0))
        {
            // Index 0 will have the live camera feed
            glActiveTexture(GL_TEXTURE0); //fixme!!!! this is the definition for GLenum GL_TEXTURE0
            glBindTexture(GL_TEXTURE_2D, _imageTextureArray[j*4 + 0]);
            glUniform1i(pu.uniformMap["imageTexture0"],0*4 + 0);
        }
  
        if (_texturesCount > (j*4+1))
        {
            glActiveTexture(GL_TEXTURE0+1);
            glBindTexture(GL_TEXTURE_2D, _imageTextureArray[j*4 + 1]);
            glUniform1i(pu.uniformMap["imageTexture1"],0*4 + 1);
        }
        if (_texturesCount > (j*4+2))
        {
            glActiveTexture(GL_TEXTURE0+2);
            glBindTexture(GL_TEXTURE_2D, _imageTextureArray[j*4 + 2]);
            glUniform1i(pu.uniformMap["imageTexture2"],0*4 + 2);
        }
        if (_texturesCount > (j*4+3))
        {
            glActiveTexture(GL_TEXTURE0+3);
            glBindTexture(GL_TEXTURE_2D, _imageTextureArray[j*4 + 3]);
            glUniform1i(pu.uniformMap["imageTexture3"],0*4 + 3);
        }
        // Overlay uo to now
        
        glActiveTexture(GL_TEXTURE0+7);
        glBindTexture(GL_TEXTURE_2D, _overlayBuffer.texture);
        glUniform1i(pu.uniformMap["imageTexture4"],0*4 +7);
        
        glBindVertexArrayOES(_drawVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3*2);
        copyBuffer2();
        //copyBufferToDisplay();
        /*
         //Get maximum number of texture units
        int MaxTextureUnits;
        glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &MaxTextureUnits);
        printf("%d\n",MaxTextureUnits);
         */
    }
    copyBufferToDisplay();
}


void livePano::copyBuffer2()
{
    
    glBindFramebuffer(GL_FRAMEBUFFER, _overlayBuffer.frameBuffer);
    //glClear(_glClearBits);
    //glViewport(_defaultViewport[0],_defaultViewport[1],_defaultViewport[2],_defaultViewport[3]);
    glViewport(0, 0, _overlayBuffer.w, _overlayBuffer.h);
    
    ProgramUniforms &pu = _programs["copyBuffer"];
    glUseProgram(pu.program);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _outputBuffer.texture);
    glUniform1i(pu.uniformMap["imageTexture"], 0);
    
    glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);
    
    glBindVertexArrayOES(_drawVertexArray);
    glDrawArrays(GL_TRIANGLES, 0, 3*2);
}

void livePano::copyBufferToDisplay()
{
    
    glBindFramebuffer(GL_FRAMEBUFFER, _defaultFrameBuffer);
    glClear(_glClearBits);
    glViewport(_defaultViewport[0],_defaultViewport[1],_defaultViewport[2],_defaultViewport[3]);
    //glViewport(_imageWidth*.25, _imageHeight*.25, _imageWidth*.5, _imageHeight*.5);
    
    ProgramUniforms &pu = _programs["copyBuffer"];
    glUseProgram(pu.program);
    
    
    glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, _defaultTexture);
    glBindTexture(GL_TEXTURE_2D, _outputBuffer.texture);
    glUniform1i(pu.uniformMap["imageTexture"], 0);
    
    glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glBindVertexArrayOES(_drawVertexArray);
    glDrawArrays(GL_TRIANGLES, 0, 3*2);
}




void livePano::myDrawImage(bool warpMode)
{
    //// Mori
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
    
    
    /*
     glBindFramebuffer(GL_FRAMEBUFFER, _defaultFrameBuffer);
     //glBindFramebuffer(GL_FRAMEBUFFER, _outputBuffer.frameBuffer);
     glClear(_glClearBits);
     glViewport(_defaultViewport[0],_defaultViewport[1],_defaultViewport[2],_defaultViewport[3]);
     */
    GLuint m_vaoID;
    
    
    GLfloat vert[] =
    {
        -1.0f, -1.0f, -10.f,       0.0f, 0.0f,
        1.0f, -1.0f, -10.f,       1.0f, 0.0f,
        -1.0f,  1.0f, -10.f,       0.0f, 1.0f,
        
        -1.0f,  1.0f, -10.f,       0.0f, 1.0f,
        1.0f, -1.0f, -10.f,       1.0f, 0.0f,
        1.0f,  1.0f, -10.f,       1.0f, 1.0f
    };
    
    if (!warpMode) {
        glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, _outputBuffer.texture);
        glBindTexture(GL_TEXTURE_2D, _defaultFrameBuffer);
        //glBindTexture(GL_TEXTURE_2D, _imageTextureLive);
        
        //glGenBuffers(1, &m_vaoID);
        //glBindBuffer(GL_ARRAY_BUFFER, m_vaoID);
        //glBufferData(GL_ARRAY_BUFFER, sizeof(float)*sizeof(vert), vert, GL_STATIC_DRAW);
        //glEnableVertexAttribArray(0);
        //glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
        //glEnableVertexAttribArray((GLuint)3);////
        //glVertexAttribPointer((GLuint)3, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));////
        
        glBindVertexArrayOES(_panoVertexBuffer);
        glDrawArrays(GL_TRIANGLES, 0, 3*2);
    }
    else{
        glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, _outputBuffer.texture);
        glBindTexture(GL_TEXTURE_2D, _defaultFrameBuffer);
        //glBindTexture(GL_TEXTURE_2D, _imageTextureLive);
        //glGenBuffers(1, &m_vaoID);
        //glBindBuffer(GL_ARRAY_BUFFER, m_vaoID);
        //glBufferData(GL_ARRAY_BUFFER, sizeof(float)*5*((_m_Stacks+1)*2*(_m_Slices-1)+2), m_VertexData, GL_STATIC_DRAW);
        //glEnableVertexAttribArray(0);
        //glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
        //glEnableVertexAttribArray((GLuint)3);////
        //glVertexAttribPointer((GLuint)3, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));////
        
        glBindVertexArrayOES(_panoSphereVertexBuffer);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (_m_Stacks +1) * 2 * (_m_Slices - 1)+2);//(2 + 1) * (2-1)+2);
        //glDrawArrays(GL_TRIANGLES, 0, 3*2);
    }
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}



void livePano::initialAllTextures()
{
    {
        glBindFramebuffer(GL_FRAMEBUFFER, _overlayBuffer.frameBuffer);
        glClear(_glClearBits);
        glViewport(0,0, _imageWidth*3,_imageHeight*3);
        ProgramUniforms &pu = _programs["initialBuffer"];
        
        glUseProgram(pu.program);
        
        glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, gComputeMVP);
        glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
        
        glBindVertexArrayOES(_drawVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3*2);
    }
    
    {
        glBindFramebuffer(GL_FRAMEBUFFER, _outputBuffer.frameBuffer);
        glClear(_glClearBits);
        glViewport(0,0, _imageWidth*3,_imageHeight*3);
        ProgramUniforms &pu = _programs["initialBuffer"];
        
        glUseProgram(pu.program);
        
        glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, gComputeMVP);
        glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
        
        glBindVertexArrayOES(_drawVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3*2);
    }
    
    /*
    {
        glBindFramebuffer(GL_FRAMEBUFFER, _defaultFrameBuffer);
        glClear(_glClearBits);
        glViewport(0,0, _imageWidth*3,_imageHeight*3);
        ProgramUniforms &pu = _programs["initialBuffer"];
        
        glUseProgram(pu.program);
        
        glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, gComputeMVP);
        glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
        
        glBindVertexArrayOES(_drawVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3*2);
    }
     */
}
///////////////////////////////////////////////////////////////////////////////
#define BUFFER_OFFSET(i) ((char *)0 + (i))


void livePano::initializeOpenGL(GLuint imageTexture, int w, int h, unsigned int computeResReductionLevel)
{
    findSphereVertices(_m_Stacks, _m_Slices, 1.);
    
    glGenBuffers(1, &_computeVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _computeVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(gComputeQuadVertexData), gComputeQuadVertexData, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(SharpenVertexAttribPosition);
    glVertexAttribPointer(SharpenVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    glEnableVertexAttribArray(SharpenVertexAttribTexCoord0);
    glVertexAttribPointer(SharpenVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
    glBindVertexArrayOES(0);
    
    // set or create image texture
    _imageTextureArray[0] = imageTexture;
    _imageTexture = imageTexture;
    _imageWidth  = _imageWidthOriginal  = w;
    _imageHeight = _imageHeightOriginal = h;
    _imageReductionScale = 1.0;
    
    //<_xMin, _yMin, xAdd, _yAdd> are the parameters, used for controlling the position of the resizable ROI window.
    _xMin = _imageWidth/2.;
    _yMin = _imageHeight/2.;
    _xAdd = _imageWidth  * 0.2;
    _yAdd = _imageHeight * 0.2;
    
    // create compute textures and buffers
    createTexturesAndFrameBuffers();
    
    //finally version;
    loadShaders_sphericalWarp();
    loadShaders_copyBuffer();
    loadShaders_initialBuffer();
    loadShaders_myDrawImage();
    
    // if needed, create reduced source image texture
    initialAllTextures();
    initialInputBuffer();

}

void livePano::destroyOpenGL()
{
    _inputBuffer.release();
    glFinish();
    _outputBuffer.release(); //// fixme, to add more buffers
    glFinish();
    _imageTextureReduced.release();//if created, we relese the texture here
    
    
    if(_drawVertexBuffer) {
        glDeleteBuffers(1, &_drawVertexBuffer);
        _drawVertexBuffer = 0;
        glFinish();
    }
    
    if(_drawVertexArray) {
        glDeleteVertexArraysOES(1, &_drawVertexArray);
        _drawVertexArray  = 0;
        glFinish();
    }
    
    if(_computeVertexBuffer) {
        glDeleteBuffers(1, &_computeVertexBuffer);
        _computeVertexBuffer = 0;
        glFinish();
    }
    
    if(_computeVertexArray) {
        glDeleteVertexArraysOES(1, &_computeVertexArray);
        _computeVertexArray  = 0;
        glFinish();
    }
    
}


void livePano::updateView(float screenWidth, float screenHeight)
{
    _screenWidth  = screenWidth;
    _screenHeight = screenHeight;
    
    // set up VB for drawing
    float quadHalfSizeX = (_imageWidth<_imageHeight ? ((float)_imageWidth/_imageHeight) : 1.f);
    float quadHalfSizeY = (_imageWidth>_imageHeight ? ((float)_imageHeight/_imageWidth) : 1.f);
    
    const int stride = 5;
    for(int i=0; i<6; i++) {
        int is = i*stride;
        _drawQuadVertexData[is+0] = gComputeQuadVertexData[is+0] * quadHalfSizeX;
        _drawQuadVertexData[is+1] = gComputeQuadVertexData[is+1] * quadHalfSizeY;
        _drawQuadVertexData[is+2] = gComputeQuadVertexData[is+2];
        _drawQuadVertexData[is+3] = gComputeQuadVertexData[is+3];
        _drawQuadVertexData[is+4] = gComputeQuadVertexData[is+4];
    }
    
    
    GLfloat vert[] =
    {
        -1.0f, -1.0f, -10.f,       0.0f, 0.0f,
        1.0f, -1.0f, -10.f,       1.0f, 0.0f,
        -1.0f,  1.0f, -10.f,       0.0f, 1.0f,
        
        -1.0f,  1.0f, -10.f,       0.0f, 1.0f,
        1.0f, -1.0f, -10.f,       1.0f, 0.0f,
        1.0f,  1.0f, -10.f,       1.0f, 1.0f
    };
    
    
    glGenVertexArraysOES(1, &_drawVertexArray);
    glBindVertexArrayOES(_drawVertexArray);
    
    glGenBuffers(1, &_drawVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _drawVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(_drawQuadVertexData), _drawQuadVertexData, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(SharpenVertexAttribPosition);
    glVertexAttribPointer(SharpenVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    glEnableVertexAttribArray(SharpenVertexAttribTexCoord0);
    glVertexAttribPointer(SharpenVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
    glBindVertexArrayOES(0);
    
    
    
    
    glGenVertexArraysOES(1, &_panoVertexBuffer);
    glBindVertexArrayOES(_panoVertexBuffer);
    
    glGenBuffers(1, &_panoVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _panoVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*sizeof(vert), vert, GL_STATIC_DRAW);
    
    //glEnableVertexAttribArray(SharpenVertexAttribPosition);
    //glVertexAttribPointer(SharpenVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    //glEnableVertexAttribArray(SharpenVertexAttribTexCoord0);
    //glVertexAttribPointer(SharpenVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
    glBindVertexArrayOES(0);
    
    
    
    
    
    glGenVertexArraysOES(1, &_panoSphereVertexBuffer);
    glBindVertexArrayOES(_panoSphereVertexBuffer);
    
    glGenBuffers(1, &_panoSphereVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _panoSphereVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*5*((_m_Stacks+1)*2*(_m_Slices-1)+2), m_VertexData, GL_STATIC_DRAW);
    
    //glBufferData(GL_ARRAY_BUFFER, sizeof(float)*sizeof(vert), vert, GL_STATIC_DRAW);
    
    
    //glEnableVertexAttribArray(SharpenVertexAttribPosition);
    //glVertexAttribPointer(SharpenVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    //glEnableVertexAttribArray(SharpenVertexAttribTexCoord0);
    //glVertexAttribPointer(SharpenVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
    glBindVertexArrayOES(0);
    
    
    
    
    
    
    
    
    // setup view matrix.
    float l = -1.f,   r =  1.f;
    float b = -1.f,   t =  1.f;
    float n =  0.1f,  f =  100.f;
    
    // adjust left and right so that scale=1, centerX=centerY=9 shows full image (always AR=1)
    if(_screenWidth>_screenHeight) b = -(t = _screenHeight / _screenWidth);
    else                           l = -(r = _screenWidth  / _screenHeight);
    
    float scaleToFill = 1.0;
    if(r<quadHalfSizeX) scaleToFill = quadHalfSizeX / r;
    if(t<quadHalfSizeY) scaleToFill = max(scaleToFill, quadHalfSizeY / t);
    l *= scaleToFill;
    t *= scaleToFill;
    r *= scaleToFill;
    b *= scaleToFill;
    
    l = -1.;
    r = 1.;
    b = -.75;
    t = .75;/////fixme
    
    _drawMVP[ 0] = 2.0f/(r-l);  _drawMVP[ 1] = 0.0f;        _drawMVP[ 2] = 0.0f;        _drawMVP[ 3] = 0.0f;
    _drawMVP[ 4] = 0.0f;        _drawMVP[ 5] = 2.0f/(t-b);  _drawMVP[ 6] = 0.0f;        _drawMVP[ 7] = 0.0f;
    _drawMVP[ 8] = 0.0f;        _drawMVP[ 9] = 0.0f;        _drawMVP[10] =-2.0f/(f-n);  _drawMVP[11] = 0.0f;
    _drawMVP[12] =-(r+l)/(r-l); _drawMVP[13] =-(t+b)/(t-b); _drawMVP[14] =-(f+n)/(f-n); _drawMVP[15] = 1.0f;
    
    /*
     GLKMatrix4Multiply(_drawMVP, GLKMatrix4MakeTranslation(10., 1.5, 0));
     GLKMatrix4Multiply(_drawMVP, GLKMatrix4MakeRotation(1., 0, .5, 0));
     */
    
    _viewRect[0] = l;
    _viewRect[1] = t;
    _viewRect[2] = r;
    _viewRect[3] = b;
    
    _imagePanZoomUpdate = false;
}



void livePano::generateRenderToTexture(GLint internalformat, GLenum format, GLenum type,
                                         TextureBuffer &tb, int w, int h, bool linearInterp)
{
    glGenTextures(1, &tb.texture);
    glBindTexture(GL_TEXTURE_2D, tb.texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, internalformat, w, h, 0, format, type, NULL);
    
    
    glGenFramebuffers(1, &tb.frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, tb.frameBuffer);
    glClear(_glClearBits);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tb.texture, 0);
    
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE)
        printf("Framebuffer status: %x", (int)status);
    
    tb.internalformat = internalformat;
    tb.format = format;
    tb.type = type;
    tb.w = w;
    tb.h = h;
}



void livePano::updateTexturesAndFrameBuffers()
{
    _inputBuffer.release();
    _outputBuffer.release();
    _overlayBuffer.release();
}

void livePano::createTexturesAndFrameBuffers()
{
    generateRenderToTexture(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, _inputBuffer, _imageWidth, _imageHeight, true);
    generateRenderToTexture(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, _outputBuffer, 3*_imageWidth, 3*_imageHeight, false); //fixme
    generateRenderToTexture(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, _overlayBuffer, 3* _imageWidth, 3*_imageHeight, true);
    /*
    bool linearInterp = true;
    int w = 3*_imageWidth;
    int h = 3*_imageHeight;
    //glGenTextures(1, &tb.texture);
    _overlayBuffer.texture = GLenum(4);
    glBindTexture(GL_TEXTURE_2D, _overlayBuffer.texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    
    
    glGenFramebuffers(1, &_overlayBuffer.frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, _overlayBuffer.frameBuffer);
    glClear(_glClearBits);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _overlayBuffer.texture, 0);
    
    
    _overlayBuffer.internalformat = GL_RGBA;
    _overlayBuffer.format = GL_RGBA;
    _overlayBuffer.type = GL_UNSIGNED_BYTE;
    _overlayBuffer.w = w;
    _overlayBuffer.h = h;
    */
    
    
}

GLint livePano::compileShaders(const GLchar vShaderText[], const GLchar fShaderText[])
{
    GLint logLength;
    
    //
    GLint vertShader = glCreateShader(GL_VERTEX_SHADER);
    const GLchar *vShaderTextArray[] = {vShaderText};
    GLint vShaderTextLength = (int)strlen(vShaderText);
    glShaderSource(vertShader,1,vShaderTextArray,&vShaderTextLength);
    glCompileShader(vertShader);
    
    glGetShaderiv(vertShader,GL_INFO_LOG_LENGTH, &logLength);
    if(logLength > 0)
    {
        GLchar *log = (GLchar *)malloc(logLength);
        glGetShaderInfoLog(vertShader, logLength, &logLength, log);
        fprintf(stderr, "Fragment ShaderInfoLog: %s\n", log);
        free(log);
    }
    
    //
    GLint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar *fShaderTextArray[] = {fShaderText};
    GLint fShaderTextLength = (int)strlen(fShaderText);
    glShaderSource(fragShader,1,fShaderTextArray,&fShaderTextLength);
    glCompileShader(fragShader);
    
    glGetShaderiv(fragShader,GL_INFO_LOG_LENGTH, &logLength);
    if(logLength > 0)
    {
        GLchar *log = (GLchar *)malloc(logLength);
        glGetShaderInfoLog(fragShader, logLength, &logLength, log);
        fprintf(stderr, "Fragment ShaderInfoLog: %s\n", log);
        free(log);
    }
    
    // Attach shaders to program.
    GLint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    
    // Bind attribute locations. This needs to be done prior to linking.
    glBindAttribLocation(program, SharpenVertexAttribPosition, "position");
    glBindAttribLocation(program, SharpenVertexAttribTexCoord0, "uv0vert");
    
    // link and validate
    glLinkProgram(program);
    glValidateProgram(program);
    
    // Check the status of the compile/link
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    if(logLength > 0)
    {
        GLchar *log = (GLchar *)malloc(logLength);
        glGetProgramInfoLog(program, logLength, &logLength, log);
        fprintf(stderr, "ProgramInfoLog: %s\n", log);
        free(log);
    }
    
    // Release vertex and fragment shaders.
    if (vertShader) {
        glDetachShader(program, vertShader);
        glDeleteShader(vertShader);
    }
    if (fragShader) {
        glDetachShader(program, fragShader);
        glDeleteShader(fragShader);
    }
    
    return program;
}

// ===============================================================================================================
// a stone-age style interface to objective c. there may be better ways..
// ===============================================================================================================
#include "livePanoInterface.h"

void *sharpen_create()
{
    return (void*) new livePano;
}

void sharpen_destroy(void *sharpenContext)
{
    delete (livePano*)sharpenContext;
}

void sharpen_initializeOpenGLTexture(void *sharpenContext, unsigned int imageTexture, unsigned int w, unsigned int h, unsigned int computeResReductionLevel)
{
    ((livePano*)sharpenContext)->initializeOpenGL((GLuint)imageTexture, w, h, computeResReductionLevel);
}

void sharpen_clear(void *sharpenContext)
{
    ((livePano*)sharpenContext)->initSelection(0);
}

void sharpen_step(void *sharpenContext)
{
    ((livePano*)sharpenContext)->step();
}

void sharpen_render(void *sharpenContext, float screenWidth, float screenHeight)
{
    ((livePano*)sharpenContext)->render(screenWidth,screenHeight);
}

void sharpen_changeWarpMode(void *sharpenContext)
{
    ((livePano*)sharpenContext)->changeWarpMode();
}

void sharpen_updateMotionData(void *sharpenContext, float roll, float pitch, float yaw)
{
    ((livePano*)sharpenContext)->updateMotionData(roll, pitch, yaw);
}

void sharpen_setTexture(void *sharpenContext, unsigned int texture, unsigned int w, unsigned int h, bool nextReady)
{
    ((livePano*)sharpenContext)->setTexture(texture, w, h, nextReady);
}


GLuint sharpen_getOutput(void *sharpenContext) //// Mori
{
    return ((livePano*)sharpenContext)->outputFrameBuffer();
}


void sharpen_getViewParams(void *sharpenContext, float viewLTRB[4], float imageLTRBInClipPlane[4])
{
    imageLTRBInClipPlane[0] = ((livePano*)sharpenContext)->_imageRectInClipPlane[0];
    imageLTRBInClipPlane[1] = ((livePano*)sharpenContext)->_imageRectInClipPlane[1];
    imageLTRBInClipPlane[2] = ((livePano*)sharpenContext)->_imageRectInClipPlane[2];
    imageLTRBInClipPlane[3] = ((livePano*)sharpenContext)->_imageRectInClipPlane[3];
    
    viewLTRB[0] = ((livePano*)sharpenContext)->_viewRect[0];
    viewLTRB[1] = ((livePano*)sharpenContext)->_viewRect[1];
    viewLTRB[2] = ((livePano*)sharpenContext)->_viewRect[2];
    viewLTRB[3] = ((livePano*)sharpenContext)->_viewRect[3];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////





void sharpen_getParams(void *sharpenContext, SharpenParams *sep)
{
    livePano *se = (livePano*)sharpenContext;
    
    sep->xMin              = se->_xMin;
    sep->yMin              = se->_yMin;
    sep->xAdd              = se->_xAdd;
    sep->yAdd              = se->_yAdd;
}

void sharpen_setParams(void *sharpenContext, const SharpenParams *sep)
{
    livePano *se = (livePano*)sharpenContext;
    
    se->_xMin              = sep->xMin;
    se->_yMin              = sep->yMin;
    se->_xAdd              = sep->xAdd;
    se->_yAdd              = sep->yAdd;
}


