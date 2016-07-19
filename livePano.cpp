//
//  livePano.cpp
//  panoOGL
//
//  Created by gaoyuan on 4/28/14.
//  Copyright (c) 2014 gaoyuan. All rights reserved.
//


#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>

#include <iostream>
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
static float b = -1.f,   t =  1.0f;
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

const int PanoVertexAttribPosition  = 0;
const int PanoVertexAttribTexCoord0 = 3;
static const GLbitfield _glClearBits = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
enum  _modeSelected {Blur,Denoise,Pano,Brightness,Contrast,Direc_denoise};

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
    bool _displayMode;
    std::vector<float> _rotationMatrixFromImage;
    float _focal;
    
    std::vector<float> _fusedRotation;
    std::vector<std::vector<float>> _fusedRotationArray;
    
    
    float   _imageRectInClipPlane[4];
    float   _viewRect[4];
    
    TextureBuffer _inputBuffer;
    TextureBuffer _outputBuffer;
    TextureBuffer _overlayBuffer;
    
    GLfloat *m_VertexData;
    GLint _m_Stacks;
    GLint _m_Slices;
    bool _warpMode;

private:
    float  _screenWidth, _screenHeight;

protected:
    GLfloat _drawQuadVertexData[sizeof(gComputeQuadVertexData)];
    GLfloat _drawMVP[16];
    
    GLuint _drawVertexArray;
    GLuint _drawVertexBuffer;
    
    GLuint _computeVertexArray;
    GLuint _computeVertexBuffer;
    
    GLuint _drawSphereVertexArray;
    GLuint _panoVertexBuffer;
    GLuint _panoSphereVertexBuffer;
    
    // source image: original
    GLuint        _imageTextureLive; // This will always have the latest image, which is moving on the fly and not stitched yet
    GLuint        _imageTextureArray[32];
    int           _imageWidthOriginal, _imageHeightOriginal;
    
    // input image texture: original
    GLuint        _imageTexture;
    int           _imageWidth, _imageHeight;

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
    void updateMotionData(float roll, float pitch, float yaw, std::vector<float> fusedRotation);
    void setTexture(GLuint imageTexture, unsigned int w, unsigned int h, bool nextReady, bool displayMode);////
    void restart();////
    void initialAllTextures();
    void warpToSphereMap();
    void rotationFromImage(std::vector<float> rotMat);
    void setRefinedRotations(std::vector<std::vector <float>> refinedRotations, float focal);
    void setFocalLength(float focal);
    std::vector<std::vector <float>> getCurrentRotations();
    
    void prepareProjectionMatrix();
    void generateRenderToTexture(GLint internalformat, GLenum format, GLenum type, TextureBuffer &tb, int w, int h, bool linearInterp);
    
    void initSelection(GLuint maskTexture);
    void initialInputBuffer();
    void updateInputBuffer();
    GLuint outputFrameBuffer();

    void findSphereVertices(GLint m_Stacks, GLint m_Slices, GLfloat m_Scale); ////
    void copyBuffer2();
    void copyBufferToDisplay();
    

protected:
    void updateTexturesAndFrameBuffers();
    void createTexturesAndFrameBuffers();
    GLint compileShaders(const GLchar vShaderText[], const GLchar fShaderText[]);
    
    void loadShaders_sphericalWarp();
    void loadShaders_copyBuffer();
    void loadShaders_initialBuffer();
    void loadShaders_sphericalWarpMultiple();
};

#define png_infopp_NULL (png_infopp)NULL

livePano::livePano() :

_screenWidth(-1),
_screenHeight(-1),
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
_texturesCount(1),  // Start from 1. The index 0 will always store the current texture from camera feed
_displayMode(false),
_rotationMatrixFromImage(std::vector<float>(9,0)),
_focal(550)
{
    
    _fusedRotation.resize(9);
    _fusedRotationArray.resize(32);
    
    for (int i = 0; i < 32; i++)
        _fusedRotationArray[i].resize(9);

}

livePano::~livePano()
{
    destroyOpenGL();
}

///////////////////////////////////////////////////////////////////////////////
// CORE


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
    
    warpToSphereMap();
}

void livePano::changeWarpMode()
{
    _warpMode = !_warpMode;
}

void livePano::updateMotionData(float roll, float pitch, float yaw, std::vector<float> fusedRotation)
{
    _roll = roll;
    _pitch = pitch;
    _yaw = yaw;
    _fusedRotation = fusedRotation;
}

void livePano::setTexture(GLuint imageTexture, unsigned int w, unsigned int h, bool nextReady, bool displayMode)
{
    
    //std::cout << imageTexture << std::endl;
    
    bool linearInterp = true;
    if (!displayMode)
    {
        ////_imageTextureArray[0] = imageTexture; // This will always have the latest image, which is moving on the fly and not stitched yet
        _imageWidth  = _imageWidthOriginal  = w;
        _imageHeight = _imageHeightOriginal = h;
        
        GLuint  oldTexture = _imageTextureArray[0];
        
        
        glGenTextures(1, &_imageTextureArray[0]);
        glBindTexture(GL_TEXTURE_2D, _imageTextureArray[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterp ? GL_LINEAR : GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        
        
        GLuint temp;
        glGenFramebuffers(1, &temp);
        glBindFramebuffer(GL_FRAMEBUFFER, temp);
        glClear(_glClearBits);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _imageTextureArray[0], 0);
        
        
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
        glFlush();
        
        
        glDeleteTextures(1, &oldTexture);
        glDeleteFramebuffers(1, &temp);
        temp = 0;
        
    }
    
    
    _pitchArray[0] = _pitch;
    _rollArray[0] = _roll;
    _yawArray[0] = _yaw;
    
    _fusedRotationArray[0] = _fusedRotation;
    
    if (nextReady)
    {
        
        _pitchArray[_texturesCount] = _pitch;
        _rollArray[_texturesCount] = _roll;
        _yawArray[_texturesCount] = _yaw;
        
        _fusedRotationArray[_texturesCount] = _fusedRotation;
        
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
        
        glDeleteFramebuffers(1, &temp);
        temp = 0;
        
        //glReadBufferNV(GL_BACK);
        //glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, w, h, 0);
        //_imageTextureArray[_texturesCount] = imageTexture;
        

        
        
        _texturesCount++;
    }
    /*
    glDeleteTextures(1, &imageTexture);
    imageTexture = 0;
    */
    
    initialInputBuffer();
    
    _displayMode = displayMode;
    
    
}

void livePano::restart()
{
    
    for (int i = 1; i < _texturesCount; i++) {
        glDeleteTextures(1, &_imageTextureArray[i]);
        _imageTextureArray[i] = 0;
    }
    
    _texturesCount = 1; //fixme, delete previous textures and make sure memory consumption drops
}


void livePano::rotationFromImage(std::vector<float> rotMat){
    _rotationMatrixFromImage = rotMat;
    
    
    /*
    //fixme, this should be commented
    _yaw = atan(rotMat[4]/rotMat[0]);
    _roll = atan(rotMat[6] / sqrt(rotMat[7] * rotMat[7] + rotMat[8] * rotMat[8]));
    _pitch = -atan(rotMat[7]/rotMat[8]);
     */
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
     uniform highp float focal;
     uniform int thisBatchHasCamera;
     uniform int displayMode;
     
     //uniform highp vec2 imageWH;
     uniform highp vec4 a11;
     uniform highp vec4 a12;
     uniform highp vec4 a13;
     uniform highp vec4 a21;
     uniform highp vec4 a22;
     uniform highp vec4 a23;
     uniform highp vec4 a31;
     uniform highp vec4 a32;
     uniform highp vec4 a33;
     uniform int number;
     uniform int processedTextures;
     /*
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
     */
     
     void main()
     {
         highp vec2 pos;
         highp vec4 newCol;
         highp vec4 col;
         
         highp float pitch2 ;
         highp float roll2;
         highp float yaw2 ;
         highp float cosYaw;
         highp float sinYaw;
         
         highp vec4 newColAverage = vec4(0., 0., 0., 0.);
         
         highp float theta = (uv0.x - 0.5) * 1. * 3.1415;
         highp float phi = (uv0.y - 0.5) * 3.1415;
         
         int count = 0;
         bool skipTheRest = false;

         for (int jj = 0; jj < number; jj++) {
             
             highp float a11_2;
             highp float a12_2;
             highp float a13_2;
             highp float a21_2;
             highp float a22_2;
             highp float a23_2;
             highp float a31_2;
             highp float a32_2;
             highp float a33_2;
             
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
                 a31_2 = a31.x;
                 a32_2 = a32.x;
                 a33_2 = a33.x;
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
                 a31_2 = a31.y;
                 a32_2 = a32.y;
                 a33_2 = a33.y;
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
                 a31_2 = a31.z;
                 a32_2 = a32.z;
                 a33_2 = a33.z;
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
                 a31_2 = a31.w;
                 a32_2 = a32.w;
                 a33_2 = a33.w;
             }

             
             // Polar (longitude - latitude)
             highp float f = focal / 1000.;
             highp float fovX = .480;//imageWH.x/1000.;
             highp float fovY = .640;//imageWH.y/1000.;//fovX * 640./480.;
             
             highp float pxx = cos(phi) * sin(theta);
             highp float pyy = sin(phi);
             highp float pzz = cos(phi) * cos(theta);

             highp float vrpzz =  (a31_2 * pxx      +       a32_2 * pyy     +       a33_2 * pzz);
             highp float vrpxx = f * (a11_2 * pxx      +       a12_2 * pyy     +       a13_2 * pzz) / vrpzz;
             highp float vrpyy = f * (a21_2 * pxx      +       a22_2 * pyy     +       a23_2 * pzz) / vrpzz;
             
             bool flag = false;
             newCol = vec4(0., 1., 0., 0.);
             
             highp float vrpxxScaled = vrpxx / fovX;
             highp float vrpyyScaled = vrpyy / fovY;
             
             if (vrpzz > 0.)
             {
                 if (vrpxxScaled > -0.5)
                     if (vrpxxScaled < 0.5)
                         if (vrpyyScaled > -0.5)
                             if (vrpyyScaled < 0.5)
                                     flag = true;
             
                 if (flag)
                 {
                     pos = vec2(vrpxxScaled + 0.5, vrpyyScaled + 0.5);
                     
                     // If there is camera feed for this pixel. only use that (thisBatchHasCamera == 0 means this batch has camera)
                     if ((thisBatchHasCamera < 1 && !skipTheRest) || thisBatchHasCamera > 0)
                     {
                         if (jj == 0) {
                             col = texture2D(imageTexture0, pos);
                              if (displayMode < 1)
                              {
                                 if (thisBatchHasCamera < 1)
                                 {
                                     skipTheRest = true;
                                     // Draw border, if this batch has the camera feed (which will be the texture at index 0)
                                     if ( (vrpxxScaled < -0.45) || (vrpxxScaled > 0.45) || (vrpyyScaled  < -0.47) || (vrpyyScaled > 0.47) )
                                         col = vec4(0., 1., 0., 0.);
                                     else if ( (vrpxxScaled  > -.01) && (vrpxxScaled < 0.01) && (vrpyyScaled > -0.01) && (vrpyyScaled < 0.01) )
                                         col = vec4(0., 1., 0., 0.);
                                 }
                              }
                              else{
                                  if (thisBatchHasCamera < 1) // it is not display mode, so show camera feed
                                  {
                                      col = vec4(0., 0., 0., 0.);
                                      count--; // Do not count in this texture
                                  }
                              }
                             
                         } else if (jj == 1) {
                             col = texture2D(imageTexture1, pos);
                         } else if (jj == 2) {
                             col = texture2D(imageTexture2, pos);
                         } else /* if (jj == 3) */ {
                             col = texture2D(imageTexture3, pos);
                         }
                         lowp float coef = 1.2;
                         newCol.x = (coef * col.x > 1.)? 1.:coef * col.x ;
                         newCol.y = (coef * col.y > 1.)? 1.:coef * col.y;
                         newCol.z = (coef * col.z > 1.)? 1.:coef * col.z;
                         newCol.w = 1.;
                         newColAverage += newCol;
                         count++;
                     }
                 }
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
    pu.uniformMap["heigth"]      = glGetUniformLocation(pu.program, "heigth");
    pu.uniformMap["roll"]      = glGetUniformLocation(pu.program, "roll");
    pu.uniformMap["pitch"]      = glGetUniformLocation(pu.program, "pitch");
    pu.uniformMap["yaw"]      = glGetUniformLocation(pu.program, "yaw");
    pu.uniformMap["fov"]      = glGetUniformLocation(pu.program, "fov");
    pu.uniformMap["a11"]      = glGetUniformLocation(pu.program, "a11");
    pu.uniformMap["a12"]      = glGetUniformLocation(pu.program, "a12");
    pu.uniformMap["a13"]      = glGetUniformLocation(pu.program, "a13");
    pu.uniformMap["a21"]      = glGetUniformLocation(pu.program, "a21");
    pu.uniformMap["a22"]      = glGetUniformLocation(pu.program, "a22");
    pu.uniformMap["a23"]      = glGetUniformLocation(pu.program, "a23");
    pu.uniformMap["a31"]      = glGetUniformLocation(pu.program, "a31");
    pu.uniformMap["a32"]      = glGetUniformLocation(pu.program, "a32");
    pu.uniformMap["a33"]      = glGetUniformLocation(pu.program, "a33");
    pu.uniformMap["prevTexture"]      = glGetUniformLocation(pu.program, "prevTexture");
    pu.uniformMap["number"]      = glGetUniformLocation(pu.program, "number");
    pu.uniformMap["thisBatchHasCamera"]      = glGetUniformLocation(pu.program, "thisBatchHasCamera");
    pu.uniformMap["displayMode"]      = glGetUniformLocation(pu.program, "displayMode");
    pu.uniformMap["processedTextures"]      = glGetUniformLocation(pu.program, "processedTextures");
    pu.uniformMap["focal"]      = glGetUniformLocation(pu.program, "focal");
}







void livePano::loadShaders_initialBuffer()
{
    const GLchar fShaderText[] = FRAGMENT_SHADER_SOURCE
    (
     varying highp vec2 uv0;
     void main()
     {
         highp vec4 col = vec4(0.0,0.0,0.,1.);
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

void livePano::warpToSphereMap()
{
    /*
    #define A11(i) (cos(-yawArray[i]) * cos(-rollArray[i]))
    #define A12(i) (-sin(-yawArray[i]) * cos(pitchArray[i]) - cos(-yawArray[i]) * sin(-rollArray[i]) * sin(pitchArray[i]))
    #define A13(i) (sin(-yawArray[i]) * sin(pitchArray[i]) - cos(-yawArray[i]) * sin(-rollArray[i]) * cos(pitchArray[i]))
    #define A21(i) (sin(-yawArray[i]) * cos(-rollArray[i]))
    #define A22(i) (cos(-yawArray[i]) * cos(pitchArray[i]) - sin(-yawArray[i]) * sin(-rollArray[i]) * sin(pitchArray[i]))
    #define A23(i) (-cos(-yawArray[i]) * sin(pitchArray[i]) - sin(-yawArray[i]) * sin(-rollArray[i]) * cos(pitchArray[i]))
    #define A31(i) (sin(-rollArray[i]))
    #define A32(i) (cos(-rollArray[i]) * sin(pitchArray[i]))
    #define A33(i) (cos(-rollArray[i]) * cos(pitchArray[i]))
    */
    /*
#define A11(i) (cos(-yawArray[i]) * cos(rollArray[i]))
#define A12(i) (-sin(-yawArray[i]) * cos(pitchArray[i]) + cos(-yawArray[i]) * sin(rollArray[i]) * sin(pitchArray[i]))
#define A13(i) (sin(-yawArray[i]) * sin(pitchArray[i]) + cos(-yawArray[i]) * sin(rollArray[i]) * cos(pitchArray[i]))
#define A21(i) (sin(-yawArray[i]) * cos(rollArray[i]))
#define A22(i) (cos(-yawArray[i]) * cos(pitchArray[i]) + sin(-yawArray[i]) * sin(rollArray[i]) * sin(pitchArray[i]))
#define A23(i) (-cos(-yawArray[i]) * sin(pitchArray[i]) + sin(-yawArray[i]) * sin(rollArray[i]) * cos(pitchArray[i]))
#define A31(i) (-sin(rollArray[i]))
#define A32(i) (cos(rollArray[i]) * sin(pitchArray[i]))
#define A33(i) (cos(rollArray[i]) * cos(pitchArray[i]))
   */
    
    /*
     #define A11(i) cos(-rollArray[i]) * cos(-yawArray[i])
     #define A12(i) -cos(-rollArray[i]) * sin(-yawArray[i])
     #define A13(i) -sin(-rollArray[i])
     #define A21(i) -sin(pitchArray[i]) * sin(-rollArray[i]) * cos(-yawArray[i]) + cos(pitchArray[i]) * sin(-yawArray[i])
     #define A22(i) sin(pitchArray[i]) * sin(-rollArray[i]) * sin(-yawArray[i]) + cos(pitchArray[i]) * cos(-yawArray[i])
     #define A23(i) -sin(pitchArray[i]) * cos(-rollArray[i])
     #define A31(i) cos(pitchArray[i]) * sin(-rollArray[i]) * cos(-yawArray[i]) + sin(pitchArray[i]) * sin(-yawArray[i])
     #define A32(i) -cos(pitchArray[i]) * sin(-rollArray[i]) * sin(-yawArray[i]) + sin(pitchArray[i]) * cos(-yawArray[i])
     #define A33(i) cos(pitchArray[i]) * cos(-rollArray[i])
     */
 /* // this is the good one
#define A11(i) cos(-rollArray[i]) * cos(-yawArray[i]) - sin(-rollArray[i]) * sin(pitchArray[i]) * sin(-yawArray[i])
#define A12(i) -cos(-rollArray[i]) * sin(-yawArray[i]) - sin(-rollArray[i]) * sin(pitchArray[i]) * cos(-yawArray[i])
#define A13(i) -sin(-rollArray[i]) * cos(pitchArray[i])
#define A21(i) cos(pitchArray[i]) * sin(-yawArray[i])
#define A22(i) cos(pitchArray[i]) * cos(-yawArray[i])
#define A23(i) -sin(pitchArray[i])
#define A31(i) sin(-rollArray[i]) * cos(-yawArray[i]) + cos(-rollArray[i]) * sin(pitchArray[i]) * sin(-yawArray[i])
#define A32(i) -sin(-rollArray[i]) * sin(-yawArray[i]) + cos(-rollArray[i]) * sin(pitchArray[i]) * cos(-yawArray[i])
#define A33(i) cos(-rollArray[i]) * cos(pitchArray[i])
    */

#define A11(i) _fusedRotationArray[i][0]
#define A12(i) _fusedRotationArray[i][1]
#define A13(i) _fusedRotationArray[i][2]
#define A21(i) _fusedRotationArray[i][3]
#define A22(i) _fusedRotationArray[i][4]
#define A23(i) _fusedRotationArray[i][5]
#define A31(i) _fusedRotationArray[i][6]
#define A32(i) _fusedRotationArray[i][7]
#define A33(i) _fusedRotationArray[i][8]
    
    float pitchArray[32];
    float rollArray[32];
    float yawArray[32];
   
    
    
    if (false || _displayMode)
    {
         // Set overlay to zero
         glBindFramebuffer(GL_FRAMEBUFFER, _overlayBuffer.frameBuffer);
         glClear(_glClearBits);
         glViewport(0,0, _overlayBuffer.w, _overlayBuffer.h);
         ProgramUniforms &pu = _programs["initialBuffer"];
         
         glUseProgram(pu.program);
         
         glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);
         glUniform2f(pu.uniformMap["imageWH"], (float) _overlayBuffer.w, (float) _overlayBuffer.h);
         
         glBindVertexArrayOES(_computeVertexArray);
         glDrawArrays(GL_TRIANGLES, 0, 3*2);
        
        for (int j =0; j < _texturesCount; j++) {
            pitchArray[j] = _pitchArray[j] - _pitchArray[0];
            rollArray[j] = _rollArray[j] - _rollArray[0];
            yawArray[j] = _yawArray[j] - _yawArray[0];
        }
        
    }
    else
    {
        // Set input buffer with frame from camera
        glBindFramebuffer(GL_FRAMEBUFFER, _overlayBuffer.frameBuffer);
        glClear(_glClearBits);
        glViewport(0,0, _overlayBuffer.w, _overlayBuffer.h);

        ProgramUniforms &pu = _programs["copyBuffer"];
        glUseProgram(pu.program);

        glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);
        glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _imageTextureArray[0]);

        glUniform1i(pu.uniformMap["imageTexture"], 0);

        glBindVertexArrayOES(_computeVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3*2);
        
        for (int j =0; j < _texturesCount; j++) {
            pitchArray[j] = _pitchArray[j] - _pitchArray[0];
            rollArray[j] = _rollArray[j] - _rollArray[0];
            yawArray[j] = _yawArray[j] - _yawArray[0];
        }
    }
    
    //printf("%.2f\n", _pitchArray[0]);
    int processedTextures = 0;
    //for (int j = 0; j < (1 + (_texturesCount - 1) / 4); j++)
    for (int j = floor (0 + (_texturesCount - 1) / 4); j >=0 ; j--) // Draw in inverse order so that the current camera feed which is at index 0 is rendered last
    {
        
        int thisIterationTextureCount;
        if (_texturesCount >  (j+1)*4)
            thisIterationTextureCount = 4;  // This means 4 textures to be processed
        else
            thisIterationTextureCount = (_texturesCount) % 4;
        
        if (thisIterationTextureCount == 0)
            thisIterationTextureCount = 4;
        
        processedTextures += thisIterationTextureCount;

        glBindFramebuffer(GL_FRAMEBUFFER, _outputBuffer.frameBuffer);
        glViewport(0, 0, _outputBuffer.w, _outputBuffer.h);
        
        ProgramUniforms &pu = _programs["sphericalWarp"];
        glUseProgram(pu.program);
        
        glUniformMatrix4fv(pu.uniformMap["mvpMatrix"], 1, 0, _drawMVP);
        glUniform2f(pu.uniformMap["imageWH"], (float)_imageWidth, (float)_imageHeight);
        glUniform4f(pu.uniformMap["roll"], -rollArray[j*4+0], -rollArray[j*4 + 1], -rollArray[j*4+ 2], -rollArray[j*4 + 3]);
        glUniform4f(pu.uniformMap["pitch"], pitchArray[j*4+0], pitchArray[j*4 + 1] , pitchArray[j*4 + 2] , pitchArray[j*4 + 3] );
        glUniform4f(pu.uniformMap["yaw"], -yawArray[j*4+0] , -yawArray[j*4 + 1] , -yawArray[j*4 + 2] , -yawArray[j*4 + 3]);
        glUniform1f(pu.uniformMap["fov"], _fov);
        glUniform1f(pu.uniformMap["focal"], _focal);
        glUniform1i(pu.uniformMap["thisBatchHasCamera"], j);
        glUniform1i(pu.uniformMap["displayMode"], int(_displayMode));
        glUniform1i(pu.uniformMap["number"], thisIterationTextureCount);
        glUniform1i(pu.uniformMap["processedTextures"], processedTextures);
        
    
        
        glUniform4f(pu.uniformMap["a11"], A11(j*4 + 0), A11(j*4 + 1), A11(j*4 + 2), A11(j*4 + 3));
        glUniform4f(pu.uniformMap["a12"], A12(j*4 + 0), A12(j*4 + 1), A12(j*4 + 2), A12(j*4 + 3));
        glUniform4f(pu.uniformMap["a13"], A13(j*4 + 0), A13(j*4 + 1), A13(j*4 + 2), A13(j*4 + 3));
        glUniform4f(pu.uniformMap["a21"], A21(j*4 + 0), A21(j*4 + 1), A21(j*4 + 2), A21(j*4 + 3));
        glUniform4f(pu.uniformMap["a22"], A22(j*4 + 0), A22(j*4 + 1), A22(j*4 + 2), A22(j*4 + 3));
        glUniform4f(pu.uniformMap["a23"], A23(j*4 + 0), A23(j*4 + 1), A23(j*4 + 2), A23(j*4 + 3));
        glUniform4f(pu.uniformMap["a31"], A31(j*4 + 0), A31(j*4 + 1), A31(j*4 + 2), A31(j*4 + 3));
        glUniform4f(pu.uniformMap["a32"], A32(j*4 + 0), A32(j*4 + 1), A32(j*4 + 2), A32(j*4 + 3));
        glUniform4f(pu.uniformMap["a33"], A33(j*4 + 0), A33(j*4 + 1), A33(j*4 + 2), A33(j*4 + 3));
        
        /*
        glUniform4f(pu.uniformMap["a11"], _rotationMatrixFromImage[0], A11(j*4 + 1), A11(j*4 + 2), A11(j*4 + 3));
        glUniform4f(pu.uniformMap["a12"], _rotationMatrixFromImage[1], A12(j*4 + 1), A12(j*4 + 2), A12(j*4 + 3));
        glUniform4f(pu.uniformMap["a13"], _rotationMatrixFromImage[2], A13(j*4 + 1), A13(j*4 + 2), A13(j*4 + 3));
        glUniform4f(pu.uniformMap["a21"], _rotationMatrixFromImage[3], A21(j*4 + 1), A21(j*4 + 2), A21(j*4 + 3));
        glUniform4f(pu.uniformMap["a22"], _rotationMatrixFromImage[4], A22(j*4 + 1), A22(j*4 + 2), A22(j*4 + 3));
        glUniform4f(pu.uniformMap["a23"], _rotationMatrixFromImage[5], A23(j*4 + 1), A23(j*4 + 2), A23(j*4 + 3));
        glUniform4f(pu.uniformMap["a31"], _rotationMatrixFromImage[6], A31(j*4 + 1), A31(j*4 + 2), A31(j*4 + 3));
        glUniform4f(pu.uniformMap["a32"], _rotationMatrixFromImage[7], A32(j*4 + 1), A32(j*4 + 2), A32(j*4 + 3));
        glUniform4f(pu.uniformMap["a33"], _rotationMatrixFromImage[8], A33(j*4 + 1), A33(j*4 + 2), A33(j*4 + 3));
        */
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
    //glViewport(_defaultViewport[0],_defaultViewport[1]+300,_defaultViewport[2],_defaultViewport[3]/2);
    glViewport(-_imageWidth*0, -_imageHeight*0, _imageWidth*1, _imageHeight*2);
    
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
    
    glEnableVertexAttribArray(PanoVertexAttribPosition);
    glVertexAttribPointer(PanoVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    glEnableVertexAttribArray(PanoVertexAttribTexCoord0);
    glVertexAttribPointer(PanoVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
    glBindVertexArrayOES(0);
    
    // set or create image texture
    _imageTextureArray[0] = imageTexture;
    _imageTexture = imageTexture;
    _imageWidth  = _imageWidthOriginal  = w;
    _imageHeight = _imageHeightOriginal = h;
    
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
    
    glEnableVertexAttribArray(PanoVertexAttribPosition);
    glVertexAttribPointer(PanoVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    glEnableVertexAttribArray(PanoVertexAttribTexCoord0);
    glVertexAttribPointer(PanoVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
    glBindVertexArrayOES(0);
    
    
    
    
    glGenVertexArraysOES(1, &_panoVertexBuffer);
    glBindVertexArrayOES(_panoVertexBuffer);
    
    glGenBuffers(1, &_panoVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _panoVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*sizeof(vert), vert, GL_STATIC_DRAW);
    
    //glEnableVertexAttribArray(PanoVertexAttribPosition);
    //glVertexAttribPointer(PanoVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    //glEnableVertexAttribArray(PanoVertexAttribTexCoord0);
    //glVertexAttribPointer(PanoVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
    glBindVertexArrayOES(0);

    
    
    glGenVertexArraysOES(1, &_panoSphereVertexBuffer);
    glBindVertexArrayOES(_panoSphereVertexBuffer);
    
    glGenBuffers(1, &_panoSphereVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _panoSphereVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*5*((_m_Stacks+1)*2*(_m_Slices-1)+2), m_VertexData, GL_STATIC_DRAW);
    
    //glBufferData(GL_ARRAY_BUFFER, sizeof(float)*sizeof(vert), vert, GL_STATIC_DRAW);
    
    
    //glEnableVertexAttribArray(PanoVertexAttribPosition);
    //glVertexAttribPointer(PanoVertexAttribPosition, 3, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(0));
    
    //glEnableVertexAttribArray(PanoVertexAttribTexCoord0);
    //glVertexAttribPointer(PanoVertexAttribTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, BUFFER_OFFSET(sizeof(float)*3));
    
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
    glBindAttribLocation(program, PanoVertexAttribPosition, "position");
    glBindAttribLocation(program, PanoVertexAttribTexCoord0, "uv0vert");
    
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

void livePano::setRefinedRotations(std::vector<std::vector <float>> refinedRotations, float focal){

    for (int i = 1; i < refinedRotations.size(); i++) {
        /*
        for (int j = 0; j < 9; j++)
            printf("%.2f, ", _fusedRotationArray[i][j]);
        
        printf("\n");
        */
        _fusedRotationArray[i] =refinedRotations[i];
        /*
        for (int j = 0; j < 9; j++)
            printf("%.2f, ", _fusedRotationArray[i][j]);
        
        printf("\n");
        */
    }
    _focal = focal;
}


void livePano::setFocalLength(float focal) {
    _focal = focal;
}
std::vector<std::vector <float>> livePano::getCurrentRotations(){
    return _fusedRotationArray;
}

// ===============================================================================================================
// a stone-age style interface to objective c. there may be better ways..
// ===============================================================================================================
#include "livePanoInterface.h"

void *pano_create()
{
    return (void*) new livePano;
}

void pano_destroy(void *panoContext)
{
    delete (livePano*)panoContext;
}

void pano_initializeOpenGLTexture(void *panoContext, unsigned int imageTexture, unsigned int w, unsigned int h, unsigned int computeResReductionLevel)
{
    ((livePano*)panoContext)->initializeOpenGL((GLuint)imageTexture, w, h, computeResReductionLevel);
}

void pano_clear(void *panoContext)
{
    ((livePano*)panoContext)->initSelection(0);
}

void pano_step(void *panoContext)
{
    ((livePano*)panoContext)->step();
}

void pano_render(void *panoContext, float screenWidth, float screenHeight)
{
    ((livePano*)panoContext)->render(screenWidth,screenHeight);
}

void pano_changeWarpMode(void *panoContext)
{
    ((livePano*)panoContext)->changeWarpMode();
}

void pano_updateMotionData(void *panoContext, float roll, float pitch, float yaw, std::vector<float> fusedRotation)
{
    ((livePano*)panoContext)->updateMotionData(roll, pitch, yaw, fusedRotation);
}

void pano_setTexture(void *panoContext, unsigned int texture, unsigned int w, unsigned int h, bool nextReady, bool displayMode)
{
    ((livePano*)panoContext)->setTexture(texture, w, h, nextReady, displayMode);
}

void pano_restart(void *panoContext)
{
    ((livePano*)panoContext)->restart();
}

void pano_setRotationFromImage(void *panoContext, std::vector<float> rotMat)
{
    ((livePano*)panoContext)->rotationFromImage(rotMat);
}

std::vector<std::vector<float>>  pano_getCurrentRotations(void *panoContext)
{
    return ((livePano*)panoContext)->getCurrentRotations();
}



GLuint pano_getOutput(void *panoContext) //// Mori
{
    return ((livePano*)panoContext)->outputFrameBuffer();
}

void pano_setRefinedRotations(void *panoContext, std::vector<std::vector <float>> refinedRotations, float focal){
    ((livePano*)panoContext)->setRefinedRotations(refinedRotations, focal);
}

void pano_setFocalLength(void *panoContext, float focal){
    ((livePano*)panoContext)->setFocalLength(focal);
}

void pano_getViewParams(void *panoContext, float viewLTRB[4], float imageLTRBInClipPlane[4])
{
    imageLTRBInClipPlane[0] = ((livePano*)panoContext)->_imageRectInClipPlane[0];
    imageLTRBInClipPlane[1] = ((livePano*)panoContext)->_imageRectInClipPlane[1];
    imageLTRBInClipPlane[2] = ((livePano*)panoContext)->_imageRectInClipPlane[2];
    imageLTRBInClipPlane[3] = ((livePano*)panoContext)->_imageRectInClipPlane[3];
    
    viewLTRB[0] = ((livePano*)panoContext)->_viewRect[0];
    viewLTRB[1] = ((livePano*)panoContext)->_viewRect[1];
    viewLTRB[2] = ((livePano*)panoContext)->_viewRect[2];
    viewLTRB[3] = ((livePano*)panoContext)->_viewRect[3];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////





void pano_getParams(void *panoContext, PanoParams *sep)
{
    livePano *se = (livePano*)panoContext;
    
    sep->xMin              = se->_xMin;
    sep->yMin              = se->_yMin;
    sep->xAdd              = se->_xAdd;
    sep->yAdd              = se->_yAdd;
}

void pano_setParams(void *panoContext, const PanoParams *sep)
{
    livePano *se = (livePano*)panoContext;
    
    se->_xMin              = sep->xMin;
    se->_yMin              = sep->yMin;
    se->_xAdd              = sep->xAdd;
    se->_yAdd              = sep->yAdd;
}


