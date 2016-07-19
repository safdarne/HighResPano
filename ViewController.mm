#import "ViewController.h"
#import "livePanoInterface.h"  
#import <QuartzCore/QuartzCore.h>
#import "cvEstimatorInterface.h"
#import "GSVideoProcessor.h"

#include "opencv2/opencv.hpp"
#include <numeric>      // std::iota

//#include "lapack_lapack_ivf.h"


#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#define PI 3.14159

@interface ViewController () <GSVideoProcessorDelegate> {
    void  *_panoContext;
    void *_cvEstimator;
    
    GLuint _imageTexture, _oldTexture;
    int    _imageW;
    int    _imageH;
    int    _width;
    int    _height;
    bool   _next;
    bool _displayMode;
    bool cvProcessFramesFlag;
    
    
    float _roll, _rollRef;
    float _pitch, _pitchRef;
    float _pitchArray[100];
    float _rollArray[100];
    float _yawArray[100];
    float _xMotion, _xVelocity;
    float _focal;

    
    int _storedImagesCount;
    float _yaw, _yawRef;
    bool _initialized;
    std::vector<std::vector <float>> _refinedRotations;
    std::vector<float> _rotationFromQuaternion3x3;
    std::vector<std::vector<float>> _vcRotationFromSensorArray;
    std::vector<std::vector<int>> _closestFrames;
}

@property (nonatomic, readwrite, strong) GSVideoProcessor *videoProcessor;
@property (nonatomic, readwrite, assign) CVOpenGLESTextureCacheRef videoTextureCache;
@property (strong, nonatomic) GLKBaseEffect *baseEffect;
@property (strong, nonatomic) GLKTextureInfo *background;

@property (strong, nonatomic) GLKTextureLoader *asyncTextureLoader;
@property (strong, nonatomic) GLKTextureInfo *hugeTexture;
@property (strong, nonatomic) EAGLContext *context;
@property (strong, nonatomic) GLKBaseEffect *effect;

- (void)initialize;
- (void)setupGL;
- (void)tearDownGL;
- (BOOL)compileShader:(GLuint *)shader type:(GLenum)type file:(NSString *)file;


- (void)setupCV;
- (cv::Mat) cvMatFromUIImage:(UIImage *)image;
- (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat;

@end

@implementation ViewController

@synthesize videoProcessor = videoProcessor_;
@synthesize videoTextureCache = videoTextureCache_;
@synthesize baseEffect = baseEffect_;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// load input image, and init a Pano context
- (void)initialize
{
    for (int i = 0; i <9 ;i++)
        _rotationFromQuaternion3x3.push_back(0);
    
    _rotationFromQuaternion3x3[0] = 1;
    _rotationFromQuaternion3x3[4] = 1;
    _rotationFromQuaternion3x3[8] = 1;
    
    _vcRotationFromSensorArray.push_back(_rotationFromQuaternion3x3);
    
    _xMotion = 0;
    _xVelocity = 0;
    
    _focal = 520;
    
    _closestFrames.push_back(std::vector<int>());
}

- (void)setupGL
{
    [EAGLContext setCurrentContext:self.context];

    _imageTexture = info.name;
    _imageW       = 640;/////fixme, info.width;
    _imageH       = 480;////fixme, info.height;
    
    _width = 480;
    _height = 640;
    
    // create/initialize a Pano context
    _panoContext = pano_create();
    pano_initializeOpenGLTexture(_panoContext, _imageTexture, _imageW, _imageH, 0);
}

// destroy the Pano context
- (void)tearDownGL
{
    [EAGLContext setCurrentContext:self.context];
    self.effect = nil;
    pano_destroy(_panoContext);
}

- (void)setupCV
{
    _cvEstimator = cvEstimator_create();
    cvEstimator_initialize(_cvEstimator, 480, 640);  //fixme
}


////**********************************************************************

// selection update here
- (void)glkView:(GLKView *)view drawInRect:(CGRect)rect
{
    [self.rollLabel setText:[NSString stringWithFormat:@"Roll:   %.01f", ( _roll) * 180 / PI]];
    [self.pitchLabel setText:[NSString stringWithFormat:@"Pitch: %.01f", ( _pitch) * 180 / PI]];
    [self.yawLabel setText:[NSString stringWithFormat:@"Yaw:   %.01f", ( _yaw) * 180 / PI]];
    
    glClearColor(1.f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    CGRect screenBound = [[UIScreen mainScreen] bounds];
    CGSize screenSize = screenBound.size;

    float width  = (float)screenSize.width;
    float height = (float)screenSize.height;
    
    _frameFromCamera = [self.videoProcessor frameFromCamera];
    cv::Mat nextMat = [self cvMatFromUIImage:_frameFromCamera];
    
    if (!_displayMode)
        [self updateTexture:_frameFromCamera];

    CGSize size = [_frameFromCamera size];

    if (_storedImagesCount > 0) // started capturing
    {
        float minRollDiff = 10;
        float minPitchDiff = 10;
        float minYawDiff = 10;
        float minDist = 10;
        float dist;
        std::vector<float> distFromFrames(_storedImagesCount + 1);
        
        
        for (int i = 1; i <= _storedImagesCount; i++)
        {
            minRollDiff = fmin(minRollDiff, fabsf(_roll - _rollArray[i]));
            minPitchDiff = fmin(minPitchDiff, fabsf(_pitch - _pitchArray[i]));
            minYawDiff = fmin(minYawDiff, fabsf(_yaw - _yawArray[i]));
            dist = fmax(fabsf(_roll - _rollArray[i]) / fabsf(atan(_width / _focal)), fabsf(_pitch - _pitchArray[i])  / fabsf(atan(_height / _focal)));
            minDist = fmin(minDist, dist);
            distFromFrames[i] = dist;
        }
        
        if (!_displayMode)
        {
            
            // Check if a new frame should be stored in memory
            if (minDist > 0.5) // 50% overlap either is x or y direction
            {
                _next = true;
                _storedImagesCount++;
                _rollArray[_storedImagesCount] = _roll;
                _pitchArray[_storedImagesCount] = _pitch;
                _yawArray[_storedImagesCount] = _yaw;
                AudioServicesPlaySystemSound(1108);
                
                // Sort distances and find closest frame to the frame under process
                std::vector<int> ind(distFromFrames.size());
                std::iota(ind.begin(), ind.end(), 0); //fixme, check if for consistency the index should start from 0 or 1
                auto comparator = [&distFromFrames](int a, int b){ return distFromFrames[a] < distFromFrames[b]; };
                std::sort(ind.begin(), ind.end(), comparator);
                //std::sort(distFromFrames.begin(), distFromFrames.end(), comparator);
                

                
                /*
                std::cout << std::endl;
                for (int i = 0; i < ind.size(); i++)
                {
                    std::cout << distFromFrames[i] << "," ;
                }
                std::cout << std::endl;
                */
                
                for (int i = 1; i < ind.size(); i++)
                {
                    // Mark if there is almost no overlap
                    if (distFromFrames[ind[i]] > 1.0)
                        ind[i] = -1;
                }

                /*
                for (int i = 1; i < ind.size(); i++)
                {
                    std::cout << ind[i] << ",";
                }
                std::cout << "-" <<  std::endl;
                */
                
                
                if (_storedImagesCount >= _closestFrames.size())
                    _closestFrames.resize(_storedImagesCount + 1);
                
                _closestFrames[_storedImagesCount] = ind;
                /*
                for (int k = 1; k < _closestFrames.size(); k++)
                {
                    for (int m = 0; m < _closestFrames[k].size(); m++)
                    {
                        std::cout << _closestFrames[k][m] << "," ;
                    }
                    std::cout << "-" <<  std::endl;
                }
                
                std::cout << std::endl <<  std::endl <<  std::endl;
                */
                int tttt = 1;
                tttt++;
            }
        }
    }
    else
        _refAttitude = nil; // Capturing not started yet, set the reference to the current reading of motion sensor so that the current frame appears in the center of the screen
    
    
    // Send the vision module the rotation estimation from the device sensor, and get the updated rotation estimation based on image features and sensor data
    cvEstimator_setRotationFromSensor(_cvEstimator, _rotationFromQuaternion3x3);
    if (_next)
    {
        _vcRotationFromSensorArray.push_back(_rotationFromQuaternion3x3);
        cvEstimator_saveNextFrame(_cvEstimator, nextMat);
        
        // Do bundle adjustment
        if (_storedImagesCount > 2)
            [self.refinementButton sendActionsForControlEvents:UIControlEventTouchUpInside];
    }
    else
        cvEstimator_procFrame(_cvEstimator, nextMat, cvProcessFramesFlag);
    std::vector<float> rotationMatrixFromImage = cvEstimator_getRotation(_cvEstimator);
    
    // Send the warping and display module, the estimated rotation for the current image
    //pano_setRotationFromImage(_panoContext, rotationMatrixFromImage);
    std::vector<float> fusedRotation = _rotationFromQuaternion3x3; //fixme, this should be the fusion of motion data and image data, not only motion
    pano_updateMotionData(_panoContext, _roll, _pitch, _yaw, _rotationFromQuaternion3x3); //fixme, this should be uncommented !!!!!!!!!!!! careful what to send as rotration
    
    
    
    
    
    
    
    // Render the frames
    pano_step(_panoContext);
    pano_render(_panoContext, width, height);
    
    // pano_setTexture(_panoContext, _imageTexture, size.height, size.width, _next, _displayMode); //// fixme
    pano_setTexture(_panoContext, self.hugeTexture.name, size.height, size.width, _next, _displayMode); //// fixme
    
    // Delete the texture corresponding to the previous camera frame
    glDeleteTextures(1, &_oldTexture);
    _oldTexture = 0;
    
    _next = false;
    
    /*
    textureFromCamera = [self.videoProcessor textureFromCamera];
    NSError *error;
    */
    
    //pano_initializeOpenGLTexture(_panoContext, _imageTexture, _imageW, _imageH, 0);
    
    /*
    info = [GLKTextureLoader textureWithCGImage:frameFromCamera.CGImage options:nil error:&error];
    if (info.name)
        glDeleteTextures(1, info.name);
    */
    /*
    NSError  *error;
    NSString *imageName = [NSString stringWithFormat:@"d4"]; // intial input image
    NSString *path = [[NSBundle mainBundle] pathForResource:imageName ofType:@"png"];
    GLKTextureInfo *info = [GLKTextureLoader textureWithContentsOfFile:path options:nil error:&error];
    */
    /*
    _imageTexture = info.name;
    _imageW       = info.width;
    _imageH       = info.height;
    */
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// some other stuffs and unused GLK funcs
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

- (void)viewDidLoad
{
    cvProcessFramesFlag = false;
    
    if (cvProcessFramesFlag)
        [self.flagLabel setText:[NSString stringWithFormat:@"On"]];
    else
        [self.flagLabel setText:[NSString stringWithFormat:@"Off"]];
    
    _initialized = false;
    NSLog(@"viewDidLoad!");
    [super viewDidLoad];
    [self initialize];
    
    self.context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    if (!self.context) {
        NSLog(@"Failed to create ES context");
    }
    
    self.asyncTextureLoader = [[GLKTextureLoader alloc] initWithSharegroup:self.context.sharegroup];
    self.preferredFramesPerSecond = 30;

    GLKView *view = (GLKView *)self.view;
    view.context = self.context;
    view.drawableDepthFormat = GLKViewDrawableDepthFormat24;
    
    [self setupCV]; //fixme, release at the end
    [self setupGL];
    
    CMAttitudeReferenceFrame frame = CMAttitudeReferenceFrameXArbitraryCorrectedZVertical;
    self.motionManager = [[CMMotionManager alloc] init];
    self.motionManager.deviceMotionUpdateInterval = 1.0/30.0;
    [self.motionManager startDeviceMotionUpdatesUsingReferenceFrame:frame];
    
    NSTimer* timer = [NSTimer timerWithTimeInterval:1.0/30.f target:self selector:@selector(updateLabel) userInfo:nil repeats:YES];
    [[NSRunLoop mainRunLoop] addTimer:timer forMode:NSRunLoopCommonModes];
    
    _accelerometer = [UIAccelerometer sharedAccelerometer];
    _accelerometer.updateInterval = 1./20.;
    _accelerometer.delegate = self;
    
    /*
    // Verify the type of view created automatically by the
    // Interface Builder storyboard
    NSAssert([view isKindOfClass:[GLKView class]],
             @"View controller's view is not a GLKView");
    
    // Create an OpenGL ES 2.0 context and provide it to the
    // view
    view.context = [[EAGLContext alloc]
                    initWithAPI:kEAGLRenderingAPIOpenGLES2];
    view.layer.opaque = YES;
    
    CAEAGLLayer *eaglLayer = (CAEAGLLayer *)view.layer;
    eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:
                                    [NSNumber numberWithBool:NO], kEAGLDrawablePropertyRetainedBacking,
                                    kEAGLColorFormatRGBA8, kEAGLDrawablePropertyColorFormat,
                                    nil];
    
    // Make the new context current
    [EAGLContext setCurrentContext:view.context];
    
    
    // Create a base effect that provides standard OpenGL ES 2.0
    // Shading Language programs and set constants to be used for
    // all subsequent rendering
    self.baseEffect = [[GLKBaseEffect alloc] init];
    
    if(nil == self.background)
    {
        self.background = [GLKTextureLoader textureWithCGImage:
                           [[UIImage imageNamed:@"Elephant.jpg"] CGImage]
                                                       options:nil
                                                         error:NULL];
    }
    self.baseEffect.texture2d0.name = self.background.name;
    self.baseEffect.texture2d0.target = self.background.target;
    
    */
    //  Create a new CVOpenGLESTexture cache
    CVReturn err = CVOpenGLESTextureCacheCreate(
                                                kCFAllocatorDefault,
                                                NULL,
                                                (__bridge CVEAGLContext)((__bridge void *)view.context),
                                                NULL,
                                                &videoTextureCache_);
    
    if (err)
    {
        NSLog(@"Error at CVOpenGLESTextureCacheCreate %d", err);
    }
    
    // Keep track of changes to the device orientation so we can
    // update the video processor
    NSNotificationCenter *notificationCenter =
    [NSNotificationCenter defaultCenter];
    [notificationCenter addObserver:self 
                           selector:@selector(deviceOrientationDidChange) 
                               name:UIDeviceOrientationDidChangeNotification 
                             object:nil];
    [[UIDevice currentDevice] 
     beginGeneratingDeviceOrientationNotifications];
    
    // Setup video processor
    self.videoProcessor = [[GSVideoProcessor alloc] init];
    self.videoProcessor.delegate = self;
    [self.videoProcessor setupAndStartCaptureSession];
    
    _storedImagesCount = 0;
    _displayMode = false;
}


- (void)dealloc
{
    [self tearDownGL];
    
    if ([EAGLContext currentContext] == self.context) {
        [EAGLContext setCurrentContext:nil];
    }
    
    _accelerometer.delegate = nil;
}

- (void)didReceiveMemoryWarning
{
    NSLog(@"didRecieveMemoryWarning");
    [super didReceiveMemoryWarning]; //fixme!
    
    if ([self isViewLoaded] && ([[self view] window] == nil)) {
        self.view = nil;
        
        [self tearDownGL];
        
        if ([EAGLContext currentContext] == self.context) {
            [EAGLContext setCurrentContext:nil];
        }
        self.context = nil;
    }
    
    // Dispose of any resources that can be recreated.
}

- (void)updateLabel {
    [self getMotionData];
}

- (void)getMotionData {
    CMDeviceMotion *deviceMotion = self.motionManager.deviceMotion;
    
    CMAcceleration acceleration = self.motionManager.deviceMotion.userAcceleration;
    
    float xAccel = acceleration.x;
    if (std::abs(xAccel) < 0.01)
        xAccel = 0;
    
     _xVelocity += xAccel * 1. / 30.;
     _xMotion += 0.5 * xAccel * (1. / 30.) * (1. / 30.) + _xVelocity * (1. / 30.);
     //_xMotion +=  _xVelocity * (1. / 30.);
     // printf("%.4f, %.4f, %.3f, %.3f, %.3f \n", std::abs(_xMotion), std::abs(_xVelocity), (xAccel), std::abs(acceleration.y), std::abs(acceleration.z));
     
    
    if(deviceMotion == nil)
        return;
    
    _currentAttitude = deviceMotion.attitude;
    _attitudeChange = _currentAttitude;
    
    if (_refAttitude == nil)
        _refAttitude = [_currentAttitude copy];
    
    [_attitudeChange multiplyByInverseOfAttitude:_refAttitude];
    
    _roll = _attitudeChange.roll;
    _pitch = _attitudeChange.pitch;
    _yaw = _attitudeChange.yaw;
    
    /*
    _roll = -atan2(-quaternionChange.x*quaternionChange.z - quaternionChange.w*quaternionChange.y, .5 - quaternionChange.y*quaternionChange.y - quaternionChange.z*quaternionChange.z);
    _pitch = -asin(-2*(quaternionChange.y*quaternionChange.z + quaternionChange.w*quaternionChange.x));
    _yaw = -atan2(quaternionChange.x*quaternionChange.y - quaternionChange.w*quaternionChange.z, .5 - quaternionChange.x*quaternionChange.x - quaternionChange.z*quaternionChange.z);
    */
        
    CMQuaternion q = _attitudeChange.quaternion;
    GLKQuaternion quaternionChangeGLK = GLKQuaternionMake(q.x, q.y, -q.z, -q.w); // "-" is added to set the rotation to its inverse, as needed for stitching
    
    _rotationFromQuaternion = GLKMatrix4MakeWithQuaternion(quaternionChangeGLK);
    _rotationFromQuaternion3x3[0] = _rotationFromQuaternion.m[0];
    _rotationFromQuaternion3x3[1] = _rotationFromQuaternion.m[1];
    _rotationFromQuaternion3x3[2] = _rotationFromQuaternion.m[2];
    _rotationFromQuaternion3x3[3] = _rotationFromQuaternion.m[4];
    _rotationFromQuaternion3x3[4] = _rotationFromQuaternion.m[5];
    _rotationFromQuaternion3x3[5] = _rotationFromQuaternion.m[6];
    _rotationFromQuaternion3x3[6] = _rotationFromQuaternion.m[8];
    _rotationFromQuaternion3x3[7] = _rotationFromQuaternion.m[9];
    _rotationFromQuaternion3x3[8] = _rotationFromQuaternion.m[10];
    
    _vcRotationFromSensorArray[0] = _rotationFromQuaternion3x3;
}



/////////////////////////////////////////////////////////////////
//
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
    // Native video orientation is landscape with the button on the right.
    // The video processor rotates vide as needed, so don't autorotate also
    return (interfaceOrientation == UIInterfaceOrientationLandscapeRight);
}


/////////////////////////////////////////////////////////////////
//
- (void)deviceOrientationDidChange
{
    UIDeviceOrientation orientation = [[UIDevice currentDevice] orientation];
    /*
     // Don't update the reference orientation when the device orientation is
     // face up/down or unknown.
     if(UIDeviceOrientationIsPortrait(orientation))
     {
     [self.videoProcessor setReferenceOrientation:
     AVCaptureVideoOrientationPortrait];
     }
     else if(UIDeviceOrientationIsLandscape(orientation) )
     {
     [self.videoProcessor setReferenceOrientation:
     AVCaptureVideoOrientationLandscapeRight];
     }
     */
    [self.videoProcessor setReferenceOrientation:
     AVCaptureVideoOrientationPortrait];
}



/////////////////////////////////////////////////////////////////
//
- (CGRect)textureSamplingRectForCroppingTextureWithAspectRatio:
(CGSize)textureAspectRatio
                                                 toAspectRatio:(CGSize)croppingAspectRatio
{
    CGRect normalizedSamplingRect = CGRectZero;
    CGSize cropScaleAmount =
    CGSizeMake(croppingAspectRatio.width / textureAspectRatio.width,
               croppingAspectRatio.height / textureAspectRatio.height);
    CGFloat maxScale = fmax(cropScaleAmount.width, cropScaleAmount.height);
    CGSize scaledTextureSize =
    CGSizeMake(textureAspectRatio.width * maxScale,
               textureAspectRatio.height * maxScale);
    
    if ( cropScaleAmount.height > cropScaleAmount.width )
    {
        normalizedSamplingRect.size.width =
        croppingAspectRatio.width / scaledTextureSize.width;
        normalizedSamplingRect.size.height = 1.0;
    }
    else
    {
        normalizedSamplingRect.size.height =
        croppingAspectRatio.height / scaledTextureSize.height;
        normalizedSamplingRect.size.width = 1.0;
    }
    
    // Center crop
    normalizedSamplingRect.origin.x =
    (1.0 - normalizedSamplingRect.size.width)/2.0;
    normalizedSamplingRect.origin.y =
    (1.0 - normalizedSamplingRect.size.height)/2.0;
    
    return normalizedSamplingRect;
}


/////////////////////////////////////////////////////////////////
//
- (void)renderWithSquareVertices:(const GLfloat*)squareVertices
                 textureVertices:(const GLfloat*)textureVertices
{
    // Update attribute values.
    glVertexAttribPointer(GLKVertexAttribPosition,
                          2,
                          GL_FLOAT,
                          0,
                          0,
                          squareVertices);
    glEnableVertexAttribArray(GLKVertexAttribPosition);
    glVertexAttribPointer(GLKVertexAttribTexCoord0,
                          2,
                          GL_FLOAT,
                          0,
                          0,
                          textureVertices);
    glEnableVertexAttribArray(GLKVertexAttribTexCoord0);
    
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}


#pragma mark - GSVideoProcessorDelegate

/////////////////////////////////////////////////////////////////
//
- (CVOpenGLESTextureRef)pixelBufferReadyForDisplay:(CVPixelBufferRef)pixelBuffer
{
    NSParameterAssert(pixelBuffer);
    NSAssert(nil != videoTextureCache_, @"nil texture cache");
   	
    // Create a CVOpenGLESTexture from the CVImageBuffer
    size_t frameWidth = CVPixelBufferGetWidth(pixelBuffer);
    size_t frameHeight = CVPixelBufferGetHeight(pixelBuffer);
    CVOpenGLESTextureRef texture = NULL;
    CVReturn err = CVOpenGLESTextureCacheCreateTextureFromImage(
                                                                kCFAllocatorDefault,
                                                                videoTextureCache_,
                                                                pixelBuffer,
                                                                NULL,
                                                                GL_TEXTURE_2D,
                                                                GL_RGBA,
                                                                frameWidth,
                                                                frameHeight,
                                                                GL_RGBA,
                                                                GL_UNSIGNED_BYTE,
                                                                0,
                                                                &texture);
    
    
    if (!texture || err)
    {
        NSLog(@"CVOpenGLESTextureCacheCreateTextureFromImage (error: %d)",
              err);
        return nil;
    }
    
    /*
    static const GLfloat squareVertices[] =
    {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        -1.0f,  1.0f,
        1.0f,  1.0f,
    };
    
    // The texture vertices are set up such that we flip the texture vertically.
    // This is so that our top left origin buffers match OpenGL's bottom left texture coordinate system.
    CGRect textureSamplingRect =
    [self textureSamplingRectForCroppingTextureWithAspectRatio:
     CGSizeMake(frameWidth, frameHeight)
                                                 toAspectRatio:self.view.bounds.size];
    
    GLfloat textureVertices[] =
    {
        CGRectGetMinX(textureSamplingRect), CGRectGetMaxY(textureSamplingRect),
        CGRectGetMaxX(textureSamplingRect), CGRectGetMaxY(textureSamplingRect),
        CGRectGetMinX(textureSamplingRect), CGRectGetMinY(textureSamplingRect),
        CGRectGetMaxX(textureSamplingRect), CGRectGetMinY(textureSamplingRect),
    };
    
    glBindTexture(
                  CVOpenGLESTextureGetTarget(texture),
                  CVOpenGLESTextureGetName(texture));
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);   
    
    // Draw the texture on the screen with OpenGL ES 2
    glDisable(GL_BLEND);
    [self renderWithSquareVertices:squareVertices
                   textureVertices:textureVertices];
    
    glBindTexture(CVOpenGLESTextureGetTarget(texture), 0);
    
    // Flush the CVOpenGLESTexture cache and release the texture
    CVOpenGLESTextureCacheFlush(videoTextureCache_, 0);
    CFRelease(texture);
    
    // Draw the texture on the screen with OpenGL ES 2
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);
    [self.baseEffect prepareToDraw];
    [self renderWithSquareVertices:squareVertices
                   textureVertices:textureVertices];
    glFlush();
    
    // Present
    //GLKView *glkView = (GLKView *)self.view;
    //[glkView.context presentRenderbuffer:GL_RENDERBUFFER];
    */
    return texture;
}


#pragma mark - update video statistics

/////////////////////////////////////////////////////////////////

- (void) updateTexture:(UIImage *)chosenImage
{

    NSData *newInput =  UIImagePNGRepresentation(chosenImage);
    
    {
        
        NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:YES],GLKTextureLoaderOriginBottomLeft, nil];
        dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
        
        void (^complete) (GLKTextureInfo*, NSError*) = ^(GLKTextureInfo *texture,
                                                         NSError *e){
            
            dispatch_sync(dispatch_get_main_queue(), ^{
                
                _oldTexture = texture.name; //fixme, it might happen that the texture is deleted before or in middle of being used?
                self.hugeTexture = texture;
                _imageW = self.hugeTexture.width;
                _imageH = self.hugeTexture.height;
                //pano_updateMotionData(_panoContext, _roll, _pitch, _yaw);
                
                
                if (!_initialized)
                {
                    _initialized = true;
                    //_next = true;
                }
            });
        };
        
        [self.asyncTextureLoader textureWithContentsOfData:newInput
                                                   options:options
                                                     queue:queue
                                         completionHandler:complete];
        
        /*
        NSString *imageName = [NSString stringWithFormat:@"d5"]; // intial input image
        NSString *path = [[NSBundle mainBundle] pathForResource:imageName ofType:@"png"];
        
        [self.asyncTextureLoader textureWithContentsOfFile:path
                                                   options:options
                                                     queue:queue
                                         completionHandler:complete];*/
    }
    
}
//////////////////////////////////////////

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

//////////////////////////////////////////

- (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

//////////////////////////////////////////

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

//////////////////////////////////////////
#pragma mark UIAccelerometerDelegate Methods

- (void)accelerometer:(UIAccelerometer *)meter
        didAccelerate:(UIAcceleration *)acceleration
{
    /*
    _xVelocity += acceleration.x * 1. / 20.;
    //_xMotion += 0.5 * acceleration.x * (1. / 60.) * (1. / 60.) + _xVelocity * (1. / 60.);
    _xMotion +=  _xVelocity * (1. / 20.);
    printf("%.4f, %.3f, %.3f, %.3f \n", _xMotion, acceleration.x, acceleration.y, acceleration.z);
    */
    
    
    //std::cout << acceleration.x << std::endl;
    /*
    xLabel.text = [NSString stringWithFormat:@"%f", acceleration.x];
    xBar.progress = ABS(acceleration.x);
    
    yLabel.text = [NSString stringWithFormat:@"%f", acceleration.y];
    yBar.progress = ABS(acceleration.y);
    
    zLabel.text = [NSString stringWithFormat:@"%f", acceleration.z];
    zBar.progress = ABS(acceleration.z);
     */
}


//////////////////////////////////////////
#pragma mark - GLKView and GLKViewController delegate methods

- (BOOL)compileShader:(GLuint *)shader type:(GLenum)type file:(NSString *)file
{
    return YES;
}

- (IBAction)cvProcessFrames:(id)sender {
    cvProcessFramesFlag = !cvProcessFramesFlag;
    if (cvProcessFramesFlag)
        [self.flagLabel setText:[NSString stringWithFormat:@"On"]];
    else
        [self.flagLabel setText:[NSString stringWithFormat:@"Off"]];
}

- (IBAction)warpToSphere:(id)sender {
    pano_changeWarpMode(_panoContext);
}

- (IBAction)startCapturing:(id)sender {
    _next = true;
    _storedImagesCount++;
    _rollArray[_storedImagesCount] = _roll;
    _pitchArray[_storedImagesCount] = _pitch;
    _yawArray[_storedImagesCount] = _yaw;
    _displayMode = false;
    AudioServicesPlaySystemSound(1108);
}

- (IBAction)stopCapturing:(id)sender {
    _next = false;
    _storedImagesCount = 0;
    _displayMode = false;
    pano_restart(_panoContext);
    cvEstimator_restart(_cvEstimator);
    _vcRotationFromSensorArray.clear();
    _vcRotationFromSensorArray.push_back(_rotationFromQuaternion3x3);
    _focal = 520;
    pano_setFocalLength(_panoContext, _focal);
    
    _closestFrames.clear();
    _closestFrames.push_back(std::vector<int>());
}

- (IBAction)displayResult:(id)sender {
    _displayMode = !_displayMode;
}

- (IBAction)startRefinedStitching:(id)sender {
    /*
    UIAlertView * alert = [[UIAlertView alloc] initWithTitle:@"Wait!" message:@"Under construction yet!" delegate:self cancelButtonTitle:@"Continue" otherButtonTitles:nil];
    [alert show];
    */
    
    if (_storedImagesCount > 2)
    {
        /*
        UIAlertView * alert = [[UIAlertView alloc] initWithTitle:@"Wait!" message:@"Refining results ..." delegate:self cancelButtonTitle:@"" otherButtonTitles:nil];
        [alert show];
        [alert performSelector:@selector(dismissWithClickedButtonIndex:animated:) withObject:[NSNumber numberWithInt:0] afterDelay:2];
         */
        //std::vector<std::vector <float>> currentRotationUsedToInitBA = pano_getCurrentRotations(_panoContext); // roatation strored in viewcontroller is more reliable than this
        _refinedRotations = cvEstimator_refinedStitching(_cvEstimator, _vcRotationFromSensorArray, _closestFrames);
        _focal = 0.5 * _focal + 0.5 * cvEstimator_getFocalLength(_cvEstimator);
        pano_setRefinedRotations(_panoContext, _refinedRotations, _focal);
    }
    else
    {
        UIAlertView * alert = [[UIAlertView alloc] initWithTitle:@"Error!" message:@"Capture at least 3 frames before refinement" delegate:self cancelButtonTitle:@"Continue" otherButtonTitles:nil];
        [alert show];
    }
    
}
@end
