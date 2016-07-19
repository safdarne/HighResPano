#import <MobileCoreServices/MobileCoreServices.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import "GSVideoProcessor.h"

@interface GSVideoProcessor ()

// Redeclared as readwrite
@property (readwrite) Float64 videoFrameRate;
@property (readwrite) CMVideoDimensions videoDimensions;
@property (readwrite) CMVideoCodecType videoType;
@property (readwrite) AVCaptureVideoOrientation videoOrientation;
@property (readwrite) CVOpenGLESTextureRef textureFromCamera;
@property (readwrite) UIImage* frameFromCamera;



@end

@implementation GSVideoProcessor

@synthesize delegate;
@synthesize videoFrameRate, videoDimensions, videoType;
@synthesize referenceOrientation;
@synthesize videoOrientation;
@synthesize textureFromCamera;
@synthesize frameFromCamera;



/////////////////////////////////////////////////////////////////
//
- (id) init
{
    if (self = [super init])
    {
        previousSecondTimestamps = [[NSMutableArray alloc] init];
        referenceOrientation = AVCaptureVideoOrientationPortrait;
    }
    return self;
}


#pragma mark Utilities


/////////////////////////////////////////////////////////////////
//
- (CGFloat)angleOffsetFromPortrait:(AVCaptureVideoOrientation)orientation
{
    CGFloat angle = 0.0;
    
    switch (orientation)
    {
        case AVCaptureVideoOrientationPortrait:
            angle = 0.0;
            break;
        case AVCaptureVideoOrientationPortraitUpsideDown:
            angle = M_PI;
            break;
        case AVCaptureVideoOrientationLandscapeRight:
            angle = -M_PI_2;
            break;
        case AVCaptureVideoOrientationLandscapeLeft:
            angle = M_PI_2;
            break;
        default:
            break;
    }
    
    return angle;
}


/////////////////////////////////////////////////////////////////
//
- (CGAffineTransform)transformForOrientation:(AVCaptureVideoOrientation)orientation
{
    CGAffineTransform transform = CGAffineTransformIdentity;
    
    // Calculate offsets from an arbitrary reference orientation (portrait)
    CGFloat orientationAngleOffset =
    [self angleOffsetFromPortrait:orientation];
    CGFloat videoOrientationAngleOffset =
    [self angleOffsetFromPortrait:self.videoOrientation];
    
    // Find the difference in angle between the passed in orientation and the
    // current video orientation
    CGFloat angleOffset = orientationAngleOffset - videoOrientationAngleOffset;
    transform = CGAffineTransformMakeRotation(angleOffset);
    
    return transform;
}


#pragma mark Video Input

/////////////////////////////////////////////////////////////////
//
- (BOOL)setupAssetWriterVideoInput:(CMFormatDescriptionRef)currentFormatDescription
{
    float bitsPerPixel;
    CMVideoDimensions dimensions =
    CMVideoFormatDescriptionGetDimensions(currentFormatDescription);
    int numPixels = dimensions.width * dimensions.height;
    int bitsPerSecond;
    
    // Assume that lower-than-SD resolutions are intended for streaming, and use
    // a lower bitrate
    if ( numPixels < (640 * 480) )
    {
        bitsPerPixel = 4.05; // matches quality of AVCaptureSessionPresetMedium.
    }
    else
    {
        bitsPerPixel = 11.4; // matches quality of AVCaptureSessionPresetHigh.
    }
    
    bitsPerSecond = numPixels * bitsPerPixel;
    
    NSDictionary *videoCompressionSettings =
    @{AVVideoCodecKey: AVVideoCodecH264,
      AVVideoWidthKey: @(dimensions.width),
      AVVideoHeightKey: @(dimensions.height),
      AVVideoCompressionPropertiesKey: @{AVVideoAverageBitRateKey:
                                             @(bitsPerSecond),
                                         AVVideoMaxKeyFrameIntervalKey: @30}};
    
    if ([assetWriter canApplyOutputSettings:videoCompressionSettings
                               forMediaType:AVMediaTypeVideo])
    {
        assetWriterVideoIn = [[AVAssetWriterInput alloc]
                              initWithMediaType:AVMediaTypeVideo
                              outputSettings:videoCompressionSettings];
        assetWriterVideoIn.expectsMediaDataInRealTime = YES;
        assetWriterVideoIn.transform =
        [self transformForOrientation:self.referenceOrientation];
        
        if ([assetWriter canAddInput:assetWriterVideoIn])
        {
            [assetWriter addInput:assetWriterVideoIn];
        }
        else
        {
            NSLog(@"Couldn't add asset writer video input.");
            return NO;
        }
    }
    else
    {
        NSLog(@"Couldn't apply video output settings.");
        return NO;
    }
    
    return YES;
}


#pragma mark Capture

/////////////////////////////////////////////////////////////////
//
- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    
    CMFormatDescriptionRef formatDescription =
    CMSampleBufferGetFormatDescription(sampleBuffer);
    /*
     if ( connection == videoConnection )
     {
     
     // Get frame dimensions (for onscreen display)
     if (self.videoDimensions.width == 0 && self.videoDimensions.height == 0)
     {
     self.videoDimensions =
     CMVideoFormatDescriptionGetDimensions( formatDescription );
     }
     
     // Get buffer type
     if ( self.videoType == 0 )
     {
     self.videoType =
     CMFormatDescriptionGetMediaSubType( formatDescription );
     }
     
     // Enqueue it for preview.  This is a shallow queue, so if image
     // processing is taking too long, we'll drop this frame for preview (this
     // keeps preview latency low).
     OSStatus err = CMBufferQueueEnqueue(previewBufferQueue, sampleBuffer);
     if ( !err ) {
     dispatch_async(dispatch_get_main_queue(), ^{
     CMSampleBufferRef sbuf =
     (CMSampleBufferRef)CMBufferQueueDequeueAndRetain(
     previewBufferQueue);
     
     if (sbuf)
     {
					CVImageBufferRef pixBuf = CMSampleBufferGetImageBuffer(sbuf);
					textureFromCamera = [self.delegate pixelBufferReadyForDisplay:pixBuf];
					
     
     CFRelease(sbuf);
     
     
     }
     });
     }
     }
     
     CFRetain(sampleBuffer);
     CFRetain(formatDescription);
     dispatch_async(movieWritingQueue,
     ^{
     if ( assetWriter )
     {
     if (connection == videoConnection)
     {
     // Initialize the video input if this is not done yet
     if (!readyToRecordVideo)
     {
					readyToRecordVideo =
     [self setupAssetWriterVideoInput:formatDescription];
     }
     }
     }
     
     CFRelease(sampleBuffer);
     CFRelease(formatDescription);
     });
     
     */
    
    frameFromCamera = [self imageFromSampleBuffer:sampleBuffer];
    
    
}
/////////////////////////////////////////////////////////////////
//
// Create a UIImage from sample buffer data
- (UIImage *) imageFromSampleBuffer:(CMSampleBufferRef) sampleBuffer
{
    // Get a CMSampleBuffer's Core Video image buffer for the media data
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    // Lock the base address of the pixel buffer
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // Get the number of bytes per row for the pixel buffer
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    // Get the number of bytes per row for the pixel buffer
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    // Get the pixel buffer width and height
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    // Create a device-dependent RGB color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // Create a bitmap graphics context with the sample buffer data
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                 bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    // Create a Quartz image from the pixel data in the bitmap graphics context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    // Unlock the pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    // Free up the context and color space
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // Create an image object from the Quartz image
    UIImage *image = [UIImage imageWithCGImage:quartzImage] ;//] scale:1 orientation:UIImageOrientationRight];  ////fixme
    
    // Release the Quartz image
    CGImageRelease(quartzImage);
    
    return (image);
}

/////////////////////////////////////////////////////////////////
//
- (AVCaptureDevice *)videoDeviceWithPosition:(AVCaptureDevicePosition)position
{
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in devices)
        if ([device position] == position)
            return device;
    
    return nil;
}


/////////////////////////////////////////////////////////////////
//
- (BOOL) setupCaptureSession
{

    captureSession = [[AVCaptureSession alloc] init];
   
    if ([captureSession canSetSessionPreset:AVCaptureSessionPreset640x480]) {
        captureSession.sessionPreset = AVCaptureSessionPreset640x480;
    }
    /*
    if ([captureSession canSetSessionPreset:AVCaptureSessionPreset1280x720]) {
        captureSession.sessionPreset = AVCaptureSessionPreset1280x720;
    }
    */
    [captureSession beginConfiguration];
    NSArray *devices = [AVCaptureDevice devices];
    NSError *error;
    for (AVCaptureDevice *device in devices) {
        if (([device hasMediaType:AVMediaTypeVideo]) &&
            ([device position] == AVCaptureDevicePositionBack) ) {
            [device lockForConfiguration:&error];
            if ([device isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeLocked]) {
                device.whiteBalanceMode = AVCaptureWhiteBalanceModeLocked;
                NSLog(@"Whitebalanced locked");
            }
            if ([device isExposureModeSupported:AVCaptureExposureModeLocked]) {
                device.exposureMode = AVCaptureExposureModeLocked;
                NSLog(@"Exposure locked");
            }
            /*
            if ([device isFocusModeSupported:AVCaptureFocusModeLocked]) {
                device.focusMode = AVCaptureFocusModeLocked;
                NSLog(@"Focus locked");
            }
            */
            [device unlockForConfiguration];
            
            
            CGFloat desiredFPS = 10;
            
            AVCaptureDeviceFormat *selectedFormat = nil;
            int32_t maxWidth = 0;
            AVFrameRateRange *frameRateRange = nil;
            
            for (AVCaptureDeviceFormat *format in [device formats]) {
                
                for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
                    
                    CMFormatDescriptionRef desc = format.formatDescription;
                    CMVideoDimensions dimensions = CMVideoFormatDescriptionGetDimensions(desc);
                    int32_t width = dimensions.width;
                    
                    if (range.minFrameRate <= desiredFPS && desiredFPS <= range.maxFrameRate && width >= maxWidth) {
                        
                        selectedFormat = format;
                        frameRateRange = range;
                        maxWidth = width;
                    }
                }
            }
            
            
            
            if (selectedFormat) {
                if ([device lockForConfiguration:nil]) {
                    
                    device.activeVideoMinFrameDuration = CMTimeMake(1, (int32_t)desiredFPS);
                    device.activeVideoMaxFrameDuration = CMTimeMake(1, (int32_t)desiredFPS);
                    [device unlockForConfiguration];
                }
            }
            
        }
    }
    [captureSession commitConfiguration];
    
    
    AVCaptureDeviceInput *videoIn = [[AVCaptureDeviceInput alloc]
                                     initWithDevice:[self videoDeviceWithPosition:AVCaptureDevicePositionBack]
                                     error:nil];
    
    
    
    if ([captureSession canAddInput:videoIn])
    {
        [captureSession addInput:videoIn];
    }
    
    AVCaptureVideoDataOutput *videoOut = [[AVCaptureVideoDataOutput alloc] init];
    /*
     Processing can take longer than real-time on some platforms.
     Clients whose image processing is faster than real-time should consider
     setting AVCaptureVideoDataOutput's alwaysDiscardsLateVideoFrames property
     to NO.
     */
    
    
    [captureSession addOutput:videoOut]; //// Mori
    
    
    
    [videoOut setAlwaysDiscardsLateVideoFrames:YES];
    [videoOut setVideoSettings:
     @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)}];
    dispatch_queue_t videoCaptureQueue =
    dispatch_queue_create("Video Capture Queue", DISPATCH_QUEUE_SERIAL);
    [videoOut setSampleBufferDelegate:self queue:videoCaptureQueue];
    //dispatch_release(videoCaptureQueue);
    if ([captureSession canAddOutput:videoOut])
        [captureSession addOutput:videoOut];
    videoConnection = [videoOut connectionWithMediaType:AVMediaTypeVideo];
    
     ////
     if ([videoConnection isVideoOrientationSupported])
     {
     [videoConnection setVideoOrientation:AVCaptureVideoOrientationPortrait];
     }
    
    

     ////
    
     self.videoOrientation = [videoConnection videoOrientation];
    
    
    
    // Configure your output.
    dispatch_queue_t queue = dispatch_queue_create("myQueue", NULL);
    [videoOut setSampleBufferDelegate:self queue:queue];
    
    // Specify the pixel format
    videoOut.videoSettings =
    [NSDictionary dictionaryWithObject:
     [NSNumber numberWithInt:kCVPixelFormatType_32BGRA]
                                forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    
    
    // If you wish to cap the frame rate to a known value, such as 15 fps, set
    // minFrameDuration.
    //videoOut.minFrameDuration = CMTimeMake(1, 15);
    
    
    // Start the session running to start the flow of data
    [captureSession startRunning];
    
    // Assign session to an ivar.
    //// [self setSession:captureSession];
    
    
    
    return YES;
}


/////////////////////////////////////////////////////////////////
//
- (void) setupAndStartCaptureSession
{
    // Create a shallow queue for buffers going to the display for preview.
    OSStatus err = CMBufferQueueCreate(
                                       kCFAllocatorDefault,
                                       1,
                                       CMBufferQueueGetCallbacksForUnsortedSampleBuffers(),
                                       &previewBufferQueue);
    
    
    
    // Create serial queue for movie writing
    movieWritingQueue =
    dispatch_queue_create("Movie Writing Queue", DISPATCH_QUEUE_SERIAL);
    
    if ( !captureSession )
    {
        [self setupCaptureSession];
    }
    
    [[NSNotificationCenter defaultCenter]
     addObserver:self
     selector:@selector(captureSessionStoppedRunningNotification:)
     name:AVCaptureSessionDidStopRunningNotification
     object:captureSession];
    
    if ( !captureSession.isRunning )
    {
        [captureSession startRunning];
    }
}


/////////////////////////////////////////////////////////////////
//
- (void)captureSessionStoppedRunningNotification:(NSNotification *)notification
{
    dispatch_async(movieWritingQueue, ^{
    });
}


/////////////////////////////////////////////////////////////////
//
- (void) stopAndTearDownCaptureSession
{
    [captureSession stopRunning];
    if (captureSession)
    {
        [[NSNotificationCenter defaultCenter]
         removeObserver:self
         name:AVCaptureSessionDidStopRunningNotification
         object:captureSession];
    }
    
    captureSession = nil;
    if (previewBufferQueue)
    {
        CFRelease(previewBufferQueue);
        previewBufferQueue = NULL;	
    }
    
    if (movieWritingQueue)
    {
        //dispatch_release(movieWritingQueue);
        movieWritingQueue = NULL;
    }
}


#pragma mark Error Handling

/////////////////////////////////////////////////////////////////






@end
