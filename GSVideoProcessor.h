
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CMBufferQueue.h>

@protocol GSVideoProcessorDelegate;

@interface GSVideoProcessor : NSObject <AVCaptureAudioDataOutputSampleBufferDelegate, AVCaptureVideoDataOutputSampleBufferDelegate> 
{  
    id <GSVideoProcessorDelegate> __weak delegate;
	
	NSMutableArray *previousSecondTimestamps;
	Float64 videoFrameRate;
	CMVideoDimensions videoDimensions;
	CMVideoCodecType videoType;

	AVCaptureSession *captureSession;
	AVCaptureConnection *videoConnection;
	CMBufferQueueRef previewBufferQueue;
	
	AVAssetWriter *assetWriter;
	AVAssetWriterInput *assetWriterVideoIn;
	dispatch_queue_t movieWritingQueue;
    
	AVCaptureVideoOrientation referenceOrientation;
	AVCaptureVideoOrientation videoOrientation;
    CVOpenGLESTextureRef textureFromCamera;
    UIImage* frameFromCamera;


    
    
    
   BOOL readyToRecordVideo;
}

@property (readwrite, weak) id <GSVideoProcessorDelegate> delegate;
@property (readonly) Float64 videoFrameRate;
@property (readonly) CMVideoDimensions videoDimensions;
@property (readonly) CMVideoCodecType videoType;
@property (readonly) CVOpenGLESTextureRef textureFromCamera;
@property (readonly) UIImage* frameFromCamera;




@property (readwrite) AVCaptureVideoOrientation referenceOrientation;

- (CGAffineTransform)transformForOrientation:(AVCaptureVideoOrientation)orientation;


- (void) setupAndStartCaptureSession;
- (void) stopAndTearDownCaptureSession;

@end


@protocol GSVideoProcessorDelegate <NSObject>


- (CVOpenGLESTextureRef)pixelBufferReadyForDisplay:(CVPixelBufferRef)pixelBuffer;	// This method is always called on the main thread.

@end
