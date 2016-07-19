#import <UIKit/UIKit.h>
#import <GLKit/GLKit.h>
#import <AVFoundation/AVFoundation.h>
#import <MediaPlayer/MediaPlayer.h>
#import <MobileCoreServices/MobileCoreServices.h>
#import <CoreMotion/CoreMotion.h>
#import <AudioToolbox/AudioToolbox.h>


@interface ViewController : GLKViewController<UIImagePickerControllerDelegate,UINavigationControllerDelegate,UIAccelerometerDelegate >
{
    CMAttitude *_refAttitude;
    CMAttitude *_currentAttitude;
    CMAttitude *_attitudeChange;
    CMQuaternion _quaternionChange;
    GLKMatrix4 _rotationFromQuaternion;
    CVOpenGLESTextureRef _textureFromCamera;
    UIImage* _frameFromCamera;
    GLKTextureInfo *info;
    UIAccelerometer *_accelerometer;
}

@property (strong, nonatomic) CMMotionManager *motionManager;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UITextField *rollLabel;
@property (weak, nonatomic) IBOutlet UITextField *pitchLabel;
@property (weak, nonatomic) IBOutlet UITextField *yawLabel;
@property (weak, nonatomic) IBOutlet UIButton *refinementButton;
@property (weak, nonatomic) IBOutlet UILabel *flagLabel;

- (IBAction)cvProcessFrames:(id)sender;
- (IBAction)warpToSphere:(id)sender;
- (IBAction)stopCapturing:(id)sender;
- (IBAction)startCapturing:(id)sender;
- (IBAction)displayResult:(id)sender;
- (IBAction)startRefinedStitching:(id)sender;
@end

