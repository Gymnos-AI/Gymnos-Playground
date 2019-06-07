camera_type = 'picamera'

# import the necessary packages
try:
    import PiCameraMain as piCam
except ImportError:
    print('No PiCamera available, setting as usb device')
    camera_type = 'usb'
    import GymnosCamera.UsbCameraMain as usbCam

import argparse

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'


# Run loop of either PiCamera or USB
if camera_type == 'picamera':
    piCam = piCam.PiCameraMain()

    piCam.run_loop()
else:
    usbCam = usbCam.UsbCameraMain()
    usbCam.run_loop()