from cam_setup import CameraDevice
import depthai as dai
import cv2

device = dai.Device()
calibData = device.readCalibration()

camSetup = CameraDevice()
camSetup.fix_focus(calibData)
camSetup.linking()
pipeline = camSetup.get_pipeline()

with device:
    device.startPipeline(pipeline)

    rgbWindowName = "rgb"
    depthWindowName = "depth"
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(rgbWindowName)
    cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
