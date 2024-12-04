from cam_setup import CameraDevice
import depthai as dai
import cv2
import numpy as np
import utils

device = dai.Device()
calibData = device.readCalibration()

camSetup = CameraDevice()
camSetup.fix_focus(calibData)
camSetup.linking()
pipeline = camSetup.get_pipeline()

color = (255, 255, 255)

projected_calib_image = cv2.imread('./image/red.jpg')
if projected_calib_image is None:
    print("Failed to load image. Check the path.")

# Create a named window
window_name = "Second Screen Fullscreen Display"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Move the window to the second screen
cv2.moveWindow(window_name, 1920, 0)

# Set the window to fullscreen
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Show the image
cv2.imshow(window_name, projected_calib_image)

with device:
    device.startPipeline(pipeline)

    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    qRgb = device.getOutputQueue(name="rgb_stream", maxSize=4, blocking=False)

    while True:
        inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame()  # depthFrame values are in millimeters

        rgbData = qRgb.get()
        rgbFrame = rgbData.getCvFrame()

        corners = utils.detectCorners(rgbFrame)
        if corners:
            # Draw the rectangle on the image for visualization
            for point in corners:
                cv2.circle(rgbFrame, tuple(point), 5, (0, 255, 0), -1)

        cv2.imshow("rgb", rgbFrame)


        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        spatialData = spatialCalcQueue.get().getSpatialLocations()
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                        fontType, 0.5, color)

            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                        fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                        fontType, 0.5, color)
            # Show the frame
        cv2.imshow("depth", depthFrameColor)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cv2.destroyAllWindows()