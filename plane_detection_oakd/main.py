import depthai as dai
import utils
from pathlib import Path
import time
import argparse
import cv2

nnPathDefault = str((Path(__file__).parent / Path('./models/mobilenet-ssd.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
nnNetworkOut.setStreamName("nnNetwork");

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setFps(40)
# Define a neural network that will make predictions based on the source frames
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(args.nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Linking
if args.sync:
    nn.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)
nn.outNetwork.link(nnNetworkOut.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qNN = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    printOutputLayersOnce = True

    while True:
        if args.sync:
            # Use blocking get() call to catch frame and inference result synced
            inRgb = qRgb.get()
            inDet = qDet.get()
            inNN = qNN.get()
        else:
            # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()
            inNN = qNN.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if printOutputLayersOnce and inNN is not None:
            toPrint = 'Output layer names:'
            for ten in inNN.getAllLayerNames():
                toPrint = f'{toPrint} {ten},'
            print(toPrint)
            printOutputLayersOnce = False;

        # If the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:
            utils.displayFrame("rgb", frame, detections)

        if cv2.waitKey(1) == ord('q'):
            break