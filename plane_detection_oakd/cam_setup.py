import depthai as dai

class CameraDevice:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        # Define sources and outputs
        self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
        self.monoRight = self.pipeline.create(dai.node.MonoCamera)
        self.rgb = self.pipeline.create(dai.node.ColorCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

        self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        self.xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
        self.xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)
        self.rgbOut = self.pipeline.create(dai.node.XLinkOut)

        self.xoutDepth.setStreamName("depth")
        self.xoutSpatialData.setStreamName("spatialData")
        self.xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
        self.rgbOut.setStreamName("rgb_stream")

        # Properties
        self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoLeft.setCamera("left")
        self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoRight.setCamera("right")

        self.rgbCamSocket = dai.CameraBoardSocket.CAM_A
        self.rgb.setBoardSocket(self.rgbCamSocket)
        self.rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.rgb.setFps(30)

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)
        self.stereo.setDepthAlign(self.rgbCamSocket)

        # ROI initial values
        self.topLeft = dai.Point2f(0.4, 0.4)
        self.bottomRight = dai.Point2f(0.6, 0.6)

        self.config = dai.SpatialLocationCalculatorConfigData()
        self.config.depthThresholds.lowerThreshold = 100
        self.config.depthThresholds.upperThreshold = 10000
        self.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
        self.config.roi = dai.Rect(self.topLeft, self.bottomRight)

        self.spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        self.spatialLocationCalculator.initialConfig.addROI(self.config)

    def fix_focus(self, calibData):
        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        try:
            lensPosition = calibData.getLensPosition(self.rgbCamSocket)
            if lensPosition:
                self.rgb.initialControl.setManualFocus(lensPosition)
        except:
            raise

    def get_pipeline(self):
        return self.pipeline

    def linking(self):
        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)
        self.rgb.video.link(self.rgbOut.input)

        self.spatialLocationCalculator.passthroughDepth.link(self.xoutDepth.input)
        self.stereo.depth.link(self.spatialLocationCalculator.inputDepth)

        self.spatialLocationCalculator.out.link(self.xoutSpatialData.input)
        self.xinSpatialCalcConfig.out.link(self.spatialLocationCalculator.inputConfig)

