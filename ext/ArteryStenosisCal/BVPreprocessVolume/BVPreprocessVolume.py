import logging
from typing import Optional

import numpy as np
import slicer
import slicer.cli
import slicer.util
import vtk
from BVPreprocessVolumeLib.BVPVConstants import BVTextConst
from slicer import (
    vtkMRMLMarkupsROINode,
    vtkMRMLNode,
    vtkMRMLScalarVolumeNode,
)
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

#
# BVPreprocessVolume
#


class BVPreprocessVolume(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.version = 3.0
        self.parent.title = 'BV Preprocess (CT) Volume'
        self.parent.categories = ['Chula BV']
        self.parent.dependencies = []
        self.parent.contributors = ['Krit Cholapand (Chulalongkorn University)']
        self.parent.helpText = f"""
    Artery Stenosis Measurement version {self.version}. Documentation is available
    <a href="https://github.com/flapperz/artery-stenosis">here</a>.
    """
        self.parent.acknowledgementText = """TODO: ACKNOWLEDGEMENT"""


#
# BVPreprocessVolumeParameterNode
#


@parameterNodeWrapper
class BVPreprocessVolumeParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    heartRoi: vtkMRMLMarkupsROINode
    costVolume: vtkMRMLScalarVolumeNode

    # No widget counterpart
    sdfVolume: vtkMRMLScalarVolumeNode


#
# BVPreprocessVolumeWidget
#


class BVPreprocessVolumeWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/BVPreprocessVolume.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BVPreprocessVolumeLogic()

        # Extra Widget Setup
        self.ui.costVolumeSelector.baseName = BVTextConst.costVolumePrefix

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # input is validated via _checkCanCreateHeartRoi
        self.ui.heartROISelector.connect(
            'nodeAddedByUser(vtkMRMLNode*)', self.fitInputHeartRoiNodeToVolume
        )
        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.nextStepButton.connect('clicked(bool)', self.onNextStepButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

        # Update roi widget and refresh parameter node
        self._onParameterUpdate()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onParameterUpdate,
            )
            # clean up each node interactive observer before exit
            self._cleanUpInputNodeObserver()

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass(
                'vtkMRMLScalarVolumeNode'
            )
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(
        self, inputParameterNode: Optional[BVPreprocessVolumeParameterNode]
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onParameterUpdate,
            )
            self._cleanUpInputNodeObserver()
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onParameterUpdate,
            )

    def fitInputHeartRoiNodeToVolume(self):
        if (
            self._parameterNode
            and self._parameterNode.inputVolume
            and self._parameterNode.heartRoi
        ):
            self.logic.fitHeartRoiNode(
                self._parameterNode.inputVolume, self._parameterNode.heartRoi
            )

    def _onParameterUpdate(self, caller=None, event=None) -> None:
        self._checkCanApply(caller=caller, event=event)
        self._setROIWidget(caller=caller, event=event)
        self._checkCanCreateHeartRoi(caller=caller, event=event)
        # self._refreshInputNodeObserver(
        #     self._parameterNode.heartRoi,
        #     vtkMRMLMarkupsNode.PointModifiedEvent,
        #     self.onHeartROIUpdate,
        # )

    def _cleanUpInputNodeObserver(self):
        # Should mirror all _refreshInputNodeObserver call
        # self._removeInputNodeObserver(
        #     vtkMRMLMarkupsNode.PointModifiedEvent,
        #     self.onHeartROIUpdate
        # )
        pass

    def _checkCanCreateHeartRoi(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume:
            self.ui.heartROISelector.addEnabled = True
        else:
            self.ui.heartROISelector.addEnabled = False

    def _setROIWidget(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.heartRoi:
            self.ui.MRMLMarkupsROIWidget.setMRMLMarkupsNode(
                self._parameterNode.heartRoi
            )
        else:
            self.ui.MRMLMarkupsROIWidget.setMRMLMarkupsNode(None)

    def _checkCanApply(self, caller=None, event=None) -> None:
        if (
            self._parameterNode
            and self._parameterNode.inputVolume
            and self._parameterNode.heartRoi
        ):
            self.ui.applyButton.toolTip = 'Compute preprocessed volume as cost volume'
            self.ui.applyButton.enabled = True
            return
        self.ui.applyButton.toolTip = (
            'Select input volume, heart ROI and output volume nodes'
        )
        self.ui.applyButton.enabled = False

    def _refreshInputNodeObserver(
        self, node: Optional[vtkMRMLNode], eventList, callback
    ):
        # https://apidocs.slicer.org/master/classvtkMRMLMarkupsNode.html#aceeef8806df28e3807988c38510e56caad628dfaf54f410d2f4c6bc5800aa8a30
        if not isinstance(eventList, list):
            eventList = [eventList]

        self._removeInputNodeObserver(eventList, callback)

        for e in eventList:
            if node is not None:
                print(
                    f'{self.moduleName} ADD observer from:',
                    node.GetName(),
                    ' : ',
                    node.GetID(),
                )
                self.addObserver(node, e, callback)

    def _removeInputNodeObserver(self, eventList, callback):
        if not isinstance(eventList, list):
            eventList = [eventList]

        for e in eventList:
            prevNode = self.observer(e, callback)
            if prevNode is not None:
                logging.debug(
                    f'{self.moduleName} Remove observer from:',
                    prevNode.GetName(),
                    ':',
                    prevNode.GetID(),
                )
                # prevMarkersNodeName = prevMarkersNode.GetName()
                self.removeObserver(prevNode, e, callback)

    # def onHeartROIUpdate(self, caller=None, event=None):
    #     if (self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.heartRoi):
    #         canApply = self.logic.validateHeartROI(self._parameterNode.inputVolume, self._parameterNode.heartRoi)
    #         print('on heart roi update:', canApply)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(
            'Failed to compute results.', waitCursor=True
        ):
            inputVolume = self._parameterNode.inputVolume
            heartRoi = self._parameterNode.heartRoi

            if not self._parameterNode.costVolume:
                self._parameterNode.costVolume = self.logic.createEmptyOutputVolume()
            costVolume = self._parameterNode.costVolume

            if not self._parameterNode.sdfVolume:
                self._parameterNode.sdfVolume = self.logic.createEmptyOutputVolume()
                self._parameterNode.sdfVolume.SetName("BV_SDFVOLUME")
            sdfVolume = self._parameterNode.sdfVolume

            if self.logic.validateHeartROI(inputVolume, heartRoi):
                self.logic.process(inputVolume, heartRoi, costVolume, sdfVolume)
                # self.logic.renderDebugCroppedVolume(inputVolume, heartRoi, costVolume)

                self.ui.MRMLMarkupsROIWidget.setInteractiveMode(False)
            else:
                e = 'ROI is bigger than input volume or too big (15mm * 15mm * 15mm)'
                slicer.util.errorDisplay('Failed to compute results: ' + str(e))

    def onNextStepButton(self) -> None:
        mainWindow = slicer.util.mainWindow()
        mainWindow.moduleSelector().selectModule('BVStenosisMeasurement')



#
# BVPreprocessVolumeLogic
#


class BVPreprocessVolumeLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return BVPreprocessVolumeParameterNode(super().getParameterNode())

    def fitHeartRoiNode(
        self, volumeNode: vtkMRMLScalarVolumeNode, roiNode: vtkMRMLMarkupsROINode
    ) -> None:
        # roiNode.GetDisplayNode().SetFillVisibility(False)

        # fit roi
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLCropVolumeParametersNode'
        )
        cropVolumeParameters.SetInputVolumeNodeID(volumeNode.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(
            cropVolumeParameters
        )  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

    def validateHeartROI(
        self, volumeNode: vtkMRMLScalarVolumeNode, roiNode: vtkMRMLMarkupsROINode
    ) -> bool:
        roiBound = np.zeros(6)
        volumeBound = np.zeros(6)
        roiNode.GetRASBounds(roiBound)
        volumeNode.GetRASBounds(volumeBound)
        minIdx = [0, 2, 4]
        maxIdx = [1, 3, 5]
        isBound = np.all(roiBound[minIdx] > volumeBound[minIdx]) and np.all(
            roiBound[maxIdx] < volumeBound[maxIdx]
        )

        sizesMM = roiBound[maxIdx] - roiBound[minIdx]
        maxSizesMM = np.array([150, 150, 150])
        # roiVolumeMM3 = sizesMM[0] * sizesMM[1] * sizesMM[2]
        # logging.debug(f"validate roi: roi volume (mm^3)= {roiVolumeMM3}")
        isNotTooBig = np.all(sizesMM < maxSizesMM)
        logging.debug(f'{isBound=} {isNotTooBig=} {sizesMM=}')
        result = isBound and isNotTooBig

        return result

    def renderDebugCroppedVolume(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        heartRoi: vtkMRMLMarkupsROINode,
        costVolume: vtkMRMLScalarVolumeNode,
    ):
        volRenLogic = slicer.modules.volumerendering.logic()
        preset = volRenLogic.GetPresetByName('CT-Cardiac3')
        displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(inputVolume)
        displayNode.GetVolumePropertyNode().Copy(preset)
        displayNode.SetAndObserveROINodeID(heartRoi.GetID())
        displayNode.CroppingEnabledOn()
        displayNode.SetVisibility(True)

    def createEmptyOutputVolume(self):
        costVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        costVolume.SetName(BVTextConst.costVolumePrefix)
        costVolume.CreateDefaultDisplayNodes()
        costVolume.CreateDefaultStorageNode()
        return costVolume

    def doCrop(self, inputVolume, heartRoi, costVolume, scaling):
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLCropVolumeParametersNode'
        )
        cropVolumeParameters.SetInputVolumeNodeID(inputVolume.GetID())
        cropVolumeParameters.SetROINodeID(heartRoi.GetID())
        cropVolumeParameters.SetOutputVolumeNodeID(costVolume.GetID())
        cropVolumeParameters.SetInterpolationMode(
            cropVolumeParameters.InterpolationLinear
        )
        # Interpolated Cropping
        cropVolumeParameters.SetVoxelBased(False)
        cropVolumeParameters.SetSpacingScalingConst(scaling)
        cropVolumeParameters.SetIsotropicResampling(True)
        slicer.modules.cropvolume.logic().Apply(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

    def copyVolume(self, inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode, name: str) -> vtkMRMLScalarVolumeNode:

        if outputVolume is None:
            outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', name)

        imageDimensions = inputVolume.GetImageData().GetDimensions()
        voxelType = vtk.VTK_DOUBLE
        # voxelType = inputVolume.GetImageData().GetScalarType() # dicom is vtk.VTK_INT
        imageOrigin = inputVolume.GetImageData().GetOrigin()
        imageSpacing = inputVolume.GetImageData().GetSpacing()
        fillVoxelValue = 0
        volumeToRAS = inputVolume.GetImageData().GetDirectionMatrix()
        imageDirections = slicer.util.arrayFromVTKMatrix(volumeToRAS)

        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageDimensions)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)

        outputVolume.SetOrigin(imageOrigin)
        outputVolume.SetSpacing(imageSpacing)
        outputVolume.SetIJKToRASDirections(imageDirections)
        outputVolume.SetAndObserveImageData(imageData)
        outputVolume.CreateDefaultDisplayNodes()
        outputVolume.CreateDefaultStorageNode()

    def process(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        heartRoi: vtkMRMLMarkupsROINode,
        costVolume: vtkMRMLScalarVolumeNode,
        sdfVolume: vtkMRMLScalarVolumeNode
    ) -> None:
        if not (inputVolume and heartRoi and costVolume and sdfVolume):
            raise ValueError('Input or output volume is invalid')

        import time

        startTime = time.time()
        logging.info('Processing started')

        heartRoi.GetDisplayNode().SetFillVisibility(False)

        self.doCrop(inputVolume, heartRoi, costVolume, scaling=0.6)

        # window sinc cropping
        # TODO test with sinc

        import SimpleITK as sitk
        import sitkUtils as su

        histogramFilter = sitk.AdaptiveHistogramEqualizationImageFilter()
        # 0.3 provide faster guideline creation than 0.5
        histogramFilter.SetAlpha(0.3)
        histogramFilter.SetBeta(0.3)
        histogramFilter.SetRadius([3,3,3])

        costSITKImage = su.PullVolumeFromSlicer(costVolume)
        costSITKImage = histogramFilter.Execute(costSITKImage)
        su.PushVolumeToSlicer(costSITKImage, costVolume)

        sourceArray = slicer.util.arrayFromVolume(costVolume).copy()
        sourceArray[sourceArray < -1000] = -1000
        sourceArray += 1000
        sourceArray[sourceArray < 450] = 100
        slicer.util.updateVolumeFromArray(costVolume, sourceArray)

        #
        # end preprocess
        #

        # set slice viewer back to patient
        slicer.util.setSliceViewerLayers(background=inputVolume, foreground=costVolume)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def process_old(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        heartRoi: vtkMRMLMarkupsROINode,
        costVolume: vtkMRMLScalarVolumeNode,
        sdfVolume: vtkMRMLScalarVolumeNode
    ) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not (inputVolume and heartRoi and costVolume and sdfVolume):
            raise ValueError('Input or output volume is invalid')

        import time

        startTime = time.time()
        logging.info('Processing started')

        heartRoi.GetDisplayNode().SetFillVisibility(False)

        self.doCrop(inputVolume, heartRoi, costVolume, scaling=1)

        # TODO may be we can use only one array here
        sourceArray = slicer.util.arrayFromVolume(costVolume).copy()
        bufferArray = sourceArray.copy()


        #
        # Create SDF
        #

        import SimpleITK as sitk
        import sitkUtils as su


        # TODO: work for guideline but cavity mark leak in to rca
        SDF_MIN_INSIDE = -300
        SDF_MIN_SQUARE_VALUE = 30
        CAVITY_DILATE_STEP = 6

        bufferArray[sourceArray > SDF_MIN_INSIDE] = -1
        bufferArray[sourceArray <= SDF_MIN_INSIDE] = 0

        sdfFilter = sitk.SignedMaurerDistanceMapImageFilter()
        sdfFilter.SetBackgroundValue(0.0)
        sdfFilter.SetDebug(False)
        sdfFilter.SetInsideIsPositive(False)
        sdfFilter.SetNumberOfThreads(8)
        sdfFilter.SetNumberOfWorkUnits(0)
        sdfFilter.SetSquaredDistance(True)
        sdfFilter.SetUseImageSpacing(False)

        slicer.util.updateVolumeFromArray(costVolume, bufferArray)
        costSITKImage = su.PullVolumeFromSlicer(costVolume)
        sdfSITKImage = sdfFilter.Execute(costSITKImage)
        # sitk.GetArrayFromImage(image)
        su.PushVolumeToSlicer(sdfSITKImage, sdfVolume)
        slicer.util.updateVolumeFromArray(costVolume, sourceArray)

        #
        # Drill Plaque
        #

        grindPeakModule = slicer.modules.grayscalegrindpeakimagefilter
        grindPeakParameter = {
            'inputVolume': costVolume,
            'outputVolume': costVolume,
        }
        grindPeakCliNode = slicer.cli.createNode(grindPeakModule, grindPeakParameter)
        slicer.cli.runSync(grindPeakModule, grindPeakCliNode)
        bufferArray = slicer.util.arrayFromVolume(costVolume)
        # peak diff
        bufferArray = sourceArray - bufferArray
        # TODO: pass through fornow change to adaptive depend on plaque (may be sdf) or erode diff
        PLAQUE_DIFF_MIN = 50
        PLAQUE_DIFF_MAX = 340

        sourceArray[(bufferArray > PLAQUE_DIFF_MIN) & (bufferArray < PLAQUE_DIFF_MAX)] = -200

        #
        # Intensity
        #

        bufferArray = sourceArray.copy()
        AIR_THRESHOLD = -150

        logging.debug(f'Type of output {costVolume.GetImageData().GetScalarTypeAsString()} -py-> {bufferArray.dtype}')

        # apply polynomial
        coef = [-3.912e-1, 2.689e0, 5.395e-3, 3.706e-6]
        bufferArray = np.polynomial.polynomial.polyval(
            bufferArray, coef
        )
        bufferArray = bufferArray.astype(np.int32)
        AIR_THRESHOLD = np.polynomial.polynomial.polyval(-150, coef)
        print(f"{AIR_THRESHOLD=}")

        CUTOFF_THRESHOLD = -500
        bufferArray[bufferArray < CUTOFF_THRESHOLD] = CUTOFF_THRESHOLD
        bufferArray[bufferArray > AIR_THRESHOLD] = 4000000
        bufferArray += -CUTOFF_THRESHOLD + 100

        # threshold
        # COST_OFFSET = 4000
        # bufferArray[bufferArray > AIR_THRESHOLD] = 4000000
        # bufferArray[bufferArray <= AIR_THRESHOLD] += COST_OFFSET
        # bufferArray += COST_OFFSET

        logging.debug(f'Type of output {costVolume.GetImageData().GetScalarTypeAsString()} -py-> {bufferArray.dtype}')

        sourceArray = bufferArray.copy()

        #
        # Fill Cavity
        #

        import scipy

        structure = np.zeros((3,3,3))
        structure[:,1,1] = 1
        structure[1,:,1] = 1
        structure[1,1,:] = 1


        bufferArray = slicer.util.arrayFromVolume(sdfVolume).copy()

        bufferArray = bufferArray >= SDF_MIN_SQUARE_VALUE
        if CAVITY_DILATE_STEP:
            bufferArray = scipy.ndimage.binary_dilation(bufferArray, structure, 6)

        sourceArray[bufferArray] = 20000000

        slicer.util.updateVolumeFromArray(costVolume, sourceArray)

        #
        # end preprocess
        #

        # set slice viewer back to patient
        slicer.util.setSliceViewerLayers(background=inputVolume, foreground=None)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

#
# BVStenosisMeasurementTest
#
# --Test--
#

class BVPreprocessVolumeTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_CreateDefaultScene()

    def test_CreateDefaultScene(self):


        slicer.util.loadScene(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/2023-10-26-Scene.mrb'
        )
        slicer.util.loadMarkups(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/markups/LCX-D.mrk.json'
        )
        slicer.util.loadMarkups(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/markups/RCA-D.mrk.json'
        )
        roiNode = slicer.util.loadNodeFromFile(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/R.mrk.json'
        )
        volumeNode = slicer.util.getNode('14: Body Soft Tissue')

        widget = (
            slicer.modules.bvpreprocessvolume.widgetRepresentation().self()
        )
        widget.ui.inputVolumeSelector.setCurrentNode(volumeNode)
        widget.ui.heartROISelector.setCurrentNode(roiNode)
        widget.onApplyButton()

        mainWindow = slicer.util.mainWindow()
        mainWindow.moduleSelector().selectModule('BVStenosisMeasurement')
