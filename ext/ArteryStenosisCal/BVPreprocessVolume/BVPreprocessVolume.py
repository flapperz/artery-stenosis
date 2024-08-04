import logging
import os
from typing import Optional

import numpy as np
import slicer
import vtk
from BVPreprocessVolumeLib.Constants import BVTextConst
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
        self.parent.dependencies = ['BVCreateGuideLine']
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/BVPreprocessVolume.ui"))
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
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # input is validated via _checkCanCreateHeartRoi
        self.ui.heartROISelector.connect(
            'nodeAddedByUser(vtkMRMLNode*)',
            self.fitInputHeartRoiNodeToVolume
        )
        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

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
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterUpdate)
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
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[BVPreprocessVolumeParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterUpdate)
            self._cleanUpInputNodeObserver()
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterUpdate)

    def fitInputHeartRoiNodeToVolume(self):
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.heartRoi:
            self.logic.fitHeartRoiNode(self._parameterNode.inputVolume, self._parameterNode.heartRoi)

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
        if (self._parameterNode and self._parameterNode.inputVolume):
            self.ui.heartROISelector.addEnabled = True
        else:
            self.ui.heartROISelector.addEnabled = False

    def _setROIWidget(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.heartRoi:
            self.ui.MRMLMarkupsROIWidget.setMRMLMarkupsNode(self._parameterNode.heartRoi)
        else:
            self.ui.MRMLMarkupsROIWidget.setMRMLMarkupsNode(None)

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.heartRoi and self._parameterNode.costVolume:
            self.ui.applyButton.toolTip = "Compute preprocessed volume as cost volume"
            self.ui.applyButton.enabled = True
            return
        self.ui.applyButton.toolTip = "Select input volume, heart ROI and output volume nodes"
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
                print(f"{self.moduleName} ADD observer from:", node.GetName(), " : ", node.GetID())
                self.addObserver(node, e, callback)

    def _removeInputNodeObserver(self, eventList, callback):
        if not isinstance(eventList, list):
            eventList = [eventList]

        for e in eventList:
            prevNode = self.observer(e, callback)
            if prevNode is not None:
                logging.debug(f"{self.moduleName} Remove observer from:", prevNode.GetName(), ":", prevNode.GetID())
                # prevMarkersNodeName = prevMarkersNode.GetName()
                self.removeObserver(prevNode, e, callback)

    # def onHeartROIUpdate(self, caller=None, event=None):
    #     if (self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.heartRoi):
    #         canApply = self.logic.validateHeartROI(self._parameterNode.inputVolume, self._parameterNode.heartRoi)
    #         print('on heart roi update:', canApply)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            inputVolume = self._parameterNode.inputVolume
            heartRoi = self._parameterNode.heartRoi
            costVolume = self._parameterNode.costVolume
            if self.logic.validateHeartROI(inputVolume, heartRoi):
                self.logic.process(inputVolume, heartRoi, costVolume)
            else:
                e = 'ROI is bigger than input volume or too big (15mm * 15mm * 15mm)'
                slicer.util.errorDisplay("Failed to compute results: "+str(e))


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

    def fitHeartRoiNode(self, volumeNode: vtkMRMLScalarVolumeNode, roiNode: vtkMRMLMarkupsROINode) -> None:
        # roiNode.GetDisplayNode().SetFillVisibility(False)

        # fit roi
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLCropVolumeParametersNode'
        )
        cropVolumeParameters.SetInputVolumeNodeID(volumeNode.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(cropVolumeParameters)  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

    def validateHeartROI(self, volumeNode: vtkMRMLScalarVolumeNode, roiNode: vtkMRMLMarkupsROINode) -> bool:
        roiBound = np.zeros(6)
        volumeBound = np.zeros(6)
        roiNode.GetRASBounds(roiBound)
        volumeNode.GetRASBounds(volumeBound)
        minIdx = [0,2,4]
        maxIdx = [1,3,5]
        isBound = np.all(roiBound[minIdx] > volumeBound[minIdx]) and np.all(roiBound[maxIdx] < volumeBound[maxIdx])

        sizesMM = roiBound[maxIdx] - roiBound[minIdx]
        maxSizesMM = np.array([150,150,150])
        # roiVolumeMM3 = sizesMM[0] * sizesMM[1] * sizesMM[2]
        # logging.debug(f"validate roi: roi volume (mm^3)= {roiVolumeMM3}")
        isNotTooBig = np.all(sizesMM < maxSizesMM)
        logging.debug(f'{isBound=} {isNotTooBig=} {sizesMM=}')
        result = isBound and isNotTooBig

        return result

    def process(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        heartRoi: vtkMRMLMarkupsROINode,
        costVolume: vtkMRMLScalarVolumeNode,
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

        if not inputVolume or not costVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLCropVolumeParametersNode'
        )
        cropVolumeParameters.SetInputVolumeNodeID(inputVolume.GetID())
        cropVolumeParameters.SetROINodeID(heartRoi.GetID())
        cropVolumeParameters.SetOutputVolumeNodeID(costVolume.GetID())
        cropVolumeParameters.SetInterpolationMode(
            cropVolumeParameters.InterpolationLinear
        )
        cropVolumeParameters.SetVoxelBased(False) # Interpolated Cropping
        cropVolumeParameters.SetSpacingScalingConst(0.4)
        cropVolumeParameters.SetIsotropicResampling(True)
        slicer.modules.cropvolume.logic().Apply(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
