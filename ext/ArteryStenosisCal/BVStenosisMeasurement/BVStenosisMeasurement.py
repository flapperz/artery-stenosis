import logging
import os
from importlib import reload
from typing import Annotated, Optional

import slicer
import vtk
from BVStenosisMeasurementLib import MRMLUtils
from slicer import (
    vtkMRMLMarkupsCurveNode,
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsNode,
    vtkMRMLMarkupsROINode,
    vtkMRMLScalarVolumeNode,
)
from slicer.parameterNodeWrapper import (
    WithinRange,
    parameterNodeWrapper,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

reload(MRMLUtils)

#
# BVStenosisMeasurement
#


class BVStenosisMeasurement(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.version = 3.0
        self.parent.title = 'BV Stenosis Measurement'
        self.parent.categories = ['Chula BV']
        self.parent.dependencies = ['BVCreateGuideLine']
        self.parent.contributors = ['John Doe (AnyWare Corp.)']
        self.parent.helpText = f"""
    Artery Stenosis Measurement version {self.version}. Documentation is available
    <a href="https://github.com/flapperz/artery-stenosis">here</a>.
    """
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)


#
# BVStenosisMeasurementParameterNode
#
# --Parameter--
#


@parameterNodeWrapper
class BVStenosisMeasurementParameterNode:
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
    markers: vtkMRMLMarkupsFiducialNode

    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode

    #
    # internal
    #
    _guideLine: vtkMRMLMarkupsCurveNode
    _processedVolume: vtkMRMLScalarVolumeNode


#
# BVStenosisMeasurementWidget
#
# --Widget--
#


class BVStenosisMeasurementWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic : Optional[BVStenosisMeasurementLogic] = None
        self._parameterNode : Optional[BVStenosisMeasurementParameterNode] = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/BVStenosisMeasurement.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BVStenosisMeasurementLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Buttons
        # connected signal is not reload
        self.ui.heartRoiSelector.connect(
            'nodeAddedByUser(vtkMRMLNode*)',
            lambda node: self.logic.fitHeartRoiNode(
                self._parameterNode.inputVolume, node
            ),
        )
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButtonClicked)
        self.ui.volumeRoiLockButton.connect('stateChanged(int)', self.onVolumeRoiLockButtonClicked)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onParameterUpdate
            )

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

        #
        # initialize internal parameter
        #

        internal_tmp_prefix = "BVInternal:DO_NOT_USE"

        if not self._parameterNode._guideLine:
            guideLine = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode')
            guideLine.SetName(f"{internal_tmp_prefix} guideLine")
            self._parameterNode._guideLine = guideLine

        if not self._parameterNode._processedVolume and self._parameterNode.inputVolume:
            # self.logic.createProcessedVolume(self._parameterNode.inputVolume, self._parameterNode._processedVolume)
            pass

    def setParameterNode(
        self, inputParameterNode: Optional[BVStenosisMeasurementParameterNode]
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onParameterUpdate
            )

        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onParameterUpdate
            )
            self.onParameterUpdate()

    def _updateInputObserver(self, caller=None, event=None) -> None:
        print('parameterNodeUpdate')
        # print(caller)
        print(event)

        eventList = [
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            vtkMRMLMarkupsNode.PointEndInteractionEvent,
        ]
        markersNode = self._parameterNode.markers

        # https://apidocs.slicer.org/master/classvtkMRMLMarkupsNode.html#aceeef8806df28e3807988c38510e56caad628dfaf54f410d2f4c6bc5800aa8a30
        prevMarkersNodeName = None
        for e in eventList:
            prevMarkersNode = self.observer(e, self._onMarkersModified)
            if prevMarkersNode is not None:
                prevMarkersNodeName = prevMarkersNode.GetName()
                self.removeObserver(prevMarkersNode, e, self._onMarkersModified)
        for e in eventList:
            if markersNode is not None:
                self.addObserver(markersNode, e, self._onMarkersModified)

        # markersNode = self._parameterNode.markers
        # if self._parameterNode._markersPrev:
        #     self.removeObserver(
        #         self._parameterNode._markersPrev,
        #         vtkMRMLMarkupsNode.PointEndInteractionEvent,
        #         self._onMarkersModified,
        #     )
        #     self.removeObserver(
        #             self._parameterNode._markersPrev,
        #             vtkMRMLMarkupsNode.PointPositionDefinedEvent,
        #             self._onMarkersModified,
        #             )
        # if self._parameterNode.markers:
        #     self.addObserver(
        #             markersNode,
        #             vtkMRMLMarkupsNode.PointEndInteractionEvent,
        #             self._onMarkersModified,
        #             )
        #     self.addObserver(
        #             markersNode,
        #             vtkMRMLMarkupsNode.PointPositionDefinedEvent,
        #             self._onMarkersModified,
        #             )
        # if markersNode != self._parameterNode._markersPrev:
        #     self._parameterNode._markersPrev = markersNode

        print(f'{prevMarkersNodeName} -> {markersNode.GetName() if markersNode else None}')


    def _onMarkersModified(self, caller=None, event=None):
        """Handle markers change case: move, add, change markups node, reorder ?"""
        # TODO: check can apply
        if self._parameterNode.inputVolume and self._parameterNode.markers.GetNumberOfControlPoints() > 1:
            self.logic.processMarkers(
                self._parameterNode.inputVolume,
                self._parameterNode.markers,
                self._parameterNode._guideLine
            )
        logging.debug("in _onMarkersModified")

    def onParameterUpdate(self, caller=None, event=None):
        self._checkCanApply(caller=caller, event=event)
        self._checkCanPreprocess(caller=caller, event=event)
        self._checkCanCreateHeartRoi(caller=caller, event=event)

    def _checkCanApply(self, caller=None, event=None) -> None:
        if (
            self._parameterNode
            and self._parameterNode.inputVolume
            and self._parameterNode.markers
            # and self._parameterNode.thresholdedVolume
        ):
            self.ui.applyButton.toolTip = 'Compute output volume'
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = 'Select Input Volume and Markers'
            self.ui.applyButton.enabled = False

    def _checkCanCreateHeartRoi(self, caller=None, event=None) -> None:
        if (self._parameterNode and self._parameterNode.inputVolume):
            self.ui.heartRoiSelector.addEnabled = True
        else:
            self.ui.heartRoiSelector.addEnabled = False

    def _checkCanPreprocess(self, caller=None, event=None):
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.heartRoi:
            self.ui.volumeRoiLockButton.toolTip = 'Lock heart RoI and input Volume selector, Compute preprocessedVolume'
            self.ui.volumeRoiLockButton.enabled = True
        else:
            self.ui.volumeRoiLockButton.toolTip = 'Select heart RoI and input Volume selector'
            self.ui.volumeRoiLockButton.toolTip = True
            self.ui.volumeRoiLockButton.enabled = False
            self.ui.volumeRoiLockButton.setChecked(False)

    def onApplyButtonClicked(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(
            'Failed to compute results.', waitCursor=True
        ):
            self.logic.createProcessedVolume(
                self._parameterNode.inputVolume, self._parameterNode.thresholdedVolume
            )
            # self.logic.processMarkers(
            #     self._parameterNode.inputVolume,
            #     self._parameterNode.markers,
            #     self._parameterNode._guideLine
            # )
            # # Compute output
            # self.logic.process(
            #     self.ui.inputVolumeSelector.currentNode(),
            #     self.ui.outputSelector.currentNode(),
            #     self.ui.imageThresholdSliderWidget.value,
            #     self.ui.invertOutputCheckBox.checked,
            # )

            # # Compute inverted output (if needed)
            # if self.ui.invertedOutputSelector.currentNode():
            #     # If additional output volume is selected then result with inverted threshold is written there
            #     self.logic.process(
            #         self.ui.inputVolumeSelector.currentNode(),
            #         self.ui.invertedOutputSelector.currentNode(),
            #         self.ui.imageThresholdSliderWidget.value,
            #         not self.ui.invertOutputCheckBox.checked,
            #         showResult=False,
            #     )

    def onVolumeRoiLockButtonClicked(self, state: int) -> None:
        """
        :param int state: 0 off / 2 on
        """

        # state = self.ui.volumeRoiLockButton.checkState()
        logging.debug(f'volumeRoiLockButton state: {state}')


#
# BVStenosisMeasurementLogic
#
# --Logic--
#


class BVStenosisMeasurementLogic(ScriptedLoadableModuleLogic):
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
        logging.debug("BVStenosisMeasurementLogic initialize")
        ScriptedLoadableModuleLogic.__init__(self)

        # state
        self.createGuideLineCliNode = None
        # TODO: change to function factory or wrapper
        self.guideLineNode = None
        self.ijk2rasMat = None
        self.ras2ijkMat = None

        # debug volume
        # shNode = slicer.vtkMRMLSubjectHierachyNode.GetSubjectHierachyNode(slicer.mrmlScene)

        self.debugVolume = None

    def getParameterNode(self):
        return BVStenosisMeasurementParameterNode(super().getParameterNode())

    def _startProcessMarkers(self, inputVolume, markers):
        markersIJK = [
            MRMLUtils.getFiducialAsIJK(markers, i, self.ras2ijkMat)
            for i in range(markers.GetNumberOfControlPoints())
        ]
        flattenMarkers = [x for ijk in markersIJK for x in ijk]
        print("input flattenMarkers:", flattenMarkers)
        parameter = {
            "inputVolume" : inputVolume,
            "inFlattenMarkersIJK" : flattenMarkers
        }
        BVCreateGuideLine = slicer.modules.bvcreateguideline
        cliNode = slicer.cli.run(BVCreateGuideLine, None, parameter)
        return cliNode

    def _onProcessMarkersUpdate(self, cliNode, event):
        print("Got a %s from a %s" % (event, cliNode.GetClassName()))
        if cliNode.IsA('vtkMRMLCommandLineModuleNode'):
            print("Status is %s" % cliNode.GetStatusString())

        if cliNode.GetStatus() & cliNode.Completed:
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                # error
                errorText = cliNode.GetErrorText()
                print("CLI execution failed: " + errorText)
            else:
                # success
                outIJK = cliNode.GetParameterAsString("outFlattenMarkersIJK")
                print("CLI execution succeeded. Output model node ID: "+outIJK)
                self.guideLineNode.RemoveAllControlPoints()
                # TODO: refactor out create curve function
                print("CLI output:", outIJK)
                outIJK = [int(x) for x in outIJK.split(',')]
                # outKJI = flattenMarkers

                pathKJI = []
                print("out path length:", len(outIJK))
                for i in range(0, len(outIJK), 3):
                    pathKJI.append([outIJK[i+2], outIJK[i+1], outIJK[i]])
                print("formatted:", pathKJI)
                MRMLUtils.createCurve(pathKJI, self.guideLineNode, self.ijk2rasMat, 0.5)
            slicer.mrmlScene.RemoveNode(cliNode)
        # TODO: better if-else

        if cliNode.GetStatus() & cliNode.Cancelled:
            slicer.mrmlScene.RemoveNode(cliNode)

    def createProcessedVolume(self, inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode):
        # ? example for create new volume
        imageDimensions = inputVolume.GetImageData().GetDimensions()
        voxelType = vtk.VTK_DOUBLE
        imageOrigin = inputVolume.GetImageData().GetOrigin()
        imageSpacing = inputVolume.GetImageData().GetSpacing()
        fillVoxelValue = 0
        volumeToRAS = inputVolume.GetImageData().GetDirectionMatrix()
        imageDirections = MRMLUtils.vtk3x3matrix2numpy(volumeToRAS)

        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageDimensions)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "mytestVolume2")
        outputVolume.SetOrigin(imageOrigin)
        outputVolume.SetSpacing(imageSpacing)
        outputVolume.SetIJKToRASDirections(imageDirections)
        outputVolume.SetAndObserveImageData(imageData)
        outputVolume.CreateDefaultDisplayNodes()
        outputVolume.CreateDefaultStorageNode()

    def fitHeartRoiNode(self, volumeNode: vtkMRMLScalarVolumeNode, roiNode: vtkMRMLMarkupsROINode):
        roiNode.GetDisplayNode().SetFillVisibility(False)

        # fit roi
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        cropVolumeParameters.SetInputVolumeNodeID(volumeNode.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(cropVolumeParameters)  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

    def processMarkers(
            self,
            processedVolume: vtkMRMLScalarVolumeNode,
            markers: vtkMRMLMarkupsFiducialNode,
            guideLine: vtkMRMLMarkupsCurveNode
    ) -> None:
        if self.createGuideLineCliNode:
            self.createGuideLineCliNode.Cancel()

        self.guideLineNode = guideLine

        x = vtk.vtkMatrix4x4()
        processedVolume.GetIJKToRASMatrix(x)
        self.ijk2rasMat = MRMLUtils.vtk4x4matrix2numpy(x)

        processedVolume.GetRASToIJKMatrix(x)
        self.ras2ijkMat = MRMLUtils.vtk4x4matrix2numpy(x)

        self.createGuideLineCliNode = self._startProcessMarkers(processedVolume, markers)
        self.createGuideLineCliNode.AddObserver('ModifiedEvent', self._onProcessMarkersUpdate)

    def process(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        outputVolume: vtkMRMLScalarVolumeNode,
        imageThreshold: float,
        invert: bool = False,
        showResult: bool = True,
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

        if not inputVolume or not outputVolume:
            raise ValueError('Input or output volume is invalid')

        import time

        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        # TODO: maybe we can reuse cliNode / parameter node
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below',
        }
        cliNode = slicer.cli.run(
            slicer.modules.thresholdscalarvolume,
            None,
            cliParams,
            wait_for_completion=True,
            update_display=showResult,
        )

        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
