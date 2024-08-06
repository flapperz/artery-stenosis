import logging
import os
from importlib import reload
from typing import Annotated, Optional

import numpy as np
import slicer
import slicer.util
import vtk
from BVStenosisMeasurementLib import MRMLUtils
from BVStenosisMeasurementLib.BVConstants import BVTextConst
from slicer import (
    vtkMRMLCommandLineModuleNode,
    vtkMRMLMarkupsCurveNode,
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsNode,
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
        self.parent.contributors = ['Krit Cholapand (Chulalongkorn University)']
        self.parent.helpText = f"""
    Artery Stenosis Measurement version {self.version}. Documentation is available
    <a href="https://github.com/flapperz/artery-stenosis">here</a>.
    """
        self.parent.acknowledgementText = """TODO: ACKNOWLEDGEMENT"""

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
    costVolume: vtkMRMLScalarVolumeNode
    markers: vtkMRMLMarkupsFiducialNode

    invertThreshold: bool = False

    #
    # internal
    #
    _guideLine: vtkMRMLMarkupsCurveNode


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

        # input is validated via _checkCanCreateHeartRoi
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButtonClicked)
        self.ui.adjustVolumeDisplayButton.connect('clicked(bool)', self.onAdjustVolumeDisplayButtonClicked)
        self.ui.prevStepButton.connect('clicked(bool)', self.onPrevStepButton)

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

        if not self._parameterNode._guideLine:
            guideLine = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode')
            guideLine.SetName(f"{BVTextConst.internal_node_prefix} guideLine")
            self._parameterNode._guideLine = guideLine

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

    def _updateNodeObserver(self, caller=None, event=None) -> None:
        print('parameterNodeUpdate')
        # print(caller)

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

        print(f'{prevMarkersNodeName} -> {markersNode.GetName() if markersNode else None}')


    def _onMarkersModified(self, caller=None, event=None):
        """Handle markers change case: move, add, change markups node, reorder ?"""
        # TODO: check can apply
        if self._parameterNode.inputVolume and self._parameterNode.markers.GetNumberOfControlPoints() > 1:
            cliNode = self.logic.processMarkers(
                self._parameterNode.costVolume,
                self._parameterNode.markers,
                self._parameterNode._guideLine
            )
            self.ui.CLIProgressBar.setCommandLineModuleNode(cliNode)
        logging.debug("in _onMarkersModified")

    def onParameterUpdate(self, caller=None, event=None):
        self._checkCanApply(caller=caller, event=event)
        self._updateNodeObserver(caller=caller, event=event)

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

    def onApplyButtonClicked(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(
            'Failed to compute results.', waitCursor=True
        ):
            # self.logic.createProcessedVolume(
            #     self._parameterNode.inputVolume, self._parameterNode.thresholdedVolume
            # )
            # self.logic.processMarkers(
            #     self._parameterNode.inputVolume,
            #     self._parameterNode.markers,
            #     self._parameterNode._guideLine
            # )
            # # Compute output
            if not self._parameterNode or not self._parameterNode.inputVolume or not self._parameterNode.costVolume or not self._parameterNode._guideLine.GetNumberOfControlPoints() > 10:
                e = "input invalid"
                slicer.util.errorDisplay('Failed to compute results: ' + str(e))
                return
            self.logic.process(
                self._parameterNode.inputVolume,
                self._parameterNode.costVolume,
                self._parameterNode.markers,
                self._parameterNode._guideLine
            )

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

    def onAdjustVolumeDisplayButtonClicked(self) -> None:
        if self._parameterNode and self._parameterNode.inputVolume:
            self.logic.adjustVolumeDisplay(self._parameterNode.inputVolume)

    def onPrevStepButton(self) -> None:
        mainWindow = slicer.util.mainWindow()
        mainWindow.moduleSelector().selectModule('BVPreprocessVolume')


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

    def _startProcessMarkers(self, costVolume, markers):
        markersIJK = [
            MRMLUtils.getFiducialAsIJK(markers, i, self.ras2ijkMat)
            for i in range(markers.GetNumberOfControlPoints())
        ]
        flattenMarkers = [x for ijk in markersIJK for x in ijk]
        logging.debug(f"input flattenMarkers: {flattenMarkers}")
        parameter = {
            "inputVolume" : costVolume,
            "inFlattenMarkersIJK" : flattenMarkers
        }
        BVCreateGuideLine = slicer.modules.bvcreateguideline
        cliNode = slicer.cli.run(BVCreateGuideLine, None, parameter)
        return cliNode

    def _onProcessMarkersUpdate(self, cliNode, event):
        # logging.debug("Got a %s from a %s : %s" % (event, cliNode.GetClassName(), cliNode.GetName()))

        status = cliNode.GetStatus()

        if status & cliNode.Completed:
            if status & cliNode.ErrorsMask:
                # error
                errorText = cliNode.GetErrorText()
                logging.debug("CLI execution failed: " + errorText)
            else:
                # success
                outIJK = cliNode.GetParameterAsString("outFlattenMarkersIJK")
                logging.debug("CLI execution succeeded. Output model node ID: " + outIJK)
                self.guideLineNode.RemoveAllControlPoints()
                # TODO: refactor out create curve function
                logging.debug(f"CLI output: {outIJK}")
                outIJK = [int(x) for x in outIJK.split(',')]
                # outKJI = flattenMarkers

                pathKJI = []
                logging.debug(f"out path length: {len(outIJK)}")
                for i in range(0, len(outIJK), 3):
                    pathKJI.append([outIJK[i+2], outIJK[i+1], outIJK[i]])
                logging.debug(f"formatted: {pathKJI}")
                MRMLUtils.createCurve(pathKJI, self.guideLineNode, self.ijk2rasMat, 0.5)
            slicer.mrmlScene.RemoveNode(cliNode)
        # TODO: better if-else

        if status & cliNode.Cancelled:

            slicer.mrmlScene.RemoveNode(cliNode)

    def adjustVolumeDisplay(self, volumeNode: vtkMRMLScalarVolumeNode) -> None:
        displayNode = volumeNode.GetDisplayNode()
        displayNode.SetDefaultColorMap()
        displayNode.SetInterpolate(1)
        displayNode.ApplyThresholdOff()
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindowLevel(1400, 50)

    def processMarkers(
            self,
            costVolume: vtkMRMLScalarVolumeNode,
            markers: vtkMRMLMarkupsFiducialNode,
            guideLine: vtkMRMLMarkupsCurveNode
    ) -> vtkMRMLCommandLineModuleNode:
        if self.createGuideLineCliNode:
            self.createGuideLineCliNode.Cancel()

        self.guideLineNode = guideLine

        x = vtk.vtkMatrix4x4()
        costVolume.GetIJKToRASMatrix(x)
        self.ijk2rasMat = MRMLUtils.vtk4x4matrix2numpy(x)

        costVolume.GetRASToIJKMatrix(x)
        self.ras2ijkMat = MRMLUtils.vtk4x4matrix2numpy(x)

        self.createGuideLineCliNode = self._startProcessMarkers(costVolume, markers)
        self.createGuideLineCliNode.AddObserver(vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self._onProcessMarkersUpdate)
        return self.createGuideLineCliNode

    def process(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        costVolume: vtkMRMLScalarVolumeNode,
        markers: vtkMRMLMarkupsFiducialNode,
        guideLine: vtkMRMLMarkupsCurveNode
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

        # if not inputVolume or not outputVolume:
        #     raise ValueError('Input or output volume is invalid')

        import time

        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        # TODO: maybe we can reuse cliNode / parameter node
        # cliParams = {
        #     'InputVolume': inputVolume.GetID(),
        #     'OutputVolume': outputVolume.GetID(),
        #     'ThresholdValue': imageThreshold,
        #     'ThresholdType': 'Above' if invert else 'Below',
        # }
        # cliNode = slicer.cli.run(
        #     slicer.modules.thresholdscalarvolume,
        #     None,
        #     cliParams,
        #     wait_for_completion=True,
        #     update_display=showResult,
        # )

        # # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        # slicer.mrmlScene.RemoveNode(cliNode)
        mainWindow = slicer.util.mainWindow()

        #
        # --- Segmentation + extract centerline
        #

        slicer.util.setSliceViewerLayers(background=costVolume)
        slicer.app.processEvents()
        mainWindow.moduleSelector().selectModule('GuidedArterySegmentation')

        vmtkSegWidget = slicer.modules.guidedarterysegmentation.widgetRepresentation().self()
        vmtkSegLogic = vmtkSegWidget.logic
        # TODO: not hard code slice node
        sliceNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()

        # must be call before curve selector
        vmtkSegWidget.ui.inputSliceNodeSelector.setCurrentNode(sliceNode)
        vmtkSegWidget.ui.inputCurveSelector.setCurrentNode(guideLine)
        vmtkSegWidget._parameterNode.inputIndensityTolerance = 100
        # in mm.
        vmtkSegWidget._parameterNode.neighbourhoodSize = 1.4
        vmtkSegWidget._parameterNode.tubeDiameter = 2.0
        vmtkSegWidget._parameterNode.extractCenterlines = True

        vmtkSegWidget.ui.applyButton.click()

        # set slice background back
        slicer.util.setSliceViewerLayers(background=inputVolume)
        slicer.app.processEvents()

        #
        # --- Cross Sectional Analysis
        #

        mainWindow.moduleSelector().selectModule('CrossSectionAnalysis')
        slicer.app.processEvents()

        segmentationNode = vmtkSegWidget._parameterNode.outputSegmentation
        segmentID = "Segment_" + guideLine.GetID()
        # TODO: make this reapply able -> check how exportvisiblesegmentstomodel logic work
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "Segments_" + guideLine.GetID())
        slicer.modules.segmentations.logic().ExportSegmentsToModels(segmentationNode, [segmentID], exportFolderItemId)
        segmentModels = vtk.vtkCollection()
        shNode.GetDataNodesInBranch(exportFolderItemId, segmentModels)
        # Get exported model of first segment
        modelNode = segmentModels.GetItemAsObject(0)
        # print("fucker", type(modelNode))

        crossSecWidget = slicer.modules.crosssectionanalysis.widgetRepresentation().self()
        crossSecLogic = crossSecWidget.logic

        centerlineNode = vmtkSegWidget._parameterNode.outputCenterlineCurve

        crossSecWidget.ui.inputCenterlineSelector.setCurrentNode(centerlineNode)
        crossSecWidget.ui.segmentationSelector.setCurrentNode(modelNode)
        # crossSecWidget.ui.segmentationSelector.setCurrentNode(segmentationNode)
        # crossSecWidget.ui.segmentSelector.setCurrentSegmentID(segmentID)

        crossSecWidget.ui.applyButton.click()
        slicer.app.processEvents()
        redSliceNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()
        greenSliceNode = slicer.app.layoutManager().sliceWidget("Green").mrmlSliceNode()
        crossSecWidget.ui.axialSliceViewSelector.setCurrentNode(redSliceNode)
        crossSecWidget.ui.longitudinalSliceViewSelector.setCurrentNode(greenSliceNode)


        with slicer.util.tryWithErrorDisplay(
            'Failed to compute cross-section area.', waitCursor=True
        ):
            outTableNode = crossSecWidget.ui.outputTableSelector.currentNode()

            cross_sec_area = slicer.util.arrayFromTableColumn(outTableNode, 'Cross-section area')
            min_area = np.min(cross_sec_area)
            max_area = np.max(cross_sec_area) # avg( max(proximal), max(distal) )
            if max_area:
                stenosis = 1 - (min_area / max_area)
                stenosis_percent_str = f'{stenosis*100:.3f}'
                slicer.util
                print(f'{min_area=}, {max_area=}, {stenosis}:{stenosis_percent_str} %')
                print(f'{min_area=} (mm^2)')
                print(f'{max_area=} (mm^2)')
                print(f'{stenosis=} %')
                outInfo = f'Result!\n{min_area=:.3f} (mm^2)\n{max_area=:.3f} (mm^2)\nstenosis : {stenosis_percent_str}%'
                slicer.util.infoDisplay(outInfo, self.moduleName)
            else:
                e = f'maximum area is zero {min_area=}, {max_area=}'
                slicer.util.errorDisplay('Failed to compute stenosis: ' + str(e))

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
