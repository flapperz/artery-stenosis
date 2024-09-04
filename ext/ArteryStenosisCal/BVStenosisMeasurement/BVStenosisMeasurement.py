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
from BVStenosisMeasurementLib.Controllers import CreateGuideLineController
from slicer import (
    vtkMRMLCommandLineModuleNode,
    vtkMRMLMarkupsCurveNode,
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsNode,
    vtkMRMLMarkupsROINode,
    vtkMRMLScalarVolumeNode,
    vtkMRMLSegmentationNode,
)
from slicer.parameterNodeWrapper import (
    WithinRange,
    parameterNodeWrapper,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
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
    markers: vtkMRMLMarkupsFiducialNode

    invertThreshold: bool = False

    #
    # internal
    #
    guideLine: vtkMRMLMarkupsCurveNode
    segmentation: vtkMRMLSegmentationNode
    costVolume: vtkMRMLScalarVolumeNode


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
        self.logic: Optional[BVStenosisMeasurementLogic] = None
        self._parameterNode: Optional[BVStenosisMeasurementParameterNode] = None
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
        self.ui.adjustVolumeDisplayButton.connect(
            'clicked(bool)', self.onAdjustVolumeDisplayButtonClicked
        )
        self.ui.prevStepButton.connect('clicked(bool)', self.onPrevStepButton)
        self.ui.markersSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)', self.setSimpleMarkups
        )

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
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.onParameterUpdate,
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

        preprocessParameterNode = (
            slicer.modules.bvpreprocessvolume.widgetRepresentation()
            .self()
            .logic.getParameterNode()
        )
        if not self._parameterNode.costVolume and preprocessParameterNode.costVolume:
            self._parameterNode.costVolume = preprocessParameterNode.costVolume


        #
        # initialize internal parameter
        #


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
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.onParameterUpdate,
            )

        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.onParameterUpdate,
            )
            self.onParameterUpdate()

    def setSimpleMarkups(self, markers):
        self.ui.simpleMarkupsWidget.setCurrentNode(markers)

    def _updateNodeObserver(self, caller=None, event=None) -> None:
        print('parameterNodeUpdate')
        # print(caller)

        eventList = [
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            vtkMRMLMarkupsNode.PointEndInteractionEvent,
            vtkMRMLMarkupsNode.PointRemovedEvent,
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

        logging.info(
            f'{prevMarkersNodeName} -> {markersNode.GetName() if markersNode else None}'
        )

    def _onMarkersModified(self, caller=None, event=None):
        """Handle markers change case: move, add, change markups node, reorder ?"""
        # TODO: check is markers in bound
        if (
            self._parameterNode.inputVolume
            and self._parameterNode.costVolume
            and self._parameterNode.markers.GetNumberOfControlPoints() > 1
        ):
            if not self._parameterNode.guideLine:
                guideLine = slicer.mrmlScene.AddNewNodeByClass(
                    'vtkMRMLMarkupsCurveNode'
                )
                guideLine.SetName(self.ui.guideLineSelector.baseName)
                guideLine.SetCurveTypeToLinear()
                self._parameterNode.guideLine = guideLine

            cliNode = self.logic.createGuideLine(
                self._parameterNode.costVolume,
                self._parameterNode.markers,
                self._parameterNode.guideLine,
            )
            self.ui.CLIProgressBar.setCommandLineModuleNode(cliNode)
        logging.debug('in _onMarkersModified')

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

            if not self._parameterNode:
                e = 'No ParameterNode'
                slicer.util.errorDisplay('Failed to start stenosis procedure: ' + str(e))
                return
            if (
                not self._parameterNode.inputVolume
                or not self._parameterNode.costVolume
            ):
                e = 'No inputVolume or costVolume specified'
                slicer.util.errorDisplay('Failed to start stenosis procedure: ' + str(e))
                return
            if self._parameterNode.guideLine and not self._parameterNode.guideLine.GetNumberOfControlPoints() > 5:
                e = 'No GuideLine or GuideLine to short'
                slicer.util.errorDisplay('Failed to start stenosis procedure: ' + str(e))
                return

            self.logic.process(
                self._parameterNode.inputVolume,
                self._parameterNode.costVolume,
                self._parameterNode.markers,
                self._parameterNode.guideLine,
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
        logging.debug('BVStenosisMeasurementLogic initialize')
        ScriptedLoadableModuleLogic.__init__(self)

        self.createGuideLineController = CreateGuideLineController()

        # debug volume
        # shNode = slicer.vtkMRMLSubjectHierachyNode.GetSubjectHierachyNode(slicer.mrmlScene)

        self.debugVolume = None

    def getParameterNode(self):
        return BVStenosisMeasurementParameterNode(super().getParameterNode())

    def adjustVolumeDisplay(self, volumeNode: vtkMRMLScalarVolumeNode) -> None:
        displayNode = volumeNode.GetDisplayNode()
        displayNode.SetDefaultColorMap()
        displayNode.SetInterpolate(1)
        displayNode.ApplyThresholdOff()
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindowLevel(1400, 50)

    def createGuideLine(
        self,
        costVolume: vtkMRMLScalarVolumeNode,
        markers: vtkMRMLMarkupsFiducialNode,
        guideLine: vtkMRMLMarkupsCurveNode,
        isSingleton = True
    ) -> vtkMRMLCommandLineModuleNode:
        return self.createGuideLineController.runCreateGuideLineAsync(costVolume, markers, guideLine, isSingleton)

    def createPatchROI(self):
        costVolumeNode = self.getParameterNode().costVolume

        # recreate heartROI
        heartROINode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode')
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLCropVolumeParametersNode'
        )
        cropVolumeParameters.SetInputVolumeNodeID(costVolumeNode.GetID())
        cropVolumeParameters.SetROINodeID(heartROINode.GetID())
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)

        guideLineNode = self.getParameterNode().guideLine

        controlPoints = np.array(
            [
                guideLineNode.GetNthControlPointPosition(i)
                for i in range(guideLineNode.GetNumberOfControlPoints())
            ]
        )

        # Compute bounding box of guideLine
        minBounds = np.min(controlPoints, axis=0)
        maxBounds = np.max(controlPoints, axis=0)

        # Expand the bounding box by 1 cm (10 mm)
        PATCH_EXPAND_MM = 10.0  # in mm
        minBounds -= PATCH_EXPAND_MM
        maxBounds += PATCH_EXPAND_MM

        # Create a new ROI encompassing the expanded bounding box
        patchROINode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsROINode', 'patchROI'
        )
        patchROINode.SetXYZ(minBounds + (maxBounds - minBounds) / 2.0)
        patchROINode.SetRadiusXYZ((maxBounds - minBounds) / 2.0)
        minBounds = np.min(controlPoints, axis=0)
        maxBounds = np.max(controlPoints, axis=0)

        # Get heartROI center and radius
        heartCenter = [0.0, 0.0, 0.0]
        heartRadius = [0.0, 0.0, 0.0]
        heartROINode.GetXYZ(heartCenter)
        heartROINode.GetRadiusXYZ(heartRadius)

        # Get patchROI center and radius
        patchCenter = [0.0, 0.0, 0.0]
        patchRadius = [0.0, 0.0, 0.0]
        patchROINode.GetXYZ(patchCenter)
        patchROINode.GetRadiusXYZ(patchRadius)

        heartCenter = np.array(heartCenter)
        heartRadius = np.array(heartRadius)
        patchCenter = np.array(patchCenter)
        patchRadius = np.array(patchRadius)

        intersectMinBounds = np.maximum(
            heartCenter - heartRadius, patchCenter - patchRadius
        )
        intersectMaxBounds = np.minimum(
            heartCenter + heartRadius, patchCenter + patchRadius
        )

        # If there is an intersection
        if np.all(intersectMinBounds <= intersectMaxBounds):
            intersectCenter = (intersectMinBounds + intersectMaxBounds) / 2.0
            intersectRadius = (intersectMaxBounds - intersectMinBounds) / 2.0
            patchROINode.SetXYZ(intersectCenter)
            patchROINode.SetRadiusXYZ(intersectRadius)
        else:
            e = "patchROI is not within heartROI bound"
            return Exception(e)

        return patchROINode

    def process(
        self,
        inputVolumeNode: vtkMRMLScalarVolumeNode,
        costVolumeNode: vtkMRMLScalarVolumeNode,
        markersNode: vtkMRMLMarkupsFiducialNode,
        guideLineNode: vtkMRMLMarkupsCurveNode,
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

        # Commonly used node
        mainWindow = slicer.util.mainWindow()
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        #
        # Segmentation
        #
        patchVolumeNode = self.createPatchROI()

        return

        slicer.util.setSliceViewerLayers(background=costVolumeNode)
        slicer.app.processEvents()
        mainWindow.moduleSelector().selectModule('GuidedArterySegmentation')

        vmtkSegWidget = (
            slicer.modules.guidedarterysegmentation.widgetRepresentation().self()
        )
        vmtkSegLogic = vmtkSegWidget.logic

        # TODO: not hard code slice node
        sliceNode = slicer.app.layoutManager().sliceWidget('Red').mrmlSliceNode()

        # must be call before curve selector
        vmtkSegWidget.ui.inputSliceNodeSelector.setCurrentNode(sliceNode)
        vmtkSegWidget.ui.inputCurveSelector.setCurrentNode(guideLineNode)
        vmtkSegWidget._parameterNode.inputIndensityTolerance = 100
        # in mm.
        vmtkSegWidget._parameterNode.neighbourhoodSize = 1.4
        vmtkSegWidget._parameterNode.tubeDiameter = 2.0
        vmtkSegWidget._parameterNode.extractCenterlines = False

        vmtkSegWidget.ui.applyButton.click()

        # set slice background back
        slicer.util.setSliceViewerLayers(background=inputVolumeNode)
        slicer.app.processEvents()

        segmentationNode = vmtkSegWidget._parameterNode.outputSegmentation
        segmentation = segmentationNode.GetSegmentation()
        # segmentation.GetNumberOfSegments()
        # segmentation.GetNthSegment(0)
        segmentID = segmentation.GetSegmentIDs()[0]

        #
        # --- Extract Centerline
        #

        mainWindow.moduleSelector().selectModule('ExtractCenterline')

        ecWidget = slicer.modules.extractcenterline.widgetRepresentation().self()
        ecLogic = ecWidget.logic

        ecWidget.ui.inputSurfaceSelector.setCurrentNode(segmentationNode)
        ecWidget.ui.inputSegmentSelectorWidget.setCurrentSegmentID(segmentID)

        # # Copy node
        # guideLineShID = shNode.GetItemByDataNode(guideLine)
        # endPointsShID = (
        #     slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(
        #         shNode, guideLineShID
        #     )
        # )
        # endPointsNode = shNode.GetItemDataNode(endPointsShID)
        # endPointsNode.SetName('Endpoints_' + markers.GetName())

        # TODO maybe we can use logic directly here
        endPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        endPointsNode.SetName('Endpoints_' + markersNode.GetName())
        slicer.util.updateMarkupsControlPointsFromArray(endPointsNode, slicer.util.arrayFromMarkupsControlPoints(guideLineNode))

        slicer.app.processEvents()
        ecWidget.ui.endPointsMarkupsSelector.setCurrentNode(endPointsNode)

        centerlineModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        centerlineModelNode.SetName('Centerline_model_' + markersNode.GetName())
        ecWidget.ui.outputCenterlineModelSelector.setCurrentNode(centerlineModelNode)

        centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsCurveNode'
        )
        centerlineCurveNode.SetName('Centerline_curve_' + markersNode.GetName())
        ecWidget.ui.outputCenterlineCurveSelector.setCurrentNode(centerlineCurveNode)
        ecWidget.onAutoDetectEndPoints()

        ecWidget.onApplyButton()

        return

        #
        # --- Cross Sectional Analysis
        #

        mainWindow.moduleSelector().selectModule('CrossSectionAnalysis')
        slicer.app.processEvents()

        # TODO: make this reapply able -> check how exportvisiblesegmentstomodel logic work
        exportFolderItemId = shNode.CreateFolderItem(
            shNode.GetSceneItemID(), 'Segments_' + guideLineNode.GetID()
        )
        slicer.modules.segmentations.logic().ExportSegmentsToModels(
            segmentationNode, [segmentID], exportFolderItemId
        )
        segmentModels = vtk.vtkCollection()
        shNode.GetDataNodesInBranch(exportFolderItemId, segmentModels)
        # Get exported model of first segment
        modelNode = segmentModels.GetItemAsObject(0)
        # print("fucker", type(modelNode))

        crossSecWidget = (
            slicer.modules.crosssectionanalysis.widgetRepresentation().self()
        )
        crossSecLogic = crossSecWidget.logic

        centerlineNode = vmtkSegWidget._parameterNode.outputCenterlineCurve

        crossSecWidget.ui.inputCenterlineSelector.setCurrentNode(centerlineNode)
        # crossSecWidget.ui.segmentationSelector.setCurrentNode(modelNode)
        crossSecWidget.ui.segmentationSelector.setCurrentNode(segmentationNode)
        crossSecWidget.ui.segmentSelector.setCurrentSegmentID(segmentID)

        crossSecWidget.ui.applyButton.click()
        slicer.app.processEvents()
        redSliceNode = slicer.app.layoutManager().sliceWidget('Red').mrmlSliceNode()
        greenSliceNode = slicer.app.layoutManager().sliceWidget('Green').mrmlSliceNode()
        crossSecWidget.ui.axialSliceViewSelector.setCurrentNode(redSliceNode)
        crossSecWidget.ui.longitudinalSliceViewSelector.setCurrentNode(greenSliceNode)

        with slicer.util.tryWithErrorDisplay(
            'Failed to compute cross-section area.', waitCursor=True
        ):
            outTableNode = crossSecWidget.ui.outputTableSelector.currentNode()

            cross_sec_area = slicer.util.arrayFromTableColumn(
                outTableNode, 'Cross-section area'
            )
            distances = slicer.util.arrayFromTableColumn(outTableNode, 'Distance')
            print(cross_sec_area.shape)
            # Fix constant for now
            max_distances = distances[-1]
            if len(distances) < 20 or (len(distances) and max_distances < 6):
                e = 'Vessel too short'
                slicer.util.errorDisplay('Failed to compute stenosis: ' + str(e))
                raise IndexError
            trim_length = 2.5
            min_trim_index = np.arange(len(distances))[distances > trim_length][0]
            max_trim_index = np.arange(len(distances))[
                distances > max_distances - trim_length
            ][0]
            trim_cross_sec_area = cross_sec_area[min_trim_index:max_trim_index]

            min_area_index = np.argmin(trim_cross_sec_area)
            min_area = np.min(trim_cross_sec_area)
            max_area = np.max(trim_cross_sec_area)  # avg( max(proximal), max(distal) )
            max_1_area = np.max(trim_cross_sec_area[:min_area_index])
            max_2_area = np.max(trim_cross_sec_area[min_area_index:])
            normal_reference = (max_1_area + max_2_area) * 0.5

            if max_area and normal_reference:
                stenosis = 1 - (min_area / max_area)
                stenosis_f0 = 1 - (min_area / normal_reference)

                stenosis_percent_str = f'{stenosis*100:.3f}'
                stenosis_f0_percent_str = f'{stenosis_f0*100:.3f}'

                outInfo = 'Result!\n'
                outInfo += '\n'
                outInfo += f'vessel length: {max_distances=:.3f} mm. trim with {trim_length} mm. both side'
                outInfo += '\n'
                outInfo += '====== stenosis ======\n'
                outInfo += '\n'
                outInfo += f'{min_area=:.3f} (mm^2)\n'
                outInfo += f'{max_area=:.3f} (mm^2)\n'
                outInfo += '\n'
                outInfo += f'stenosis : {stenosis_percent_str}%\n'
                outInfo += '\n'
                outInfo += '====== formula 0 ======\n'
                outInfo += '\n'
                outInfo += f'{min_area=:.3f} (mm^2)\n'
                outInfo += f'{max_1_area=:.3f} (mm^2)\n'
                outInfo += f'{max_2_area=:.3f} (mm^2)\n'
                outInfo += f'{normal_reference=:.3f} (mm^2)\n'
                outInfo += '\n'
                outInfo += f'stenosis : {stenosis_f0_percent_str}%'

                print('===== Report stenosis result =====')
                print(outInfo)

                slicer.util.infoDisplay(outInfo, self.moduleName)
            else:
                e = f'maximum area is zero {min_area=}, {max_area=}'
                slicer.util.errorDisplay('Failed to compute stenosis: ' + str(e))

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

#
# BVStenosisMeasurementTest
#
# --Test--
#

class BVStenosisMeasurementTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        # self.test_BVStenosisMeasurement1()
        # self.test_GuideLines()
        self.test_Segmentation()

    def test_BVStenosisMeasurement1(self):
        self.delayDisplay('Starting the test')
        self.assertEqual(1,1)

        slicer.util.loadScene( '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/2023-10-26-Scene.mrb')

        self.delayDisplay('Test passed')

    def test_Segmentation(self):
        mainWindow = slicer.util.mainWindow()

        slicer.util.loadScene(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/createsegtest.mrb'
        )

        ladNode = slicer.util.getNode('LAD')
        lcxDNode = slicer.util.getNode('LCX-D')
        rcaDNode = slicer.util.getNode('RCA-D')
        roiNode = slicer.util.getNode('R')
        inputVolumeNode = slicer.util.getNode('14: Body Soft Tissue')
        costVolumeNode = slicer.util.getNode('BV_COSTVOLUME')

        ladGLNode = slicer.util.getNode('GL_LAD')
        markers = slicer.util.getNode('SEED_SPARSE_LAD')

        logic = BVStenosisMeasurementLogic()
        logic.process(inputVolumeNode, costVolumeNode, markers, ladGLNode)

    def test_GuideLines(self):
        import time

        self.delayDisplay('Start testing GuideLines Creation')

        slicer.util.reloadScriptedModule('BVPreprocessVolume')
        mainWindow = slicer.util.mainWindow()

        ladNode, lcxDNode, rcaDNode, volumeNode, roiNode = self.loadScene()

        # run preprocess volume
        startTime = time.time()
        costVolumeNode = self.runPreprocessVolume(volumeNode, roiNode)
        stopTime = time.time()
        preprocessReport = f'Preprocess Volume in {stopTime-startTime:.3f} s'
        self.delayDisplay(preprocessReport)
        print(preprocessReport)
        logic = BVStenosisMeasurementLogic()

        # run craeteGuideLine

        for name, markersIndices, markupsNode in zip(
            ('LAD', 'LCX', 'RCA'), ((0, -1), (0, 15, -1), (0, -1)), (ladNode, lcxDNode, rcaDNode)
        ):
            startTime = time.time()

            markupsArray = slicer.util.arrayFromMarkupsControlPoints(markupsNode)

            markersNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            markersNode.SetName(f'SEED_SPARSE_{name}')
            slicer.util.updateMarkupsControlPointsFromArray(markersNode, markupsArray[markersIndices, :])

            curveNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode')
            curveName = f'GL_{name}'
            curveNode.SetName(curveName)
            logic.createGuideLine(costVolumeNode, markersNode, curveNode, isSingleton=False)
            self.delayDisplay(f'Finish Create Guideline: {name} -> {curveName}')

            stopTime = time.time()
            report = '\n'
            report = f'CreateGuideLine: {name}\n'
            report += '-' * len(report) + '\n'
            report += f'time: {stopTime - startTime:.3f}\n'
            report += f'seedIndex: {markersIndices}\n'
            print(report)

        mainWindow.moduleSelector().selectModule('Markups')

    def loadScene(self):
        slicer.util.loadScene(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/update3.mrb'
        )
        # lcxDNode = slicer.util.loadMarkups(
        #     '/Users/flap/Source/artery-stenosis/data/slicer-scene/markups/LCX-D.mrk.json'
        # )
        # rcaDNode = slicer.util.loadMarkups(
        #     '/Users/flap/Source/artery-stenosis/data/slicer-scene/markups/RCA-D.mrk.json'
        # )
        # roiNode = slicer.util.loadNodeFromFile(
        #     '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/R.mrk.json'
        # )
        ladNode = slicer.util.getNode('LAD')
        lcxDNode = slicer.util.getNode('LCX-D')
        rcaDNode = slicer.util.getNode('RCA-D')
        roiNode = slicer.util.getNode('R')
        volumeNode = slicer.util.getNode('14: Body Soft Tissue')

        return ladNode, lcxDNode, rcaDNode, volumeNode, roiNode

    def runPreprocessVolume(self, volumeNode, roiNode):
        preprocessWidget = (
            slicer.modules.bvpreprocessvolume.widgetRepresentation().self()
        )
        preprocessWidget.ui.inputVolumeSelector.setCurrentNode(volumeNode)
        preprocessWidget.ui.heartROISelector.setCurrentNode(roiNode)
        preprocessWidget.onApplyButton()
        costVolume = preprocessWidget.logic.getParameterNode().costVolume
        return costVolume
