import logging
import os
from importlib import reload
from typing import Annotated, Optional

import BVStenosisMeasurementLib.Controllers as Controllers
import numpy as np
import qt
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
    vtkMRMLMarkupsROINode,
    vtkMRMLScalarVolumeNode,
    vtkMRMLSegmentationNode,
    vtkMRMLTableNode,
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
reload(Controllers)

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
    reportTable: vtkMRMLTableNode


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
        self.resetReformatView()

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

    def resetReformatView(self) -> None:
        redSliceNode = slicer.app.layoutManager().sliceWidget('Red').mrmlSliceNode()
        redSliceNode.SetOrientationToDefault()
        greenSliceNode = slicer.app.layoutManager().sliceWidget('Green').mrmlSliceNode()
        greenSliceNode.SetOrientationToDefault()

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

            if not self._parameterNode.segmentation:
                self._parameterNode.segmentation = slicer.mrmlScene.AddNewNodeByClass(
                    'vtkMRMLSegmentationNode'
                )
            if not self._parameterNode.reportTable:
                self._parameterNode.reportTable = slicer.mrmlScene.AddNewNodeByClass(
                    'vtkMRMLTableNode'
                )

            self.logic.process(
                self._parameterNode.inputVolume,
                self._parameterNode.costVolume,
                self._parameterNode.markers,
                self._parameterNode.guideLine,
                self._parameterNode.segmentation,
                self._parameterNode.reportTable
            )

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

        self.createGuideLineController = Controllers.CreateGuideLineController()

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

    def createPatchROI(
        self,
        costVolumeNode: vtkMRMLScalarVolumeNode,
        guideLineNode: vtkMRMLMarkupsCurveNode,
    ):
        # recreate heartROI
        heartROINode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode')
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLCropVolumeParametersNode'
        )
        cropVolumeParameters.SetInputVolumeNodeID(costVolumeNode.GetID())
        cropVolumeParameters.SetROINodeID(heartROINode.GetID())
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)

        nControlPoints = guideLineNode.GetNumberOfControlPoints()
        controlPoints = np.array(
            [
                guideLineNode.GetNthControlPointPosition(i)
                for i in range(nControlPoints)
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

        # Cleanup
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(heartROINode)

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

    def createPatchVolume(
            self,
            inputVolumeNode,
            patchROINode
    ):
        # Create simulate contrast patch

        SPACING_MM = 0.25
        # HIST_RADIUS = [7,7,7]
        HIST_RADIUS = 5
        HIST_ALPHA = 0.6
        HIST_BETA = 0.3

        patchVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')

        # Create reference resample volume
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLCropVolumeParametersNode'
        )
        cropVolumeParameters.SetInputVolumeNodeID(inputVolumeNode.GetID())
        cropVolumeParameters.SetROINodeID(patchROINode.GetID())
        cropVolumeParameters.SetOutputVolumeNodeID(patchVolume.GetID())
        cropVolumeParameters.SetInterpolationMode(
            cropVolumeParameters.InterpolationNearestNeighbor
        )

        # Assume all volume have minimum spacing of 0.78125
        spacingScaling = SPACING_MM / 0.78125
        cropVolumeParameters.SetVoxelBased(False)
        cropVolumeParameters.SetSpacingScalingConst(spacingScaling)
        cropVolumeParameters.SetIsotropicResampling(True)
        slicer.modules.cropvolume.logic().Apply(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

        # Resample
        resampleParameters = {
            'inputVolume': inputVolumeNode.GetID(),
            'outputVolume': patchVolume.GetID(),
            'referenceVolume': patchVolume.GetID(),
            'interpolationType': 'ws',
            'windowFunction': 'l'
        }
        cliNode = slicer.cli.runSync(
            slicer.modules.resamplescalarvectordwivolume, None, resampleParameters
        )
        slicer.mrmlScene.RemoveNode(cliNode)

        # Preprocess Resample Volume
        import SimpleITK as sitk
        import sitkUtils as su

        histogramFilter = sitk.AdaptiveHistogramEqualizationImageFilter()
        histogramFilter.SetAlpha(HIST_ALPHA)
        histogramFilter.SetBeta(HIST_BETA)
        histogramFilter.SetRadius(HIST_RADIUS)

        sitkImage = su.PullVolumeFromSlicer(patchVolume)
        sitkImage = histogramFilter.Execute(sitkImage)
        sitkImage *= -1
        su.PushVolumeToSlicer(sitkImage, patchVolume)

        return patchVolume

    def createPiecesIndices(self):
        return

    @staticmethod
    def getIJKFromRAS(referenceVolume, rasArray):
        """
        :param rasArray: Nx3 numpy array
        """
        x = vtk.vtkMatrix4x4()
        referenceVolume.GetRASToIJKMatrix(x)
        ras2ijkMat = slicer.util.arrayFromVTKMatrix(x)
        rasHomo = np.ones((rasArray.shape[0], 4), dtype=np.double)
        rasHomo[:, :3] = rasArray
        return (np.round(ras2ijkMat @ rasHomo.T).astype(np.uint16).T)[:, :3]

    def handleReportTable(self, rowData, reportTableNode, markersName):
        columnName = [
            'name',
            'length',
            'min-area',
            'max-area',
            'stenosis',
            'min-area',
            'max-area1',
            'max-area2',
            'normal-reference',
            'stenosis-formula0',
            'minidx',
            'maxidx',
            'max1idx',
            'max2id',
        ]

        toAddData = [markersName] + rowData
        # if table is empty
        if not reportTableNode.GetNumberOfRows():
            slicer.util.updateTableFromArray(
                reportTableNode, np.zeros((1, len(toAddData))), columnName
            )

        newRowIndex = reportTableNode.AddEmptyRow()
        rowVtk = vtk.vtkVariantArray()
        for cell in toAddData:
            rowVtk.InsertNextValue(vtk.vtkVariant(cell))
        reportTableNode.GetTable().SetRow(newRowIndex, rowVtk)

    def getFirstAreaDecline(self, array):
        # (N,) array
        arraySize = len(array)
        for i in range(1, arraySize):
            prev = array[i-1]
            val = array[i]
            if val < prev:
                return i
        return 0

    def createStenosisReport(self, tableNode):

        cross_sec_area = slicer.util.arrayFromTableColumn(
            tableNode, 'Cross-section area'
        )
        distances = slicer.util.arrayFromTableColumn(tableNode, 'Distance')

        # Fix constant for now
        max_distances = distances[-1]
        if len(distances) < 20 or (len(distances) and max_distances < 6):
            e = 'Vessel too short'
            raise Exception(e)

        left_trim_index = self.getFirstAreaDecline(cross_sec_area)
        right_trim_exclude_index = len(cross_sec_area) - self.getFirstAreaDecline(cross_sec_area[::-1])
        trim_cross_sec_area = cross_sec_area[left_trim_index:right_trim_exclude_index]

        # min_area_index = np.argmin(cross_sec_area)
        min_area_index = np.argmin(trim_cross_sec_area) + left_trim_index
        min_area = cross_sec_area[min_area_index]
        max_area_index = np.argmax(cross_sec_area)
        max_area = cross_sec_area[max_area_index]  # avg( max(proximal), max(distal) )

        max_1_area = max_area
        max_1_area_index = None
        # For case min_area_index is at edge
        left_to_min = cross_sec_area[:min_area_index]
        if len(left_to_min):
            max_1_area_index = np.argmax(left_to_min)
            max_1_area = cross_sec_area[max_1_area_index]


        max_2_area = max_area
        max_2_area_index = None
        # For case min_area_index is at edge
        right_to_min = cross_sec_area[min_area_index:]
        if len(right_to_min):
            max_2_area_index = np.argmax(right_to_min) + min_area_index
            max_2_area = cross_sec_area[max_2_area_index]


        normal_reference = (max_1_area + max_2_area) * 0.5

        if max_area and normal_reference:
            stenosis = 1 - (min_area / max_area)
            stenosis_f0 = 1 - (min_area / normal_reference)

            stenosis_percent_str = f'{stenosis*100:.3f}'
            stenosis_f0_percent_str = f'{stenosis_f0*100:.3f}'

            rowData = [
                distances,
                min_area,
                max_area,
                stenosis,
                min_area,
                max_1_area,
                max_2_area,
                normal_reference,
                stenosis_f0,
                min_area_index,
                max_area_index,
                max_1_area_index,
                max_2_area_index
            ]

            outInfo = 'Result!\n'
            outInfo += '\n'
            # outInfo += f'vessel length: {max_distances=:.3f} mm. trim with {trim_length} mm. both side'
            outInfo += f'vessel length: {max_distances=:.3f} mm.'
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
            return outInfo, rowData

        raise Exception('max_area is 0')


    def process(
        self,
        inputVolumeNode: vtkMRMLScalarVolumeNode,
        costVolumeNode: vtkMRMLScalarVolumeNode,
        markersNode: vtkMRMLMarkupsFiducialNode,
        guideLineNode: vtkMRMLMarkupsCurveNode,
        segmentationNode: vtkMRMLSegmentationNode,
        reportTableNode: vtkMRMLTableNode
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

        from BVStenosisMeasurementLib.Timer import Timer

        timer = Timer(isDebug=True)
        logging.info('Processing started')

        # Commonly used node
        mainWindow = slicer.util.mainWindow()
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        markersName = markersNode.GetName()

        #
        # Segmentation
        #

        # Create patch volume
        timer.start('Create Patch ROI')
        patchROINode = self.createPatchROI(costVolumeNode, guideLineNode)
        timer.stop()

        timer.start('Create Patch Volume')
        patchVolumeNode = self.createPatchVolume(inputVolumeNode, patchROINode)
        patchVolumeNode.SetName("BV_PatchVolume")
        timer.stop()

        # Create vesselness volume
        timer.start('Create Vesselness volume')

        guideSeedControlPoints = slicer.util.arrayFromMarkupsControlPoints(guideLineNode)

        guideSeedNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        guideSeedNode.SetName("BV_GuideSeed")
        slicer.util.updateMarkupsControlPointsFromArray(guideSeedNode, guideSeedControlPoints)

        vesselnessVolumeNode = (
            Controllers.VesselnessFilteringController.createVesselnessVolume(
                patchVolumeNode,
                guideSeedNode,
                minDiameterMM=None,
                maxDiameterMM=None,
                contrast=None,
                suppressPlate=10,
                suppressBlob=10,
                lowerThreshold=0.1,
                isCalculateParameter=True,
            )
        )
        vesselnessVolumeNode.SetName("BV_Vesselness")
        timer.stop()

        # Create Segmentation Labelmap
        SEG_VESSELNESS_MIN = 0.05

        timer.start('Segmentation')
        segController = Controllers.LevelSetSegmentationController(
            patchVolumeNode, vesselnessVolumeNode
        )


        # generate guideLine piece
        # piece is coninuous chunk of guideLine which all have vesselness value morethan threshold
        # TODO: move end of piece to furthest position

        piecesIndices = []
        isPieceClose = True
        guideSeedIJK = self.getIJKFromRAS(patchVolumeNode, guideSeedControlPoints)
        vesselnessArray = slicer.util.arrayFromVolume(vesselnessVolumeNode)

        for i in range(len(guideSeedControlPoints)):
            seedI, seedJ, seedK = guideSeedIJK[i, :].tolist()
            vesselnessValue = vesselnessArray[seedK,seedJ,seedI]
            if isPieceClose:
                if vesselnessValue > SEG_VESSELNESS_MIN:
                    isPieceClose = False
                    piecesIndices.append([i])
            else:
                if vesselnessValue > SEG_VESSELNESS_MIN:
                    piecesIndices[-1].append(i)
                else:
                    isPieceClose = True
        logging.debug(f'{piecesIndices=}')
        print(f'number of pieces: {len(piecesIndices)}')

        tmpSeedsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        tmpStoppersNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')

        # do segment on each piece
        for pieceIndices in piecesIndices:
            startSeedIndex = pieceIndices[0]
            stopSeedIndex = pieceIndices[-1]
            slicer.util.updateMarkupsControlPointsFromArray(
                tmpSeedsNode, guideSeedControlPoints[(startSeedIndex, stopSeedIndex), :]
            )
            slicer.util.updateMarkupsControlPointsFromArray(
                tmpStoppersNode, guideSeedControlPoints[(stopSeedIndex,), :]
            )

            segController.performEvolution(
                tmpSeedsNode,
                tmpStoppersNode,
                minVesselnessThreshold=SEG_VESSELNESS_MIN,
                iteration=15,
                inflation=0,
                curvature=70,
                attraction=50,
                method=segController.methodCollidingFronts,
                levelSetsType=segController.levelSetsTypeCurves,
            )

        slicer.mrmlScene.RemoveNode(tmpSeedsNode)
        slicer.mrmlScene.RemoveNode(tmpStoppersNode)

        labelMapNode = segController.createResultLabelMapNode()
        # output label map from levelset is float, we need discrete type
        slicer.cli.runSync(slicer.modules.castscalarvolume, None, {
            'InputVolume': labelMapNode.GetID(),
            'OutputVolume': labelMapNode.GetID(),
            'Type': 'Short'
        })

        # Specify geometry, should effect labelmap representation
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(patchVolumeNode)

        # Initialize segmentation tools
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        # This should effect only intensity query
        segmentEditorWidget.setSourceVolumeNode(inputVolumeNode)

        # Convert label map to segmentation

        vesselSegmentName = f'{markersName}'
        # Replace old segmentation
        oldSegmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(vesselSegmentName)
        if oldSegmentID:
            segmentationNode.RemoveSegment(oldSegmentID)

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelMapNode, segmentationNode
        )
        vesselSegmentID = segmentationNode.GetSegmentation().GetSegmentIDs()[-1]
        segmentationNode.GetSegmentation().GetSegment(vesselSegmentID).SetName(vesselSegmentName)

        timer.stop()

        # do padding on fragmented blood vessel
        # enum reference
        # https://github.com/SlicerIGT/SlicerMarkupsToModel/blob/master/MarkupsToModel/MRML/vtkMRMLMarkupsToModelNode.h
        timer.start('Pad segmentation')

        paddingCoreModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        slicer.modules.markupstomodel.logic().UpdateOutputCurveModel(
            guideLineNode,
            paddingCoreModelNode,
            3,  # Linear, CardinalSpline, KochanekSpline, Polynomial
            False,  # tube loop
            0.4,  # tube Radius
            8,  # tubeNumber of sides
            1,  # tube segment between control point
            True,  # clean markups ?
            3,  # polynomial order
            0,  # point parameter type
            None,
            1,  # global least square, moving least square
            0.12,  # sample width
            3,  # weight: Rectangular = 0, Triangular, Cosine, Gaussian,
            True,
        )
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(
            paddingCoreModelNode, segmentationNode
        )
        paddingCoreSegmentID = segmentationNode.GetSegmentation().GetSegmentIDs()[-1]
        slicer.mrmlScene.RemoveNode(paddingCoreModelNode)

        paddedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment()

        segmentEditorNode.SetSelectedSegmentID(paddedSegmentID)
        segmentEditorNode.SetOverwriteMode(2) # Allow overlap
        segmentEditorWidget.setActiveEffectByName('Logical operators')
        effect = segmentEditorWidget.activeEffect()
        # copy core of guideline model
        effect.setParameter('Operation', 'COPY')
        effect.setParameter('ModifierSegmentID', paddingCoreSegmentID)
        effect.self().onApply()
        # add vessel pieces segmentation to core
        effect.setParameter('Operation', 'UNION')
        effect.setParameter('ModifierSegmentID', vesselSegmentID)
        effect.self().onApply()
        # Remove (small) islands
        segmentEditorWidget.setActiveEffectByName('Islands')
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter('Operation', 'KEEP_LARGEST_ISLAND')
        effect.self().onApply()

        # no use anymore
        segmentationNode.RemoveSegment(paddingCoreSegmentID)

        timer.stop()

        #
        # --- Extract Centerline
        #
        timer.start('Extract Centerline')

        mainWindow.moduleSelector().selectModule('ExtractCenterline')
        slicer.app.processEvents()

        ecWidget = slicer.modules.extractcenterline.widgetRepresentation().self()
        # ecLogic = ecWidget.logic

        ecWidget.ui.inputSurfaceSelector.setCurrentNode(segmentationNode)
        ecWidget.ui.inputSegmentSelectorWidget.setCurrentSegmentID(paddedSegmentID)

        endPointsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        endPointsNode.SetName('BV_EndPoints')

        startPoint = guideSeedControlPoints[0]
        stopPoint = guideSeedControlPoints[-1]
        endPointsNode.RemoveAllControlPoints()
        endPointsNode.AddControlPoint(startPoint[0], startPoint[1], startPoint[2])
        endPointsNode.AddControlPoint(stopPoint[0], stopPoint[1], stopPoint[2])
        ecWidget.ui.endPointsMarkupsSelector.setCurrentNode(endPointsNode)
        # ecWidget.ui.curveSamplingDistanceSpinBox.setValue(1.0)
        # centerlineModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        # centerlineModelNode.SetName(f'Centerline_model_{i}')
        # ecWidget.ui.outputCenterlineModelSelector.setCurrentNode(centerlineModelNode)

        centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsCurveNode'
        )
        centerlineCurveNode.SetName(f'{markersName}_Centerline')
        ecWidget.ui.outputCenterlineCurveSelector.setCurrentNode(centerlineCurveNode)
        ecWidget.ui.preprocessInputSurfaceModelCheckBox.setChecked(False)

        # TODO: None Type Object have no attribute getvalue/ some times parameter node is gone
        ecWidget.onAutoDetectEndPoints()
        slicer.app.processEvents()

        ecWidget.onApplyButton()


        #
        # --- Cross Sectional Analysis
        #

        mainWindow.moduleSelector().selectModule('CrossSectionAnalysis')
        slicer.app.processEvents()

        # TODO: make this reapply able -> check how exportvisiblesegmentstomodel logic work
        # exportFolderItemId = shNode.CreateFolderItem(
        #     shNode.GetSceneItemID(), 'Segments_' + guideLineNode.GetID()
        # )
        # slicer.modules.segmentations.logic().ExportSegmentsToModels(
        #     segmentationNode, [segmentID], exportFolderItemId
        # )
        # segmentModels = vtk.vtkCollection()
        # shNode.GetDataNodesInBranch(exportFolderItemId, segmentModels)
        # Get exported model of first segment
        # modelNode = segmentModels.GetItemAsObject(0)

        crossSecWidget = (
            slicer.modules.crosssectionanalysis.widgetRepresentation().self()
        )
        crossSecLogic = crossSecWidget.logic

        crossSecWidget.ui.inputCenterlineSelector.setCurrentNode(centerlineCurveNode)
        # crossSecWidget.ui.segmentationSelector.setCurrentNode(modelNode)
        crossSecWidget.ui.segmentationSelector.setCurrentNode(segmentationNode)
        crossSecWidget.onInputSegmentationNode()
        slicer.app.processEvents()
        crossSecWidget.ui.segmentSelector.setCurrentSegmentID(vesselSegmentID)

        crossSecWidget.onApplyButton()

        redSliceNode = slicer.app.layoutManager().sliceWidget('Red').mrmlSliceNode()
        greenSliceNode = slicer.app.layoutManager().sliceWidget('Green').mrmlSliceNode()
        crossSecWidget.ui.axialSliceViewSelector.setCurrentNode(redSliceNode)
        crossSecWidget.ui.longitudinalSliceViewSelector.setCurrentNode(greenSliceNode)

        #
        # Clean up
        #

        slicer.mrmlScene.RemoveNode(patchROINode)
        slicer.mrmlScene.RemoveNode(guideSeedNode)
        slicer.mrmlScene.RemoveNode(segmentEditorNode)
        slicer.mrmlScene.RemoveNode(endPointsNode)
        del segmentEditorWidget
        # TODO: bring back when finish
        segmentationNode.RemoveSegment(paddedSegmentID)
        slicer.mrmlScene.RemoveNode(vesselnessVolumeNode)
        slicer.mrmlScene.RemoveNode(labelMapNode)
        slicer.mrmlScene.RemoveNode(patchVolumeNode)
        slicer.util.setSliceViewerLayers(
            background=inputVolumeNode,
            foreground=vesselnessVolumeNode,
            label=labelMapNode,
        )

        outTableNode = crossSecWidget.ui.outputTableSelector.currentNode()
        outInfo, rowData = self.createStenosisReport(outTableNode)
        slicer.util.infoDisplay(outInfo, self.moduleName)
        # self.handleReportTable(rowData, reportTableNode, markersName)


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
        lcxGLNode = slicer.util.getNode('GL_LCX')
        rcaGLNode = slicer.util.getNode('GL_RCA')
        # markers = slicer.util.getNode('SEED_SPARSE_RCA')
        markers = slicer.util.getNode('SEED_SPARSE_LCX')

        widget = slicer.modules.bvstenosismeasurement.widgetRepresentation().self()
        widget.ui.inputVolumeSelector.setCurrentNode(inputVolumeNode)
        widget.ui.costVolumeSelector.setCurrentNode(costVolumeNode)
        widget.ui.guideLineSelector.setCurrentNode(lcxGLNode)
        widget.ui.markersSelector.setCurrentNode(markers)

        widget.onApplyButtonClicked()



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
