import unittest
from importlib import reload

import ctk
import numpy as np
import qt
import slicer
import slicer.util
import vtk
from slicer.parameterNodeWrapper import WithinRange, parameterNodeWrapper
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
    logging,
)
from slicer.util import VTKObservationMixin

# import StenosisCalLib
# import StenosisCalLib.MainLogic as MainLogic
from StenosisCalLib import MainLogic, MyFunc

reload(MainLogic)

#
# StenosisCal
#

StenosisCal_VERSION = 'v0'


class StenosisCal(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = 'Stenosis Calculator'
        self.parent.categories = ['Artery Stenosis']
        self.parent.dependencies = ['ExtractCenterline', 'ExtractCenterline']
        self.parent.contributors = ['Krit Cholapand (Chulalongkorn U.)']
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = f"""
Version: {StenosisCal_VERSION}
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)


#
# StenosisCalWidget
#


class StenosisCalWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._version = StenosisCal_VERSION

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        logging.info(f'Stenosis Calculator Version: {StenosisCal_VERSION}')

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/StenosisCal.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = StenosisCalLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.arterySeedSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI
        )
        self.ui.inputVolumeSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI
        )
        self.ui.guidelineSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI
        )
        self.ui.segmentSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI
        )
        # self.ui.outputSelector.connect(
        #     'currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI
        # )
        # self.ui.imageThresholdSliderWidget.connect(
        #     'valueChanged(double)', self.updateParameterNodeFromGUI
        # )
        # self.ui.invertOutputCheckBox.connect(
        #     'toggled(bool)', self.updateParameterNodeFromGUI
        # )
        # self.ui.invertedOutputSelector.connect(
        #     'currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI
        # )

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(
            self._parameterNode,
            vtk.vtkCommand.ModifiedEvent,
            self.updateGUIFromParameterNode,
        )

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference('InputVolume'):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass(
                'vtkMRMLScalarVolumeNode'
            )
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID(
                    'InputVolume', firstVolumeNode.GetID()
                )
        if not self._parameterNode.GetNodeReference('ArterySeed'):
            firstNode = slicer.mrmlScene.GetFirstNodeByClass(
                'vtkMRMLMarkupsFiducialNode'
            )
            if firstNode:
                self._parameterNode.SetNodeReferenceID('ArterySeed', firstNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputVolumeSelector.setCurrentNode(
            self._parameterNode.GetNodeReference('InputVolume')
        )
        self.ui.arterySeedSelector.setCurrentNode(
            self._parameterNode.GetNodeReference('ArterySeed')
        )
        self.ui.guidelineSelector.setCurrentNode(
            self._parameterNode.GetNodeReference('Guideline')
        )
        self.ui.segmentSelector.setCurrentNode(
            self._parameterNode.GetNodeReference('Segment')
        )

        """ self.ui.invertedOutputSelector.setCurrentNode(
            self._parameterNode.GetNodeReference('OutputVolumeInverse')
        )
        self.ui.imageThresholdSliderWidget.value = float(
            self._parameterNode.GetParameter('Threshold')
        )
        self.ui.invertOutputCheckBox.checked = (
            self._parameterNode.GetParameter('Invert') == 'true'
        ) """

        # Update buttons states and tooltips
        if (
            self._parameterNode.GetNodeReference('InputVolume')
            and self._parameterNode.GetNodeReference('ArterySeed')
            and self._parameterNode.GetNodeReference('Guideline')
            and self._parameterNode.GetNodeReference('Segment')
        ):
            self.ui.applyButton.toolTip = 'Compute Stenosis'
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = 'Select input and output volume nodes'
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = (
            self._parameterNode.StartModify()
        )  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID(
            'ArterySeed', self.ui.arterySeedSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            'InputVolume', self.ui.inputVolumeSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            'Guideline', self.ui.guidelineSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            'Segment', self.ui.segmentSelector.currentNodeID
        )
        # self._parameterNode.SetParameter(
        #     'Threshold', str(self.ui.imageThresholdSliderWidget.value)
        # )
        # self._parameterNode.SetParameter(
        #     'Invert', 'true' if self.ui.invertOutputCheckBox.checked else 'false'
        # )
        # self._parameterNode.SetNodeReferenceID(
        #     'OutputVolumeInverse', self.ui.invertedOutputSelector.currentNodeID
        # )

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        try:
            # Compute output
            self.logic.process(
                self.ui.arterySeedSelector.currentNode(),
                self.ui.inputVolumeSelector.currentNode(),
                self.ui.guidelineSelector.currentNode(),
                self.ui.segmentSelector.currentNode(),
            )

        except Exception as e:
            slicer.util.errorDisplay('Failed to compute results: ' + str(e))
            import traceback

            traceback.print_exc()


#
# StenosisCalLogic
#

class StenosisCalECParameterNode:
    def __init__(self):

        # These do not have widget coutnerparts.
        self.outputFiducialNode = None
          # 'Extract centerline' endpoints
        self.outputCenterlineModel = None
        self.outputCenterlineCurve = None

        # CSA
        self.outputPlot = None
        self.outputTable = None

class StenosisCalLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    """
  Bundle Stenosis Calculation
  - Guided Line computation
  - Segmentation
  - VMTK: Centerline
  - VMTK: Stenosis Visualization
  """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.logger = logging.getLogger('stenosiscal_logger')
        # inspiration from https://github.com/vmtk/SlicerExtension-VMTK/blob/master/GuidedArterySegmentation/GuidedArterySegmentation.py
        self.initMemberVariables()

    def initMemberVariables(self):
        self._parameterNode = StenosisCalECParameterNode()
        self._segmentEditorWidgets = None
        self._extractCenterlineWidgets = None
        self._csaWidgets = None

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter('Threshold'):
            parameterNode.SetParameter('Threshold', '100.0')
        if not parameterNode.GetParameter('Invert'):
            parameterNode.SetParameter('Invert', 'false')

    def process(
        self,
        arterySeed,
        inputVolume,
        guideline: slicer.vtkMRMLMarkupsCurveNode,
        segment,
    ):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """
        import time
        startTime = time.time()
        # print(arterySeed)
        # print(inputVolume)
        
        
        weight_arr = slicer.util.arrayFromVolume(inputVolume).copy()
        COST_OFFSET = -50
        weight_arr[weight_arr > COST_OFFSET] = 2_000_000_000
        weight_arr[weight_arr <= COST_OFFSET] += 4000
        seed = (204, 160, 307)
        end_node = (190, 159, 329)
        ijk_spacing = inputVolume.GetSpacing()
        kernel_ras_distance = np.zeros((3, 3, 3), dtype=np.float64)
        for pos_k in range(-1, 2):
            for pos_j in range(-1, 2):
                for pos_i in range(-1, 2):
                    kernel_ras_distance[pos_k + 1, pos_j + 1, pos_i + 1] = (
                        (pos_i * ijk_spacing[0]) ** 2
                        + (pos_j * ijk_spacing[1]) ** 2
                        + (pos_k * ijk_spacing[2]) ** 2
                    ) ** 0.5
        sub_path, _ = MyFunc.dijkstra(seed, end_node, weight_arr, kernel_ras_distance)
        

        # segmentation, segmentID = MainLogic.MyProcess(
        #     inputVolume, arterySeed, guideline, segment
        # )

        # x = vtk.vtkMatrix4x4()
        # inputVolume.GetRASToIJKMatrix(x)
        # ras2ijk_mat = MainLogic.vtk4x4matrix_to_numpy(x)

        # num_fid = arterySeed.GetNumberOfControlPoints()
        # control_points_label2kji_map = {
        #     arterySeed.GetNthControlPointLabel(i): np.array(
        #         MainLogic.get_fiducial_as_kji(arterySeed, i, ras2ijk_mat)
        #     )
        #     for i in range(num_fid)
        # }
        # print(control_points_label2kji_map)
        # print(segmentID)

        print('time take:', time.time() - startTime)

        mainWindow = slicer.util.mainWindow()
        segmentation, segmentID = StenosisCalLib.MainLogic.MyProcess(
            inputVolume, arterySeed, guideline, segment
        )

        # ---------- Extract centerlines ----------
        slicer.app.processEvents()
        mainWindow.moduleSelector().selectModule('ExtractCenterline')
        if not self._extractCenterlineWidgets:
            self._extractCenterlineWidgets = ExtractCenterlineWidgets()
            self._extractCenterlineWidgets.findWidgets()

        inputSurfaceComboBox = self._extractCenterlineWidgets.inputSurfaceComboBox
        inputSegmentSelectorWidget = (
            self._extractCenterlineWidgets.inputSegmentSelectorWidget
        )
        endPointsMarkupsSelector = (
            self._extractCenterlineWidgets.endPointsMarkupsSelector
        )
        outputCenterlineModelSelector = (
            self._extractCenterlineWidgets.outputCenterlineModelSelector
        )
        outputCenterlineCurveSelector = (
            self._extractCenterlineWidgets.outputCenterlineCurveSelector
        )
        preprocessInputSurfaceModelCheckBox = (
            self._extractCenterlineWidgets.preprocessInputSurfaceModelCheckBox
        )
        applyButton = self._extractCenterlineWidgets.applyButton

        # Set input segmentation
        inputSurfaceComboBox.setCurrentNode(segmentation)
        inputSegmentSelectorWidget.setCurrentSegmentID(segmentID)
        # Create 2 fiducial endpoints, at start and end of input curve. We call it output because it is not user input.
        outputFiducialNode = self._parameterNode.outputFiducialNode
        if not outputFiducialNode:
            outputFiducialNode = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode'
            )
            # Visually identify the segment by the input fiducial name
            outputFiducialNode.SetName('Endpoints_' + guideline.GetName())
            firstInputCurveControlPoint = guideline.GetNthControlPointPositionVector(0)
            outputFiducialNode.AddControlPointWorld(firstInputCurveControlPoint)
            endPointsMarkupsSelector.setCurrentNode(outputFiducialNode)
            curveControlPoints = vtk.vtkPoints()
            guideline.GetControlPointPositionsWorld(curveControlPoints)
            lastInputCurveControlPoint = guideline.GetNthControlPointPositionVector(
                curveControlPoints.GetNumberOfPoints() - 1
            )
            outputFiducialNode.AddControlPointWorld(lastInputCurveControlPoint)
            endPointsMarkupsSelector.setCurrentNode(outputFiducialNode)
            self._parameterNode.outputFiducialNode = outputFiducialNode
        # Account for rename. Control points are not remaned though.
        outputFiducialNode.SetName('Endpoints_' + guideline.GetName())
        # Output centerline model. A single node throughout.

        centerlineModel = self._parameterNode.outputCenterlineModel
        if not centerlineModel:
            centerlineModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
            # Visually identify the segment by the input fiducial name
            centerlineModel.SetName(
                'Centerline_model_' + guideline.GetName()
            )
            self._parameterNode.outputCenterlineModel = centerlineModel
        # Account for rename
        centerlineModel.SetName(
            'Centerline_model_' + guideline.GetName()
        )
        outputCenterlineModelSelector.setCurrentNode(centerlineModel)

        # Output centerline curve. A single node throughout.
        centerlineCurve = self._parameterNode.outputCenterlineCurve
        if not centerlineCurve:
            centerlineCurve = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsCurveNode'
            )
            # Visually identify the segment by the input fiducial name
            centerlineCurve.SetName(
                'Centerline_curve_' + guideline.GetName()
            )
            self._parameterNode.outputCenterlineCurve = centerlineCurve
        # Account for rename
        centerlineCurve.SetName(
            'Centerline_curve_' + guideline.GetName()
        )
        # ? extractcenterline not make intuitive function to run in background so we have to switch back to our ui
        # TODO: switch back to our UI

        outputCenterlineCurveSelector.setCurrentNode(centerlineCurve)
        """
        Don't preprocess input surface. Decimation error may crash Slicer. Quadric method for decimation is slower but more reliable.
        """
        preprocessInputSurfaceModelCheckBox.setChecked(False)
        # Apply
        applyButton.click()
        # Hide the input curve to show the centerlines
        guideline.SetDisplayVisibility(False)
        # Close network pane; we don't use this here.
        self._extractCenterlineWidgets.outputNetworkGroupBox.collapsed = True

        stopTime = time.time()
        durationValue = '%.2f' % (stopTime - startTime)
        message = f'Processing completed in {durationValue} seconds'
        logging.info(message)
        slicer.util.showStatusMessage(message, 5000)

        #---------- Cross-Section Analysis ----------
        slicer.app.processEvents()
        mainWindow.moduleSelector().selectModule('CrossSectionAnalysis')
        slicer.app.processEvents()

        if not self._csaWidgets:
            self._csaWidgets = CrossSectionAnalysisWidgets()
            self._csaWidgets.findWidgets()

        applyButton = self._csaWidgets.applyButton
        inputCenterlineSelector = self._csaWidgets.inputCenterlineSelector
        inputSurfaceComboBox = self._csaWidgets.inputSurfaceComboBox
        inputSegmentSelectorWidget = self._csaWidgets.inputSegmentSelectorWidget
        outputTableSelector = self._csaWidgets.outputTableSelector
        outputPlotSelector = self._csaWidgets.outputPlotSelector

        inputCenterlineSelector.setCurrentNode(centerlineCurve)
        inputSurfaceComboBox.setCurrentNode(segmentation)
        inputSegmentSelectorWidget.setCurrentSegmentID(segmentID)

        if not self._parameterNode.outputTable:
            self._parameterNode.outputTable = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLTableNode'
            )
            # Visually identify the segment by the input fiducial name
            self._parameterNode.outputTable.SetName('MyTable ' + guideline.GetName())
        outputTableSelector.setCurrentNode(self._parameterNode.outputTable)

        if not self._parameterNode.outputPlot:
            self._parameterNode.outputPlot = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLPlotSeriesNode'
            )
            self._parameterNode.outputPlot.SetName('MyPlot ' + guideline.GetName())
        outputPlotSelector.setCurrentNode(self._parameterNode.outputPlot)

        applyButton.click()

        slicer.app.processEvents()
        # TODO: properly calculate stenosis
        cross_sec_area = slicer.util.arrayFromTableColumn(self._parameterNode.outputTable, 'Cross-section area')
        min_area = np.min(cross_sec_area)
        max_area = np.max(cross_sec_area) # avg( max(proximal), max(distal) )
        stenosis = 1 - (min_area / max_area)
        print('min, max, stenosis')
        print(f'{min_area}, {max_area}, {stenosis*100:.3f}%')



#
# StenosisCalTest
#


class StenosisCalTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_StenosisCal1()

    def test_StenosisCal1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        return  # TODO: ignore test for now
        # self.delayDisplay("Starting the test")

        # # Get/create input data

        # import SampleData
        # registerSampleData()
        # inputVolume = SampleData.downloadSample('StenosisCal1')
        # self.delayDisplay('Loaded test data set')

        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100

        # # Test the module logic

        # logic = StenosisCalLogic()

        # # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)

        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        # self.delayDisplay('Test passed')


class ExtractCenterlineWidgets(ScriptedLoadableModule):
    def __init__(self) -> None:
        self.mainContainer = None
        self.inputCollapsibleButton = None
        self.outputCollapsibleButton = None
        self.advancedCollapsibleButton = None
        self.applyButton = None
        self.inputSurfaceComboBox = None
        self.endPointsMarkupsSelector = None
        self.inputSegmentSelectorWidget = None
        self.outputNetworkGroupBox = None
        self.outputTreeGroupBox = None
        self.outputCenterlineModelSelector = None
        self.outputCenterlineCurveSelector = None
        self.preprocessInputSurfaceModelCheckBox = None

    def findWidgets(self) -> None:
        ecWidgetRepresentation = slicer.modules.extractcenterline.widgetRepresentation()

        # Containers
        self.mainContainer = ecWidgetRepresentation.findChild(
            slicer.qMRMLWidget, 'ExtractCenterline'
        )
        self.inputCollapsibleButton = self.mainContainer.findChild(
            ctk.ctkCollapsibleButton, 'inputsCollapsibleButton'
        )
        self.outputCollapsibleButton = self.mainContainer.findChild(
            ctk.ctkCollapsibleButton, 'outputsCollapsibleButton'
        )
        self.advancedCollapsibleButton = self.mainContainer.findChild(
            ctk.ctkCollapsibleButton, 'advancedCollapsibleButton'
        )
        self.applyButton = self.mainContainer.findChild(qt.QPushButton, 'applyButton')

        # Input widgets
        self.inputSurfaceComboBox = self.inputCollapsibleButton.findChild(
            slicer.qMRMLNodeComboBox, 'inputSurfaceSelector'
        )
        self.endPointsMarkupsSelector = self.inputCollapsibleButton.findChild(
            slicer.qMRMLNodeComboBox, 'endPointsMarkupsSelector'
        )
        self.inputSegmentSelectorWidget = self.inputCollapsibleButton.findChild(
            slicer.qMRMLSegmentSelectorWidget, 'inputSegmentSelectorWidget'
        )

        # Output widgets
        self.outputNetworkGroupBox = self.outputCollapsibleButton.findChild(
            ctk.ctkCollapsibleGroupBox, 'CollapsibleGroupBox'
        )
        self.outputTreeGroupBox = self.outputCollapsibleButton.findChild(
            ctk.ctkCollapsibleGroupBox, 'CollapsibleGroupBox_2'
        )
        self.outputCenterlineModelSelector = self.outputTreeGroupBox.findChild(
            slicer.qMRMLNodeComboBox, 'outputCenterlineModelSelector'
        )
        self.outputCenterlineCurveSelector = self.outputTreeGroupBox.findChild(
            slicer.qMRMLNodeComboBox, 'outputCenterlineCurveSelector'
        )

        # Advanced widgets
        self.preprocessInputSurfaceModelCheckBox = (
            self.advancedCollapsibleButton.findChild(
                qt.QCheckBox, 'preprocessInputSurfaceModelCheckBox'
            )
        )

class CrossSectionAnalysisWidgets(ScriptedLoadableModule):
    def __init__(self) -> None:
        self.mainContainer = None
        self.parametersCollapsibleButton = None
        self.BrowseCollapsibleButton = None
        self.applyButton = None
        self.inputCenterlineSelector = None
        self.inputSurfaceComboBox = None
        self.inputSegmentSelectorWidget = None
        self.outputTableSelector = None
        self.outputPlotSelector = None

    def findWidgets(self) -> None:
        csaWidgetRepresentation = slicer.modules.crosssectionanalysis.widgetRepresentation()

        # Containers
        self.mainContainer = csaWidgetRepresentation.findChild(
            slicer.qMRMLWidget, 'CrossSectionAnalysis'
        )
        self.parametersCollapsibleButton = self.mainContainer.findChild(
            ctk.ctkCollapsibleButton, 'parametersCollapsibleButton'
        )
        self.BrowseCollapsibleButton = self.mainContainer.findChild(
            ctk.ctkCollapsibleButton, 'browseCollapsibleButton'
        )

        # parameters widgets
        self.applyButton = self.parametersCollapsibleButton.findChild(qt.QPushButton, 'applyButton')
        self.inputCenterlineSelector = self.parametersCollapsibleButton.findChild(
            slicer.qMRMLNodeComboBox, 'inputCenterlineSelector'
        )
        self.inputSurfaceComboBox = self.parametersCollapsibleButton.findChild(
            slicer.qMRMLNodeComboBox, 'segmentationSelector'
        )
        self.inputSegmentSelectorWidget = self.parametersCollapsibleButton.findChild(
            slicer.qMRMLSegmentSelectorWidget, 'segmentSelector'
        )
        self.outputTableSelector = self.parametersCollapsibleButton.findChild(
            slicer.qMRMLNodeComboBox, 'outputTableSelector'
        )
        self.outputPlotSelector = self.parametersCollapsibleButton.findChild(
            slicer.qMRMLNodeComboBox, 'outputPlotSeriesSelector'
        )

        # # Output widgets
        # self.outputNetworkGroupBox = self.outputCollapsibleButton.findChild(
        #     ctk.ctkCollapsibleGroupBox, 'CollapsibleGroupBox'
        # )
        # self.outputTreeGroupBox = self.outputCollapsibleButton.findChild(
        #     ctk.ctkCollapsibleGroupBox, 'CollapsibleGroupBox_2'
        # )
        # self.outputCenterlineModelSelector = self.outputTreeGroupBox.findChild(
        #     slicer.qMRMLNodeComboBox, 'outputCenterlineModelSelector'
        # )
        # self.outputCenterlineCurveSelector = self.outputTreeGroupBox.findChild(
        #     slicer.qMRMLNodeComboBox, 'outputCenterlineCurveSelector'
        # )

        # # Advanced widgets
        # self.preprocessInputSurfaceModelCheckBox = (
        #     self.advancedCollapsibleButton.findChild(
        #         qt.QCheckBox, 'preprocessInputSurfaceModelCheckBox'
        #     )
        # )
