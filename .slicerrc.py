# Python commands in this file are executed on Slicer startup
import slicer.util

# Examples:
#
# Load a scene file
# slicer.util.loadScene('c:/Users/SomeUser/Documents/SlicerScenes/SomeScene.mrb')
#
# Open a module (overrides default startup module in application settings / modules)
# slicer.util.mainWindow().moduleSelector().selectModule('SegmentEditor')
#

# slicer.util.loadScene( '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/2023-10-26-Scene.mrb')

# auto preprocess data

# slicer.util.mainWindow().moduleSelector().selectModule('BVPreprocessVolume')
# slicer.util.mainWindow().moduleSelector().selectModule('BVCreateGuideLine')

class mytest:
    @staticmethod
    def testLAD():
        return True

class myutil:
    @staticmethod
    def getLogic():
        return slicer.modules.bvstenosismeasurement.widgetRepresentation().self().logic

    @staticmethod
    def getWidget():
        return slicer.modules.bvstenosismeasurement.widgetRepresentation()

    @staticmethod
    def getParameterNodeID():
        logic = myutil.getLogic()
        return logic.parameterNode.GetID()

    @staticmethod
    def clearConsole():
        slicer.app.pythonConsole().clear()

    @staticmethod
    def autorun():
        volumeNode = getNode('14: Body Soft Tissue')
        roiNode = slicer.util.loadNodeFromFile(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/R.mrk.json'
        )
        costNode = myutil.runPreprocess(volumeNode, roiNode)

    @staticmethod
    def loadScene():
        slicer.util.loadScene(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/slicer_gs_clean_update2mrb/2023-10-26-Scene.mrb'
        )
        slicer.util.loadMarkups(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/markups/LCX-D.mrk.json'
        )
        slicer.util.loadMarkups(
            '/Users/flap/Source/artery-stenosis/data/slicer-scene/markups/RCA-D.mrk.json'
        )

    @staticmethod
    def runPreprocess(volumeNode, roiNode, costVolume=None):
        if not costVolume:
            costVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
            costVolume.SetName('BV_COSTVOLUME')
            costVolume.CreateDefaultDisplayNodes()
            costVolume.CreateDefaultStorageNode()
        preprocessLogic = (
            slicer.modules.bvpreprocessvolume.widgetRepresentation().self().logic
        )
        preprocessLogic.process(volumeNode, roiNode, costVolume)
        return costVolume

    @staticmethod
    def test():
        preprocessLogic = (
            slicer.modules.bvpreprocessvolume.widgetRepresentation().self().logic
        )
        stenosisLogic = (
            slicer.modules.bvpreprocessvolume.widgetRepresentation().self().logic
        )

        # process volume
        # Create ROI
        # https://apidocs.slicer.org/main/classvtkMRMLMarkupsROINode.html

        # Process Volume

        # create

    @staticmethod
    def runCreateGuideLine(markersName, costVolumeName='BV_COSTVOLUME'):
        newCurve = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode')
        costVolume = getNode(costVolumeName)
        markers = getNode(markersName)

        stenosisLogic = (
            slicer.modules.bvpreprocessvolume.widgetRepresentation().self().logic
        )
        stenosisLogic.processMarkers(costVolume, markers, newCurve)