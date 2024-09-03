import slicer
import slicer.cli
import slicer.util
import vtk
from slicer import vtkMRMLTransformNode

roi = slicer.util.getNode('R')
inputVolume = slicer.util.getNode('14: Body Soft Tissue')

resampleLogic = slicer.modules.resamplescalarvectordwivolume.logic()
cropLogic = slicer.modules.cropvolume

outputExtent = [0, -1, 0, -1, 0, -1]
outputSpacing = [0, 0, 0]
isotropicResampling = True
cropLogic.GetInterpolatedCropOutputGeometry(
    roi, inputVolume, isotropicResampling, outputSpacing, outputExtent
)
roiXYZ = [0, 0, 0]
roiRadius = [0, 0, 0]
roi.GetRadiusXYZ(roiRadius)
outputIJKToRas = vtk.vtkMatrix4x4()

outputIJKToRas.SetElemement(0, 0, outputSpacing[0])
outputIJKToRas.SetElemement(1, 1, outputSpacing[1])
outputIJKToRas.SetElemement(2, 2, outputSpacing[2])
outputIJKToRas.SetElemement(0, 3, roiXYZ[0] - roiRadius[0])
outputIJKToRas.SetElemement(1, 3, roiXYZ[1] - roiRadius[1])
outputIJKToRas.SetElemement(2, 3, roiXYZ[2] - roiRadius[2])

roiTransform = roi.GetParentTransformNode()
# outputTransform = outputVolume.GetParentTransformNode
outputTransform = None

roiMatrix = vtk.vtkMatrix4x4()

# GetMatrixTransformFromObjectToNode
vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(roi.GetParentTransformNode(), toNode.GetParentTransformNode(), objectToNode)
vtk.vtkMatrix4x4.Multiply4x4(objectToNode, roi.GetObjectToNodeMatrix(), objectToNode)

# give up if it's too long do this another time


