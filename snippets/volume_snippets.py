import numpy as np
import slicer
from slicer import (
    vtkMRMLScalarVolumeNode,
)


def onVolumeRoiLockButtonClicked(self, state: int) -> None:
    """
    :param int state: 0 off / 2 on
    """
    processedVolumeName = "new volume name"
    isChecked = state == 2
    heartRoiNode = self._parameterNode.heartRoi
    # when uncheck on delete ROI
    if not heartRoiNode:
        return

    heartRoiDisplay = heartRoiNode.GetDisplayNode()
    if isChecked:
        # lock
        heartRoiNode.SetLocked(True)
        heartRoiDisplay.SetScaleHandleVisibility(False)
        heartRoiDisplay.SetTranslationHandleVisibility(False)
        self.ui.heartRoiSelector.enabled = False
        self.ui.inputVolumeSelector.enabled = False

        if self._parameterNode and not self._parameterNode._processedVolume:
            processedVolume = self.logic.createEmptyVolume(processedVolumeName)
            self._parameterNode._processedVolume = processedVolume

        self.logic.createProcessedVolume(
            self._parameterNode.inputVolume,
            self._parameterNode.heartRoi,
            self._parameterNode._processedVolume,
        )
    else:
        # unlock
        heartRoiNode.SetLocked(False)
        heartRoiDisplay.SetScaleHandleVisibility(True)
        heartRoiDisplay.SetTranslationHandleVisibility(True)
        self.ui.heartRoiSelector.enabled = True
        self.ui.inputVolumeSelector.enabled = True

    # state = self.ui.volumeRoiLockButton.checkState()

def processVolume(inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode):
    COST_OFFSET = 4000
    AIR_THRESHOLD = -50

    array = slicer.util.arrayFromVolume(inputVolume).copy()
    array[array > AIR_THRESHOLD] = 2_000_000_000
    array[array <= AIR_THRESHOLD] += COST_OFFSET

    slicer.updateVolumeFromArray(outputVolume, array)

def createProcessedVolume(self, inputVolume: vtkMRMLScalarVolumeNode,  heartRoi: vtkMRMLMarkupsROINode, processedVolume: vtkMRMLScalarVolumeNode):
    cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
        'vtkMRMLCropVolumeParametersNode'
    )
    cropVolumeParameters.SetInputVolumeNodeID(inputVolume.GetID())
    cropVolumeParameters.SetROINodeID(heartRoi.GetID())
    cropVolumeParameters.SetOutputVolumeNodeID(processedVolume.GetID())
    cropVolumeParameters.SetInterpolationMode(
        cropVolumeParameters.InterpolationLinear
    )
    cropVolumeParameters.SetVoxelBased(False) # Interpolated Cropping
    cropVolumeParameters.SetSpacingScalingConst(0.4)
    cropVolumeParameters.SetIsotropicResampling(True)
    slicer.modules.cropvolume.logic().Apply(cropVolumeParameters)
    slicer.mrmlScene.RemoveNode(cropVolumeParameters)
    pass

def createCopyVolume(self, inputVolume: vtkMRMLScalarVolumeNode, name: str) -> vtkMRMLScalarVolumeNode:
    imageDimensions = inputVolume.GetImageData().GetDimensions()
    voxelType = vtk.VTK_DOUBLE
    # voxelType = inputVolume.GetImageData().GetScalarType() # dicom is vtk.VTK_INT
    imageOrigin = inputVolume.GetImageData().GetOrigin()
    imageSpacing = inputVolume.GetImageData().GetSpacing()
    fillVoxelValue = 0
    volumeToRAS = inputVolume.GetImageData().GetDirectionMatrix()
    imageDirections = MRMLUtils.vtk3x3matrix2numpy(volumeToRAS)

    imageData = vtk.vtkImageData()
    imageData.SetDimensions(imageDimensions)
    imageData.AllocateScalars(voxelType, 1)
    imageData.GetPointData().GetScalars().Fill(fillVoxelValue)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolume.SetName(name)
    outputVolume.SetOrigin(imageOrigin)
    outputVolume.SetSpacing(imageSpacing)
    outputVolume.SetIJKToRASDirections(imageDirections)
    outputVolume.SetAndObserveImageData(imageData)
    outputVolume.CreateDefaultDisplayNodes()
    outputVolume.CreateDefaultStorageNode()
    return outputVolume

def createEmptyVolume(self, name: str) -> vtkMRMLScalarVolumeNode:

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolume.SetName(name)
    outputVolume.CreateDefaultDisplayNodes()
    outputVolume.CreateDefaultStorageNode()
    return outputVolume

# convol
# create cross
structure = np.zeros((3,3,3))
structure[:,1,1] = 1
structure[1,:,1] = 1
structure[1,1,:] = 1
binsdf[:] = source > 41
binsdf[:] = scipy.ndimage.binary_dilation(binsdf, structure, 4)


# # Copy node
# guideLineShID = shNode.GetItemByDataNode(guideLine)
# endPointsShID = (
#     slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(
#         shNode, guideLineShID
#     )
# )
# endPointsNode = shNode.GetItemDataNode(endPointsShID)
# endPointsNode.SetName('Endpoints_' + markers.GetName())