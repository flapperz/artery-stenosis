import numpy as np
import slicer
from slicer import (
    vtkMRMLScalarVolumeNode,
)


def processVolume(inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode):
    COST_OFFSET = 4000
    AIR_THRESHOLD = -50

    array = slicer.util.arrayFromVolume(inputVolume).copy()
    array[array > AIR_THRESHOLD] = 2_000_000_000
    array[array <= AIR_THRESHOLD] += COST_OFFSET

    slicer.updateVolumeFromArray(outputVolume, array)
