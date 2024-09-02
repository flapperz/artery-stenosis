import time

import slicer
import slicer.util

ladNode = slicer.util.getNode('LAD')
lcxDNode = slicer.util.getNode('LCX-D')
rcaDNode = slicer.util.getNode('RCA-D')
costVolumeNode = slicer.util.getNode('BV_COSTVOLUME')

logic = slicer.modules.bvstenosismeasurement.widgetRepresentation().self().logic

indices = ((0, -1), (0, 15, -1), (0, -1))
indices = ((0, -1), (0, 15, -1), (0, 8, -1))

for name, markersIndices, markupsNode in zip(
    ('LAD', 'LCX', 'RCA'), ((0, -1), (0, 15, -1), (0, -1)), (ladNode, lcxDNode, rcaDNode)
):
    startTime = time.time()

    markupsArray = slicer.util.arrayFromMarkupsControlPoints(markupsNode)

    try:
        markersNode = slicer.util.getNode(f'SEED_SPARSE_{name}')
        markersNode.RemoveAllControlPoints()
    except slicer.util.MRMLNodeNotFoundException:
        markersNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        markersNode.SetName(f'SEED_SPARSE_{name}')
    slicer.util.updateMarkupsControlPointsFromArray(markersNode, markupsArray[markersIndices, :])

    curveName = f'GL_{name}'
    try:
        curveNode = slicer.util.getNode(curveName)
        curveNode.RemoveAllControlPoints()
    except slicer.util.MRMLNodeNotFoundException:
        curveNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode')
        curveNode.SetName(curveName)
    logic.createGuideLine(costVolumeNode, markersNode, curveNode, isSingleton=False)
    slicer.util.delayDisplay(f'Finish Create Guideline: {name} -> {curveName}')

    stopTime = time.time()
    report = '\n'
    report = f'CreateGuideLine: {name}\n'
    report += '-' * len(report) + '\n'
    report += f'time: {stopTime - startTime:.3f}\n'
    report += f'seedIndex: {markersIndices}\n'
    print(report)
