import logging

import numpy as np
import slicer
import slicer.util
import vtk
from LevelSetSegmentation import LevelSetSegmentationLogic, LevelSetSegmentationWidget
from slicer import (
    vtkMRMLCommandLineModuleNode,
)


class CreateGuideLineController:
    def __init__(self):
        self.createGuideLineCliNode = None

    def runCreateGuideLineAsync(self, costVolume, markers, guideLine, isSingleton=True):
        if self.createGuideLineCliNode and isSingleton:
            self.createGuideLineCliNode.Cancel()

        x = vtk.vtkMatrix4x4()
        costVolume.GetRASToIJKMatrix(x)
        ras2ijkMat = slicer.util.arrayFromVTKMatrix(x)

        costVolume.GetIJKToRASMatrix(x)
        ijk2rasMat = slicer.util.arrayFromVTKMatrix(x)

        # transform local position (no parent) to IJK of costVolume
        nMarkers = markers.GetNumberOfControlPoints()
        markersRASHomo = np.ones([4, nMarkers])
        for i in range(nMarkers):
            markers.GetNthControlPointPosition(i, markersRASHomo[:3, i])

        markersIJK = (np.round(ras2ijkMat @ markersRASHomo).astype(np.uint16).T)[
            :, :3
        ].tolist()

        flattenMarkers = [x for ijk in markersIJK for x in ijk]

        logging.debug(f'input flattenMarkers: {flattenMarkers}')
        parameter = {'inputVolume': costVolume, 'inFlattenMarkersIJK': flattenMarkers}
        BVCreateGuideLine = slicer.modules.bvcreateguideline
        if isSingleton:
            cliNode = slicer.cli.run(BVCreateGuideLine, None, parameter)
            cliNode.AddObserver(
                vtkMRMLCommandLineModuleNode.StatusModifiedEvent,
                self._createUpdateCb(ijk2rasMat, guideLine, jumpSliceOnComplete=True),
            )
            self.createGuideLineCliNode = cliNode
        else:
            # for test
            cliNode = slicer.cli.createNode(BVCreateGuideLine, parameter)
            cliNode.AddObserver(
                vtkMRMLCommandLineModuleNode.StatusModifiedEvent,
                self._createUpdateCb(ijk2rasMat, guideLine, jumpSliceOnComplete=False),
            )
            slicer.cli.runSync(BVCreateGuideLine, cliNode)

        return cliNode

    def _createLinearCurve(self, point_kji, curve_node, ijk2ras_mat):
        x = np.array(point_kji, float)
        x = x.T
        x = x[::-1, :]
        x = np.concatenate([x, np.ones((1, x.shape[1]))], axis=0)
        x = ijk2ras_mat @ x
        x = x[:3, :].T

        slicer.util.updateMarkupsControlPointsFromArray(curve_node, x)

    def _createUpdateCb(self, ijk2rasMat, guideLineNode, jumpSliceOnComplete=True):
        def updateCb(cliNode, event):
            # logging.debug("Got a %s from a %s : %s" % (event, cliNode.GetClassName(), cliNode.GetName()))

            status = cliNode.GetStatus()

            if status & cliNode.Completed:
                if status & cliNode.ErrorsMask:
                    # error
                    errorText = cliNode.GetErrorText()
                    logging.debug('CLI execution failed: ' + errorText)
                else:
                    # success
                    outIJK = cliNode.GetParameterAsString('outFlattenMarkersIJK')
                    logging.debug(
                        'CLI execution succeeded. Output model node ID: ' + outIJK
                    )
                    guideLineNode.RemoveAllControlPoints()
                    guideLineNode.SetCurveTypeToLinear()

                    logging.debug(f'CLI output: {outIJK}')
                    outIJK = [int(x) for x in outIJK.split(',')]

                    logging.debug(f'out path length: {len(outIJK)}')
                    pathKJI = [
                        [outIJK[i + 2], outIJK[i + 1], outIJK[i]]
                        for i in range(0, len(outIJK), 3)
                    ]
                    logging.debug(f'formatted: {pathKJI}')
                    self._createLinearCurve(pathKJI, guideLineNode, ijk2rasMat)
                    # MRMLUtils.createCurve(pathKJI, self.guideLineNode, self.ijk2rasMat, 0.5)
                    slicer.modules.markups.logic().SetAllControlPointsVisibility(guideLineNode, False)

                    if jumpSliceOnComplete:
                        guideLineSize = guideLineNode.GetNumberOfControlPoints()
                        if guideLineSize:
                            markupsLogic = slicer.modules.markups.logic()
                            markupsLogic.FocusCamerasOnNthPointInMarkup(
                                guideLineNode.GetID(), guideLineSize // 2
                            )

                slicer.mrmlScene.RemoveNode(cliNode)
                return

            if status & cliNode.Cancelled:
                slicer.mrmlScene.RemoveNode(cliNode)

        return updateCb


class VesselnessFilteringController:
    @staticmethod
    def createVesselnessVolume(
        inputVolumeNode,
        seedNode,
        minDiameterMM,
        maxDiameterMM,
        contrast,
        suppressPlate=10,
        suppressBlob=10,
        lowerThreshold=0.1,
        isCalculateParameter=False,
    ):
        from VesselnessFiltering import VesselnessFilteringLogic

        logic = VesselnessFilteringLogic()

        outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        outputVolumeNode.CreateDefaultDisplayNodes()
        outputDisplayNode = outputVolumeNode.GetDisplayNode()

        # Set threshold
        outputDisplayNode.AutoThresholdOff()
        outputDisplayNode.SetLowerThreshold(lowerThreshold)
        outputDisplayNode.SetUpperThreshold(1.0)
        outputDisplayNode.ApplyThresholdOn()

        # Set red colormap
        outputDisplayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeRed')
        outputDisplayNode.AutoWindowLevelOff()
        outputDisplayNode.SetWindowLevelMinMax(-0.6, 2.0)

        if isCalculateParameter:
            vesselPositionIJK = logic.getIJKFromRAS(
                inputVolumeNode, logic.getSeedPositionRAS(seedNode)
            )
            detectedDiameterVoxel = logic.getDiameter(
                inputVolumeNode.GetImageData(), vesselPositionIJK
            )
            contrast = logic.calculateContrastMeasure(
                inputVolumeNode.GetImageData(), vesselPositionIJK, detectedDiameterVoxel
            )
            minDiameterMM = min(inputVolumeNode.GetSpacing())
            maxDiameterMM = detectedDiameterVoxel * min(inputVolumeNode.GetSpacing())

        alpha = logic.alphaFromSuppressPlatesPercentage(suppressPlate)
        beta = logic.betaFromSuppressBlobsPercentage(suppressBlob)

        logging.info(
            f'Vesselness parameter: {minDiameterMM=} {maxDiameterMM=} {alpha=} {beta=} {contrast=}'
        )

        previewRegionSizeVoxel = -1
        previewRegionCenterRAS = None
        logic.computeVesselnessVolume(
            inputVolumeNode,
            outputVolumeNode,
            previewRegionCenterRAS,
            previewRegionSizeVoxel,
            minDiameterMM,
            maxDiameterMM,
            alpha,
            beta,
            contrast,
        )

        return outputVolumeNode

class LevelSetSegmentationController:
    methodCollidingFronts = 'collidingfronts'
    methodFastMarching = 'fastmarching'
    levelSetsTypeGeodesic = 'geodesic'
    levelSetsTypeCurves = 'curves'

    def __init__(self, volumeNode, vesselnessNode):

        self.logic = LevelSetSegmentationLogic()

        self.volumeNode = volumeNode
        self.inputImage = vtk.vtkImageData()
        self.inputImage.DeepCopy(vesselnessNode.GetImageData())
        self.labelMapData = None

    def performEvolution(
        self,
        seedsNode,
        stoppersNode=None,
        minVesselnessThreshold=0.1,
        iteration=10,
        inflation=0,
        curvature=70,
        attraction=50,
        method='fastmarching',
        levelSetsType='geodesic'
    ):
        seeds = LevelSetSegmentationWidget.convertFiducialHierarchyToVtkIdList(
            seedsNode, self.volumeNode
        )

        if stoppersNode:
            stoppers = LevelSetSegmentationWidget.convertFiducialHierarchyToVtkIdList(
                stoppersNode, self.volumeNode
            )
        else:
            stoppers = vtk.vtkIdList()

        # initialization
        initImageData = vtk.vtkImageData()
        initImageData.DeepCopy(
            self.logic.performInitialization(
                self.inputImage,
                minVesselnessThreshold,
                1.00,
                seeds,
                stoppers,
                method,
            )
        )

        # evolution
        evolImageData = vtk.vtkImageData()
        evolImageData.DeepCopy(
            self.logic.performEvolution(
                self.volumeNode.GetImageData(),
                initImageData,
                iteration,
                inflation,
                curvature,
                attraction,
                levelSetsType,
            )
        )

        if not self.labelMapData:
            self.labelMapData = vtk.vtkImageData()
            self.labelMapData.DeepCopy(self.logic.buildSimpleLabelMap(evolImageData, 5, 0))
        else:
            newLabelMapData = vtk.vtkImageData()
            newLabelMapData.DeepCopy(self.logic.buildSimpleLabelMap(evolImageData, 5, 0))
            self.updateCurrentLabelMapData(newLabelMapData)

    def updateCurrentLabelMapData(self, newLabelMapData):
        imageMath = vtk.vtkImageMathematics()
        imageMath.SetOperationToMax()  # Max operation will work like logical OR for binary images
        imageMath.SetInput1Data(self.labelMapData)
        imageMath.SetInput2Data(newLabelMapData)
        imageMath.Update()

        self.labelMapData.DeepCopy(imageMath.GetOutput())

    def createResultLabelMapNode(self):
        labelMapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        labelMapNode.CopyOrientation(self.volumeNode)
        labelMapNode.SetAndObserveImageData(self.labelMapData)
        labelMapNode.CreateDefaultDisplayNodes()
        return labelMapNode
