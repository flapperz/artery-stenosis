import logging

import numpy as np
import slicer.util
import vtk
from slicer import (
    vtkMRMLCommandLineModuleNode,
)


class CreateGuideLineController():
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
                    pathKJI = [[outIJK[i + 2], outIJK[i + 1], outIJK[i]] for i in range(0, len(outIJK), 3)]
                    logging.debug(f'formatted: {pathKJI}')
                    self._createLinearCurve(pathKJI, guideLineNode, ijk2rasMat)
                    # MRMLUtils.createCurve(pathKJI, self.guideLineNode, self.ijk2rasMat, 0.5)

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

