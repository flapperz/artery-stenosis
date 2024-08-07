import numpy as np
import vtk


def createCurve(points_kji, curve_node, ijk2ras_mat, spacing=0.4):
    x = np.array(points_kji, float)
    x = x.T
    x = x[::-1, :]
    x = np.concatenate([x, np.ones((1, x.shape[1]))], axis=0)
    x = ijk2ras_mat @ x
    x = x[:3, :]

    # TODO optimize to vectorize accumulation
    CURVE_CTRL_SPACING = spacing  # may be mm
    count = 0
    last_add_dist = 0
    curr_dist = 0
    curve_node.AddControlPoint(vtk.vtkVector3d(*x[:, 0]))

    for pos_j in range(1, x.shape[1]):
        prev_ras = x[:, pos_j - 1]
        curr_ras = x[:, pos_j]
        curr_dist += np.linalg.norm(curr_ras - prev_ras)
        if curr_dist - last_add_dist >= CURVE_CTRL_SPACING:
            count += 1
            # print(curr_dist - last_add_dist)
            curve_node.AddControlPoint(vtk.vtkVector3d(*x[:, pos_j]))
            last_add_dist = curr_dist

def getFiducialAsKJI(fiducial_node, i, ras2ijk_mat):
    ras_homo = np.ones([4, 1])
    fiducial_node.GetNthControlPointPosition(i, ras_homo[:3, 0])
    return tuple(
        (np.round(ras2ijk_mat @ ras_homo).astype(np.uint16).T)[
            (0, 0, 0), (2, 1, 0)
        ].tolist()
    )

# def flatten_fiducial_ijk(fiducialsIJK):
#     out = ''
#     for (i,j,k) in fiducialsIJK:
#         out += f'{i},{j},{k},'
#     if out:
#         out = out[:-1]
#     return out

def getFiducialAsIJK(fiducial_node, i, ras2ijk_mat):
    ras_homo = np.ones([4, 1])
    fiducial_node.GetNthControlPointPosition(i, ras_homo[:3, 0])
    return tuple(
        (np.round(ras2ijk_mat @ ras_homo).astype(np.uint16).T)[
            (0, 0, 0), (0, 1, 2)
        ].tolist()
    )


def vtk4x4matrix2numpy(matrix, outDtype='d'):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    m = np.ones((4, 4), outDtype)
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m

def vtk3x3matrix2numpy(matrix, outDtype='d'):
    m = np.ones((3, 3), outDtype)
    for i in range(3):
        for j in range(3):
            m[i, j] = matrix.GetElement(i, j)
    return m
