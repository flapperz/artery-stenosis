from importlib import reload
from queue import PriorityQueue

import numpy as np
import slicer
import slicer.util
import vtk

import StenosisCalLib.MyFunc as MyFunc

reload(MyFunc)


def create_segmentation_from_labelmap(seg_node, volume, volume_node, segname):
    seg_id = seg_node.GetSegmentation().AddEmptySegment(segname)

    labelmap_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    labelmap_node.CreateDefaultDisplayNodes()

    slicer.util.updateVolumeFromArray(labelmap_node, volume)

    direction = np.zeros((3, 3))
    volume_node.GetIJKToRASDirections(direction)

    labelmap_node.SetOrigin(volume_node.GetOrigin())
    labelmap_node.SetSpacing(volume_node.GetSpacing())
    labelmap_node.SetIJKToRASDirections(direction)

    seg_id_vtk = vtk.vtkStringArray()
    seg_id_vtk.InsertNextValue(seg_id)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmap_node, seg_node, seg_id_vtk
    )

    slicer.mrmlScene.RemoveNode(labelmap_node)
    return seg_id


def create_curve(points_kji, curve_node, ijk2ras_mat, spacing=0.4):
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


def vtk4x4matrix_to_numpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m


def get_fiducial_as_kji(fiducial_node, i, ras2ijk_mat):
    ras_homo = np.ones([4, 1])
    fiducial_node.GetNthControlPointPosition(i, ras_homo[:3, 0])
    return tuple(
        (np.round(ras2ijk_mat @ ras_homo).astype(np.uint16).T)[
            (0, 0, 0), (2, 1, 0)
        ].tolist()
    )


def get_fiducial_data(fiducial_node, i, ras2ijk_mat):
    ras_homo = np.ones([4, 1])
    fiducial_node.GetNthControlPointPosition(i, ras_homo[:3, 0])
    posk, posj, posi = (np.round(ras2ijk_mat @ ras_homo).astype(np.uint16).T)[
        (0, 0, 0), (2, 1, 0)
    ].tolist()
    r, a, s = ras_homo[:3, 0]
    lb = fiducial_node.GetNthControlPointLabel(i)
    return (lb, r, a, s, posi, posj, posk)


def ijk2kji(ijk):
    return (ijk[2], ijk[1], ijk[0])


def multi_add_fiducial(fid_node, nodes_ras_Nx3):
    for i in range(len(nodes_ras_Nx3)):
        fid_node.AddFiducialFromArray(nodes_ras_Nx3[i, :])


def MyProcess(inputVolumeNode, arterySeedNode, guidelineNode, segNode):  # calculate backbone 2
    print('MyProcess')
    print(inputVolumeNode.GetName())
    print(arterySeedNode.GetName())
    print(guidelineNode.GetName())
    ijk_spacing = inputVolumeNode.GetSpacing()
    kernel_ras_distance = np.zeros((3, 3, 3), dtype=np.float64)
    for pos_k in range(-1, 2):
        for pos_j in range(-1, 2):
            for pos_i in range(-1, 2):
                kernel_ras_distance[pos_k + 1, pos_j + 1, pos_i + 1] = (
                    (pos_i * ijk_spacing[0]) ** 2
                    + (pos_j * ijk_spacing[1]) ** 2
                    + (pos_k * ijk_spacing[2]) ** 2
                ) ** 0.5
    x = vtk.vtkMatrix4x4()
    inputVolumeNode.GetIJKToRASMatrix(x)
    ijk2ras_mat = vtk4x4matrix_to_numpy(x)

    inputVolumeNode.GetRASToIJKMatrix(x)
    ras2ijk_mat = vtk4x4matrix_to_numpy(x)

    if guidelineNode.GetNumberOfControlPoints() > 0:
        guidelineNode.RemoveAllControlPoints()

    num_fid = arterySeedNode.GetNumberOfControlPoints()
    control_points_label2kji_map = {
        arterySeedNode.GetNthControlPointLabel(i): np.array(
            get_fiducial_as_kji(arterySeedNode, i, ras2ijk_mat)
        )
        for i in range(num_fid)
    }

    first_node_label = arterySeedNode.GetNthControlPointLabel(0)
    last_node_label = arterySeedNode.GetNthControlPointLabel(num_fid - 1)

    # order node
    unselected_pool = set(control_points_label2kji_map.keys())
    control_points_order = [first_node_label]
    unselected_pool.remove(first_node_label)
    while len(unselected_pool):
        min_dist = 999999
        min_label = None
        current_node = control_points_label2kji_map[control_points_order[-1]]
        for label in unselected_pool:
            neighbor_node = control_points_label2kji_map[label]
            diff = current_node - neighbor_node
            dist = np.dot(diff, diff)
            if dist < min_dist:
                min_dist = dist
                min_label = label
        unselected_pool.remove(min_label)
        control_points_order.append(min_label)

        if min_label == last_node_label:
            break

    print(control_points_order)
    control_points_kji = [
        tuple(control_points_label2kji_map[e].tolist()) for e in control_points_order
    ]

    weight_arr = slicer.util.arrayFromVolume(inputVolumeNode).copy()
    COST_OFFSET = -50
    weight_arr[weight_arr > COST_OFFSET] = 2_000_000_000
    weight_arr[weight_arr <= COST_OFFSET] += 4000

    sub_paths = []
    for i in range(len(control_points_kji) - 1):
        # seed = get_fiducial_as_kji(fid_node, i)
        # end_node = get_fiducial_as_kji(fid_node, i+1)
        seed = control_points_kji[i]
        end_node = control_points_kji[i + 1]
        sub_path, _ = MyFunc.dijkstra(seed, end_node, weight_arr, kernel_ras_distance)
        if sub_path is not None:
            sub_paths.append(sub_path)

    # for sub_path in sub_paths:
    #     print(sub_path)

    full_path = []

    for e in sub_paths:
        full_path += e[0:-1]
    full_path.append(sub_paths[-1][-1])

    create_curve(full_path, guidelineNode, ijk2ras_mat)

    print('Finish create guide line')

    # Flood FIll
    _curve_points = guidelineNode.GetCurvePointsWorld()
    curve_points = [
        _curve_points.GetPoint(i) for i in range(_curve_points.GetNumberOfPoints())
    ]
    del _curve_points

    curve_points_kji = (
        np.round(
            ras2ijk_mat
            @ np.concatenate(
                [np.array(curve_points), np.ones((len(curve_points), 1))], axis=1
            ).T
        )
        .astype(np.uint16)
        .T
    )[:, (2, 1, 0)].tolist()
    curve_points_kji = {tuple(e) for e in curve_points_kji}
    print(len(curve_points_kji))

    # ----- Flood Fill -----
    # v1
    kernel_mask = np.zeros((5, 5, 5))
    kernel_mask[1:4, 2, 2] = 1
    kernel_mask[2, 1:4, 2] = 1
    kernel_mask[2, 2, 1:4] = 1
    kernel_mask[2, 2, 2] = 0
    kernel_idxs = np.vstack(np.where(kernel_mask == 1)) - np.array([[2], [2], [2]])

    # dist_grid = np.zeros_like(v_arr, np.float64)
    # dist_grid[:] = 2e9
    # dist_grid[seed] = 0

    visited = set()
    artery = []

    # pred_grid = np.full(v_arr.shape, -1, dtype=np.int32)

    pq = PriorityQueue()
    it = 0

    for seed in curve_points_kji:
        pq.put((0, seed))

    weight_arr = slicer.util.arrayFromVolume(inputVolumeNode).copy()
    MAX_PQ_OFFSET = 5
    # DIFF_MAX = 300
    DIFF_MAX = 150  # upsampling

    while (not pq.empty()) and it < 1e5:
        it += 1

        u_dist, u = pq.get()
        uk, uj, ui = u
        u_pos = np.array(u)
        artery.append((u_dist, u))

        for i in range(kernel_idxs.shape[1]):
            v = tuple((kernel_idxs[:, i] + u_pos).tolist())
            vk, vj, vi = v

            diff = abs(weight_arr[uk, uj, ui] - weight_arr[vk, vj, vi])
            if v not in visited and u_dist < MAX_PQ_OFFSET and weight_arr[vk, vj, vi] <= -100:
                dist = u_dist + 1
                pq.put((dist, v))

            visited.add(v)

            # if v == end_node:
            #     is_reach = True

    print(len(artery))

    weight_arr = weight_arr.astype(bool)
    weight_arr[:] = False
    for u_dist, (k, j, i) in artery:
        # k,j,i = u
        weight_arr[k, j, i] = True

    mask = np.zeros_like(weight_arr, bool)
    mask[weight_arr] = 1
    result = mask.copy()

    kernel_mask = np.zeros((5, 5, 5))
    kernel_mask[1:4, 1:4, 1:4] = 1
    kernel_mask[2, 2, 2] = 0
    kernel_idxs = np.vstack(np.where(kernel_mask == 1)) - np.array([[2], [2], [2]])

    K, J, I = mask.shape

    for i in range(kernel_idxs.shape[1]):
        oi, oj, ok = kernel_idxs[:, i]
        result[1 + ok : K - 1 + ok, 1 + oj : J - 1 + oj, 1 + oi : I - 1 + oi] += mask[
            1 : K - 1, 1 : J - 1, 1 : I - 1
        ]
    seg_id = create_segmentation_from_labelmap(segNode, result.astype(np.int8), inputVolumeNode, 'artery')

    # # dilate

    # mask = np.zeros_like(c_arr, np.bool)
    # mask[c_arr != MYNULL] = 1
    # result = mask.copy()

    # kernel_mask = np.zeros((5, 5, 5))
    # kernel_mask[1:4, 1:4, 1:4] = 1
    # kernel_mask[2, 2, 2] = 0
    # kernel_idxs = np.vstack(np.where(kernel_mask == 1)) - np.array([[2], [2], [2]])

    # K, J, I = mask.shape

    # for i in range(kernel_idxs.shape[1]):
    #     oi, oj, ok = kernel_idxs[:, i]
    #     result[1 + ok : K - 1 + ok, 1 + oj : J - 1 + oj, 1 + oi : I - 1 + oi] += mask[
    #         1 : K - 1, 1 : J - 1, 1 : I - 1
    #     ]
    #     # print(np.sum(crystal_mask[85:186,115:256,240:401]))


    # print(np.sum(result))

    # create_segmentation_from_labelmap(seg_node, result.astype(np.int8), 'dilate')


# updateVolumeFromArray(outc_node, c_arr)
    # MyFunc.dijkstra()
    return segNode, seg_id
