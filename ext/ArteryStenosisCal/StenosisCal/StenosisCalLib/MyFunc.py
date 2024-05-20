from queue import PriorityQueue

import numpy as np
import vtk


def dijkstra(seed, end_node, weight_grid, kernel_ras_distance):
    kernel_mask = np.zeros((5, 5, 5))
    kernel_mask[1:4, 1:4, 1:4] = 1
    kernel_mask[2, 2, 2] = 0
    kernel_idxs = np.vstack(np.where(kernel_mask == 1)) - np.array([[2], [2], [2]])

    # dist_grid = np.zeros_like(v_arr, np.float64)
    # dist_grid[:] = 2e9
    # dist_grid[seed] = 0

    dist_map = dict()
    dist_map[seed] = 0

    pred_map = dict()

    # pred_grid = np.full(v_arr.shape, -1, dtype=np.int32)

    pq = PriorityQueue()
    pq.put((0, seed))
    it = 0

    is_reach = False
    debug_dict = dict()

    while (not pq.empty()) and it < 5e6 and (not is_reach):
        it += 1

        u_dist, u = pq.get()
        ui, uj, uk = u
        u_pos = np.array(u)

        if u == end_node:
            is_reach = True
            break

        if u in dist_map and u_dist > dist_map[u]:
            continue

        for i in range(kernel_idxs.shape[1]):
            v = tuple((kernel_idxs[:, i] + u_pos).tolist())
            vi, vj, vk = v

            # ras_vecx2 = kji_list2ras([u,v])
            # dist_uv = np.linalg.norm(ras_vecx2[:,0] - ras_vecx2[:,1])

            _uvk, _uvj, _uvi = kernel_idxs[:, i] + 1
            dist_uv = kernel_ras_distance[_uvk, _uvj, _uvi]
            # dist_uv = 1
            curr_cost = dist_uv * weight_grid[vi, vj, vk]

            dist = u_dist + curr_cost

            if v not in dist_map or dist < dist_map[v]:
                dist_map[v] = dist
                pred_map[v] = u
                pq.put((dist, v))

            # if v == end_node:
            #     is_reach = True

    print(it)

    # debug_dict['distgrid'] = dist_grid

    if not is_reach:
        print('not reach')
        return [], debug_dict

    # get path inclusive [start --> end]
    crawl = end_node
    i, j, k = crawl
    path = [crawl]
    it = 0
    # print(pred_map)
    # while  pred_grid[i,j,k] != -1 and it < 512+275+512:
    while crawl in pred_map and it < 512 + 275 + 512:
        it += 1
        # parent = z2coord(pred_grid[i,j,k])
        parent = pred_map[crawl]
        path.append(parent)
        crawl = parent
        # i,j,k = crawl
    # path.append(crawl)
    # print(path)
    path.reverse()

    # return
    return path, debug_dict


def create_curve(points_kji, curve_node, ijk2ras_mat, spacing=0.4):
    x = np.array(points_kji, np.float)
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

    return
