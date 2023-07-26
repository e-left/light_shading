import numpy as np
import transformation_functions

def pin_hole(f, cv, cx, cy, cz, p3d):
    # need to go from wcs to camera 
    # apply change_coordinate_system
    t = cv
    R = np.hstack((cx, cy, cz)).T
    p3d_camera = transformation_functions.change_coordinate_system(p3d, R, t)

    # need to go from camera to image
    # check if multiple points or single point
    if len(p3d_camera.shape) > 1:
        # extract variables
        xs = p3d_camera[0, :]
        ys = p3d_camera[1, :]
        zs = p3d_camera[2, :]
        # perspective transformation
        xs = f * xs / zs
        ys = f * ys / zs

        res = np.vstack((xs, ys))
        return (res, zs)
    else:
        # extract variables
        x = p3d_camera[0]
        y = p3d_camera[1]
        z = p3d_camera[2]

        # perpsective transformation
        x = f * x / z
        y = f * y / z

        point = np.array([x, y])
        return (point, z)

def camera_looking_at(f, cv, cK, cup, p3d):
    # will need to determine cx, cy, cz from cK and cup

    VK = cK - cv
    cz = VK / np.linalg.norm(VK)

    cup_cz_dot = np.dot(np.squeeze(cz), np.squeeze(cup))

    t = cup - cup * cup_cz_dot
    cy = t / np.linalg.norm(t)

    cx = np.cross(np.squeeze(cy), np.squeeze(cz))
    cx = np.expand_dims(cx, axis=1)
    cy = np.expand_dims(cy, axis=1)
    cz = np.expand_dims(cz, axis=1)
    
    # obtain and return results
    res, depths = pin_hole(f, cv, cx, cy, cz, p3d)
    return (res, depths)

def rasterize(p2d, rows, columns, H, W):
    # assume that p2d is 2 by N when multiple points
    if len(p2d.shape) > 1:
        p2d = p2d.T
        mult_part = np.array((columns / W, rows / H))
        add_part = np.array((W / 2, H / 2))
        # add one because it starts from zero
        p2d = 1 + p2d + add_part
        p2d = p2d * mult_part

        p2d = np.round(p2d)

        return p2d
    else:
        x = p2d[0]
        y = p2d[1]

        # add 1 because we start from zero
        x = columns * (1 + x + W / 2) / W
        y = rows * (1 + y + H / 2) / H

        res = np.array((x, y))
        res = np.round(res)
        return res