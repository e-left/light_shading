import numpy as np

import illumination
import camera_functions
import fill_triangles

from helpers import interpolate_vectors, normalize_color

def calculate_normals(verts, faces):
    verts = verts.T
    Nv = verts.shape[0]

    faces = faces.T

    normals = np.zeros((Nv, 3))

    # calculate additive normal for each point
    # ie add normals from all faces it belongs
    for face in faces:
        # point indices
        iA = face[0]
        iB = face[1]
        iC = face[2]
        # points
        A = verts[iA]
        B = verts[iB]
        C = verts[iC]

        AB = B - A
        AC = C - A

        # calculate and normalize
        N = np.cross(np.squeeze(AB), np.squeeze(AC))
        N = N / np.linalg.norm(N)

        # add to all points
        normals[iA] = normals[iA] + N
        normals[iB] = normals[iB] + N
        normals[iC] = normals[iC] + N


    # in the end, normalize the normals
    norms = np.linalg.norm(normals, axis=1)
    norms = np.expand_dims(norms, axis=1)
    normals = np.divide(normals, norms)

    # transpose to comply with shape requirements
    normals = normals.T
    return normals

def render_object(shader, focal, eye, lookat, 
                  up, bg_color, M, N, H, W, verts, 
                  vert_colors, faces, mat, 
                  lights, light_amb):
    # determine plotting function
    plotting_function = None
    if shader == "gouraud":
        plotting_function = shade_gouraud
    elif shader == "phong":
        plotting_function = shade_phong
    
    # calculate normals
    normals = calculate_normals(verts, faces)

    # project points
    p2d, depth = camera_functions.camera_looking_at(focal, eye, lookat, up, verts)
    verts2d = camera_functions.rasterize(p2d, M, N, H, W)

    # construct initial image
    img = np.zeros((M, N, 3))
    # color background
    img[:, :] = np.squeeze(bg_color)

    # transform
    faces = faces.T
    vert_colors = vert_colors.T
    normals = normals.T
    verts = verts.T

    # grab useful quantities
    k = len(faces) 

    # find all triangles, with the proper order according to depth
    # constuct cog depths hashmap
    cog_depths = np.zeros((k))
    cog_depths = {}
    for i in range(k):
        # first coordinate
        idx_a = faces[i][0]
        idx_b = faces[i][1]
        idx_c = faces[i][2]

        # find points
        point_a_depth = depth[idx_a]
        point_b_depth = depth[idx_b]
        point_c_depth = depth[idx_c]

        # calculate cog coordinate in z axis (depth)
        cog_depth = (point_a_depth + point_b_depth + point_c_depth) / 3.0

        # store depth (z coordinate) in hashmap
        cog_depths[str(cog_depth)] = i

    # color all triangles
    # obtain all depths (keys of dictionary, strings)
    depths = list(cog_depths.keys())
    # sort in reverse, since we want further triangles
    # (higher depth) to be colored firsts
    # sort by numerical value
    depths.sort(key=lambda x: float(x), reverse=True)
    # iterate and draw triangle
    for depth in depths:
        # get index
        faces_idx = cog_depths[depth]
        # get point idices
        point_idx = faces[faces_idx]

        # get vertices
        triangle = np.array(verts2d[point_idx])
        triangle = triangle.astype(np.int64)

        # return img
        # paint triangle over canvas
        # only if valid coordinates
        plot_triangle = True
        for point in triangle:
            if (point[0] < 0 or point[0] > M - 1) or \
                (point[1] < 0 or point[1] > N - 1):
                plot_triangle = False

        if plot_triangle:
            colors = np.array(vert_colors[point_idx])
            norms = normals[point_idx]
            cog = np.mean(verts[point_idx], axis=1)
            img = plotting_function(triangle, norms, colors, cog, eye, mat, lights, light_amb, img)

    return img


def shade_gouraud(vertsp, vertsn, vertsc, 
                  bcoords, cam_pos, mat, lights, light_amb, X):
    
    # calculate color based on light model
    vert_colors = np.zeros((3, 3))
    for i in range(3):
        vert_colors[i] = np.squeeze(illumination.light(bcoords, vertsn[i], vertsc[i], cam_pos, mat, lights, light_amb))

    # color triangle
    img = fill_triangles.gourauds(X, vertsp, vert_colors)

    return img

def shade_phong(vertsp, vertsn, vertsc, 
                  bcoords, cam_pos, mat, lights, light_amb, X):
    # find points to be colored
    # and color the points

    # triangle
    k = 3

    # find xkmin, ykmin, xmkax, ykmax
    xkmin = []
    ykmin = []
    xkmax = []
    ykmax = []

    for i in range(k):
        xkmin.append(min(vertsp[i][0], vertsp[(i + 1) % k][0]))
        ykmin.append(min(vertsp[i][1], vertsp[(i + 1) % k][1]))
        xkmax.append(max(vertsp[i][0], vertsp[(i + 1) % k][0]))
        ykmax.append(max(vertsp[i][1], vertsp[(i + 1) % k][1]))

    # compute ymin, ymax
    ymin = min(ykmin)
    ymax = max(ykmax)

    # keep track of horizontal edges
    horizontal_edges = []

    # calculate slopes
    slopes = []
    for edge in range(k):
        point_a = vertsp[edge]
        point_b = vertsp[(edge + 1) % k]
        if point_b[0] == point_a[0]:
            slopes.append("inf")
            continue
        slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
        slopes.append(slope)
    
    # find horizontal lines
    for i in range(k):
        if slopes[i] == 0:
            horizontal_edges.append(i)

    # find initial active edges
    active_edges = []
    for i in range(k):
        if ykmin[i] == ymin and i not in horizontal_edges:
            active_edges.append(i)

    # find initial boundary points
    active_boundary_points = []    
    for edge in active_edges:
        # find two edge points
        point_a = vertsp[edge]
        point_b = vertsp[(edge + 1) % k]

        # determine x for ymin initially from line equation
        x = 0
        # vertical line
        if point_a[0] == point_b[0]:
            x = point_a[0]
        else:
            # y = slope * x + bias
            # slope
            slope = slopes[edge]
            bias = point_b[1] - slope * point_b[0]
            x = (ymin - bias) / slope
        active_boundary_points.append([x, ymin, edge])

    # main computation loop
    for y in range(ymin, ymax + 1):
        if len(active_boundary_points) < 1:
            continue
        # sort active boundary points by x
        active_boundary_points.sort(key = lambda x: x[0])
        # create edge colors by interpolating on y between lowest and highest points
        boundary_points_colors = []
        boundary_points_normals = []
        for point in active_boundary_points:
            # for each point find edge it belongs to
            # slope (need it for checking)
            # and edge points
            edge = point[2]
            slope = slopes[edge]
            point_a = vertsp[edge]
            point_b = vertsp[(edge + 1) % k]
            color_a = vertsc[edge]
            color_b = vertsc[(edge + 1) % k]
            normal_a = vertsn[edge]
            normal_b = vertsn[(edge + 1) % k]

            color = None
            norm = None

            # check if the two points are the same
            if point_a[0] == point_b[0] and point_a[1] == point_b[1]:
                # set color to their color (same edge -> same color)
                color = color_a
                norm = normal_a
            else:
                color = interpolate_vectors(point_a, point_b, color_a, color_b, point[1], 2)
                norm = interpolate_vectors(point_a, point_b, normal_a, normal_b, point[1], 2)
            # save colors in the same order
            boundary_points_colors.append(color)
            boundary_points_normals.append(norm)

        # use fast implementation
        xlist = list(map(lambda x: round(x[0]), active_boundary_points))
        for x in range(xlist[0], xlist[-1] + 1):
            # going over a vertex
            if xlist[0] == xlist[-1]:
                x = xlist[0]
                for i in range(k):
                    if (x == vertsp[i][0]) and (y == vertsp[i][1]):
                        X[x][y] = vertsc[i]
                continue
            color = interpolate_vectors(active_boundary_points[0], active_boundary_points[-1], \
                boundary_points_colors[0], boundary_points_colors[-1], \
                    x, 1)
            normal = interpolate_vectors(active_boundary_points[0], active_boundary_points[-1], \
                boundary_points_normals[0], boundary_points_normals[-1], \
                    x, 1)
            color = normalize_color(color)
            normal = normal / np.linalg.norm(normal)
            c = illumination.light(bcoords, normal, color, cam_pos, mat, lights, light_amb)
            c = np.squeeze(c)
            X[x][y] = c


        # recursively update active edges
        for temp_edge in range(3):
            if ykmin[temp_edge] == y + 1 and temp_edge not in horizontal_edges:
                active_edges.append(temp_edge)
        for edge in active_edges:
            if ykmax[edge] == y:
                active_edges.remove(edge)

        # recursively update active boundary points
        # add a 1 to y of the existing points
        # add 1/slope to x of the existing points
        for i in range(len(active_boundary_points)):
            slope = slopes[active_boundary_points[i][2]]
            x_incr = 0
            # we cover horizontal lines so no need to check 
            # if its vertical don't increment x
            if slope != "inf":
                x_incr = 1 / slope
            active_boundary_points[i][0] += x_incr
            active_boundary_points[i][1] += 1

        # add points from new edges
        for edge in active_edges:
            if ykmin[edge] == y + 1:
                # find previous x
                # take x of point with lowest y from two points that comprise edge
                point_a = vertsp[edge]
                point_b = vertsp[(edge + 1) % k]
                x = 0
                if point_a[1] < point_b[1]:
                    x = point_a[0]
                else:
                    x = point_b[0]
                active_boundary_points.append([x, y + 1, edge])
        
        # remove any point that is on a line not currently active
        for point in active_boundary_points:
            if point[2] not in active_edges:
                active_boundary_points.remove(point)
        
    # color horizontal lines according to convention
    for edge in horizontal_edges:
        # get points
        point_a = vertsp[edge]
        point_b = vertsp[(edge + 1) % k]
        color_a = vertsc[edge]
        color_b = vertsc[(edge + 1) % k]
        normal_a = vertsn[edge]
        normal_b = vertsn[(edge + 1) % k]
        # rearrange according to point_a being the first one regarding x
        if point_a[0] > point_b[0]:
            temp = point_b
            point_b = point_a
            point_a = temp

            temp = color_b
            color_b = color_a
            color_a = temp

            temp = normal_b
            normal_b = normal_a
            normal_a = temp
        y = point_a[1] # = point_b[1]
        if y == ymax:
            for x in range(xkmin[edge], xkmax[edge]):
                color = interpolate_vectors(point_a, point_b, \
                    color_a, color_b, \
                        x, 1)
                normal = interpolate_vectors(point_a, point_b, \
                    normal_a, normal_b, \
                        x, 1)
                normal = normal / np.linalg.norm(normal)
                c = illumination.light(bcoords, normal, color, cam_pos, mat, lights, light_amb)
                c = np.squeeze(c)
                X[x][y] = c

    return X
