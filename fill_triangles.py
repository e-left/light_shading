from helpers import interpolate_vectors, normalize_color

def gourauds(canvas, vertices, vcolors):
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
        xkmin.append(min(vertices[i][0], vertices[(i + 1) % k][0]))
        ykmin.append(min(vertices[i][1], vertices[(i + 1) % k][1]))
        xkmax.append(max(vertices[i][0], vertices[(i + 1) % k][0]))
        ykmax.append(max(vertices[i][1], vertices[(i + 1) % k][1]))

    # compute ymin, ymax
    ymin = min(ykmin)
    ymax = max(ykmax)

    # keep track of horizontal edges
    horizontal_edges = []

    # calculate slopes
    slopes = []
    for edge in range(k):
        point_a = vertices[edge]
        point_b = vertices[(edge + 1) % k]
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
        point_a = vertices[edge]
        point_b = vertices[(edge + 1) % k]

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
        for point in active_boundary_points:
            # for each point find edge it belongs to
            # slope (need it for checking)
            # and edge points
            edge = point[2]
            slope = slopes[edge]
            point_a = vertices[edge]
            point_b = vertices[(edge + 1) % k]
            color_a = vcolors[edge]
            color_b = vcolors[(edge + 1) % k]

            color = None

            # check if the two points are the same
            if point_a[0] == point_b[0] and point_a[1] == point_b[1]:
                # set color to their color (same edge -> same color)
                color = color_a
            else:
                color = interpolate_vectors(point_a, point_b, color_a, color_b, point[1], 2)
            # save colors in the same order
            boundary_points_colors.append(color)

        # use fast implementation
        xlist = list(map(lambda x: round(x[0]), active_boundary_points))
        for x in range(xlist[0], xlist[-1] + 1):
            # going over a vertex
            if xlist[0] == xlist[-1]:
                x = xlist[0]
                for i in range(k):
                    if (x == vertices[i][0]) and (y == vertices[i][1]):
                        canvas[x][y] = vcolors[i]
                continue
            color = interpolate_vectors(active_boundary_points[0], active_boundary_points[-1], \
                boundary_points_colors[0], boundary_points_colors[-1], \
                    x, 1)
            color = normalize_color(color)
            canvas[x][y] = color


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
                point_a = vertices[edge]
                point_b = vertices[(edge + 1) % k]
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
        point_a = vertices[edge]
        point_b = vertices[(edge + 1) % k]
        color_a = vcolors[edge]
        color_b = vcolors[(edge + 1) % k]
        # rearrange according to point_a being the first one regarding x
        if point_a[0] > point_b[0]:
            temp = point_b
            point_b = point_a
            point_a = temp

            temp = color_b
            color_b = color_a
            color_a = temp
        y = point_a[1] # = point_b[1]
        if y == ymax:
            for x in range(xkmin[edge], xkmax[edge]):
                color = interpolate_vectors(point_a, point_b, \
                    color_a, color_b, \
                        x, 1)
                canvas[x][y] = color

    return canvas