def interpolate_vectors(p1, p2, V1, V2, xy, dim):
    return 1.0 * ((p2[dim - 1] - xy) * V1 + (xy - p1[dim - 1]) * V2) / (p2[dim - 1] - p1[dim - 1])

def normalize_color(color):
    for i in range(len(color)):
        color[i] = max(0, min(color[i], 1))
    
    return color
