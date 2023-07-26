import numpy as np

class PhongMaterial:
    def __init__(self, ka, kd, ks, n):
        self.ka = ka 
        self.kd = kd
        self.ks = ks
        self.n = n

class PointLight:
    def __init__(self, pos, intensity):
        self.pos = pos
        self.intensity = intensity

def light(point, normal, vcolor, cam_pos, mat, lights, light_amb):
    I = np.zeros((3, 1))

    for light in lights:
        S = light.pos
        I_light = light.intensity
        # create unit vectors L and V
        L = S - point
        L = L / np.linalg.norm(L)
        V = cam_pos - point
        V = V / np.linalg.norm(V)
        # calculate angles a and b
        a = np.dot(np.squeeze(L), np.squeeze(normal))
        cosa = a
        a = np.arccos(a)
        b = np.dot(np.squeeze(normal), np.squeeze(V))
        b = np.arccos(b)
        cosba = np.cos(b - a)

        # get parameters
        kd = mat.kd
        ks = mat.ks
        n = mat.n

        # calculate light for this specific light source
        I_light_source = I_light * (kd * cosa + ks * cosba**n)

        # take color into account
        I_l = np.multiply(vcolor, I_light_source)
        I_l = np.expand_dims(I_l, axis=1)

        # add it up to total light
        I = I + I_l

    # ambient
    ka = mat.ka
    I_amb = light_amb * ka
    # take color into account
    I_l = np.multiply(vcolor, I_amb)
    I_l = np.expand_dims(I_l, axis=1)

    # add it up to total light
    I = I + I_l


    return I