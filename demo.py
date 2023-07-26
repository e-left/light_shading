import numpy as np
import matplotlib.pyplot as plt

from illumination import PhongMaterial, PointLight
from shading import render_object

# load the data
data = np.load('h3.npy', allow_pickle=True)
data = data.item()

# prepare arrays

# points
verts = data["verts"]
vertex_colors = data["vertex_colors"]
face_indices = data["face_indices"]
# camera
c_eye = data["cam_eye"]
c_up = data["cam_up"]
c_lookat = data["cam_lookat"]
# material
ka = data["ka"]
kd = data["kd"]
ks = data["ks"]
n = data["n"]
# lights
light_positions = data["light_positions"]
light_intensities = data["light_intensities"]
Ia = data["Ia"]
# image
M = data["M"]
N = data["N"]
W = data["W"]
H = data["H"]
bg_color = data["bg_color"]
focal = data["focal"]

# construct material
material_all = PhongMaterial(ka, kd, ks, n)
material_noKa = PhongMaterial(ka, 0, 0, n)
material_noKd = PhongMaterial(0, kd, 0, n)
material_noKs = PhongMaterial(0, 0, ks, n)
materials = [material_all, material_noKa, material_noKd, material_noKs]

# construct lights array
lights = []
num_lights = len(light_positions)
for i in range(num_lights):
    light_t = PointLight(light_positions[i], light_intensities[i])
    lights.append(light_t)

for i in range(len(materials)):
    material = materials[i]
    img = render_object("gouraud", focal, c_eye, c_lookat, c_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, material, lights, Ia)
    img = np.transpose(img, (1, 0, 2))
    img = np.flip(img, axis=0)
    plt.imshow(img)
    plt.savefig(f'gouraud_{i}.png')
    plt.show()

    img = render_object("phong", focal, c_eye, c_lookat, c_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, material, lights, Ia)
    img = np.transpose(img, (1, 0, 2))
    img = np.flip(img, axis=0)
    plt.imshow(img)
    plt.savefig(f'phong_{i}.png')
    plt.show()
