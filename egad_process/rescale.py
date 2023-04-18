import numpy as np
import trimesh
import os
from tqdm import tqdm


path = 'egad/eval'
half_extend = 0.02


def main():
    root = os.path.join(os.path.expanduser("~"), path)

    for obj_file in tqdm(os.listdir(root)):
        if obj_file.endswith('.obj'):
            mesh = trimesh.load(os.path.join(root, obj_file))
            mesh.vertices -= mesh.center_mass
            mesh.vertices /= mesh.scale / 2 / np.sqrt(3)
            mesh.vertices *= half_extend
            assert abs(mesh.scale / 2 / np.sqrt(3) - half_extend) < 1e-5
            obj_file = obj_file.replace('.obj', '_rescaled.obj')
            mesh.export(os.path.join(root, obj_file))


if __name__ == '__main__':
    main()
