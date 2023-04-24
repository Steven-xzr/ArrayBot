import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os
from tqdm import tqdm


path = 'egad/train'
# path = 'egad/eval'
half_extend = 0.02


def main():
    root = os.path.join(os.path.expanduser("~"), path)

    # ratio_dict = {'A': {'00': []}}

    ratio_dict = {}
    nine_count = 0
    eight_count = 0
    all = 0

    # for obj_file in tqdm(os.listdir(root)):
    for obj_file in tqdm(sorted(os.listdir(root))):
        if obj_file.endswith('.obj') and not obj_file.endswith('_rescaled.obj') and not obj_file.endswith('_rescaled_4cm.obj'):
        # if obj_file.endswith('_rescaled.obj'):
            mesh = trimesh.load(os.path.join(root, obj_file))
            bounds = mesh.bounds
            ratio = (bounds[1] - bounds[0]).min() / (bounds[1] - bounds[0]).max()

            letter = obj_file[0]
            number = obj_file[1:3]
            # number = obj_file[1]

            ratio_dict.setdefault(letter, {})
            ratio_dict[letter].setdefault(number, [])
            ratio_dict[letter][number].append(ratio)


            # print(obj_file, ratio)
            #
            # if ratio > 0.9:
            #     nine_count += 1
            # elif ratio > 0.8:
            #     eight_count += 1
            # all += 1

            # mesh.vertices -= mesh.center_mass
            # mesh.vertices /= mesh.scale / 2 / np.sqrt(3)
            # mesh.vertices *= half_extend
            # assert abs(mesh.scale / 2 / np.sqrt(3) - half_extend) < 1e-5


            # obj_file = obj_file.replace('.obj', '_rescaled_4cm.obj')
            # mesh.export(os.path.join(root, obj_file))

    z = np.zeros((24, 26))
    # z = np.zeros((7, 7))
    for l_idx, letter in enumerate(ratio_dict):
        for n_idx, number in enumerate(ratio_dict[letter]):
            if len(ratio_dict[letter][number]) == 1:
                z[l_idx, n_idx] = ratio_dict[letter][number][0]
            else:
                z[l_idx, n_idx] = np.mean(ratio_dict[letter][number])

    plt.imshow(z, cmap='gray', origin='lower')
    plt.ylabel('grasp difficulty')
    plt.xlabel('shape complexity')
    plt.title('train set length-width-height ratio')
    plt.show()


    print(nine_count)
    print(eight_count)
    print(all)

if __name__ == '__main__':
    main()
