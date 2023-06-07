import numpy as np
import torch
import torch_dct
import matplotlib.pyplot as plt

gap = 1
pixel = 5

zigzag = np.array([[0, 0],
                       [1, 0], [0, 1],
                       [0, 2], [1, 1], [2, 0],
                       [3, 0], [2, 1], [1, 2], [0, 3],
                       [0, 4], [1, 3], [2, 2], [3, 1], [4, 0],
                       [4, 1], [3, 2], [2, 3], [1, 4],
                       [2, 4], [3, 3], [4, 2],
                       [4, 3], [3, 4],
                       [4, 4]])

RGB = np.zeros((25 + 4 * gap, 25 + 4 * gap, 3))
# R = np.zeros((25, 25))
# G = np.zeros((25, 25))
# B = np.zeros((25, 25))

zero_freq = torch.zeros(5, 5)

for ij in zigzag:
    i = ij[0]
    j = ij[1]
    freq = zero_freq
    freq[i, j] = 1
    spatial = torch_dct.idct_2d(freq).numpy()
    if i == 0 and j == 0:
        spatial = np.ones((5, 5))
    else:
        spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min())
    if i + j <= 2:
        RGB[i * (5 + gap):i * (5 + gap) + 5, j * (5 + gap):j * (5 + gap) + 5, 0] = spatial * 0 / 255
        RGB[i * (5 + gap):i * (5 + gap) + 5, j * (5 + gap):j * (5 + gap) + 5, 1] = spatial * 255 / 255
        RGB[i * (5 + gap):i * (5 + gap) + 5, j * (5 + gap):j * (5 + gap) + 5, 2] = spatial * 100 / 255
    else:
        RGB[i * (5 + gap):i * (5 + gap) + 5, j * (5 + gap):j * (5 + gap) + 5, 0] = spatial
        RGB[i * (5 + gap):i * (5 + gap) + 5, j * (5 + gap):j * (5 + gap) + 5, 1] = spatial
        RGB[i * (5 + gap):i * (5 + gap) + 5, j * (5 + gap):j * (5 + gap) + 5, 2] = spatial

plt.imshow(RGB)
plt.axis('off')
plt.savefig('DCT_map.png', dpi=400)
plt.show()

