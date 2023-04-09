# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
import torch
import torch_dct

zigzag = torch.tensor([[0, 0],
                       [1, 0], [0, 1],
                       [0, 2], [1, 1], [2, 0],
                       [3, 0], [2, 1], [1, 2], [0, 3],
                       [0, 4], [1, 3], [2, 2], [3, 1], [4, 0],
                       [5, 0], [4, 1], [3, 2], [2, 3], [1, 4], [0, 5],
                       [0, 6], [1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 0],
                       [7, 0], [6, 1], [5, 2], [4, 3], [3, 4], [2, 5], [1, 6], [0, 7], ])


class BatchDCT:
    """
    Batch Discrete Cosine Transform
    """
    # TODO: add device
    def __init__(self, freq_order=None, dim_local=5):
        self.dim_local = dim_local
        self.freq_order = 3 if freq_order is None else freq_order
        if (self.freq_order ** 2 + self.freq_order) % 2 != 0:
            raise ValueError("Wrong number of frequencies!")
        self.n_freq = (self.freq_order ** 2 + self.freq_order) // 2
        self.indices = torch.tensor([i[0] * self.dim_local + i[1] for i in zigzag[:self.n_freq]])
        # r_ind, c_ind = np.triu_indices(8, 8 - self.freq_order)
        # self.indices = tuple([zigzag[:self.n_freq, 0], zigzag[:self.n_freq, 1]])

    def dct(self, spatial: torch.Tensor):
        """
        transform a [num_batch, dim_local, dim_local] tensor in the spatial domain to [num_batch, num_freq] frequency descriptor
        """
        assert len(spatial.shape) == 3 and spatial.shape[1] == self.dim_local and spatial.shape[2] == self.dim_local,\
            "The input shape should be [num_batch, dim_local, dim_local]!"
        return torch_dct.dct_2d(spatial).reshape(spatial.shape[0], self.dim_local ** 2)[:, self.indices]

    def idct(self, freq: torch.Tensor):
        """
        transform a [num_batch, num_freq] frequency descriptor to [num_batch, dim_local, dim_local] tensor in the spatial domain
        """
        assert len(freq.shape) == 2 and freq.shape[1] == self.n_freq, \
            "The input shape should be [num_batch, num_freq]!"
        all_freq = torch.zeros(freq.shape[0], self.dim_local ** 2)
        all_freq[:, self.indices] = freq
        return torch_dct.idct_2d(all_freq.reshape(freq.shape[0], self.dim_local, self.dim_local))


def main():
    # a = np.random.random((8, 8)).astype(np.float32)
    dct_handler = BatchDCT()
    # a = np.ones((8, 8)).astype(np.float32)
    # a = torch.ones(2, 5, 5)
    a = torch.rand(50).reshape(2, 5, 5)
    print(a)
    f = dct_handler.dct(a)
    print(f)
    b = dct_handler.idct(f)
    print(b)
    print((a - b).sum())

    # plt.imshow(a)
    # plt.show()
    # # plt.imshow(f)
    # # plt.show()
    # plt.imshow(b)
    # plt.show()


if __name__ == '__main__':
    main()
