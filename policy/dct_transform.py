import numpy as np
import matplotlib.pyplot as plt
import cv2


zigzag = np.array([[0, 0],
                   [1, 0], [0, 1],
                   [0, 2], [1, 1], [2, 0],
                   [3, 0], [2, 1], [1, 2], [0, 3],
                   [0, 4], [1, 3], [2, 2], [3, 1], [4, 0],
                   [5, 0], [4, 1], [3, 2], [2, 3], [1, 4], [0, 5],
                   [0, 6], [1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 0],
                   [7, 0], [6, 1], [5, 2], [4, 3], [3, 4], [2, 5], [1, 6], [0, 7], ])


class DCT:
    def __init__(self, freq_order=None):
        self.freq_order = 4 if freq_order is None else freq_order
        self.n_freq = self.freq_order ** 2 - 2 * self.freq_order
        # r_ind, c_ind = np.triu_indices(8, 8 - self.freq_order)
        self.indices = tuple([zigzag[:self.n_freq, 0], zigzag[:self.n_freq, 1]])

    def dct(self, spatial):
        """
        transform a [8, 8] array in the spatial domain to [10, ] frequency descriptor
        """
        return cv2.dct(spatial.astype(np.float32))[self.indices]

    def idct(self, freq):
        """
        transform a [10, ] frequency descriptor to [8, 8] array in the spatial domain
        """
        all_freq = np.zeros((8, 8))
        all_freq[self.indices] = freq
        return cv2.idct(all_freq.astype(np.float32))


def main():
    # a = np.random.random((8, 8)).astype(np.float32)
    dct_handler = DCT()
    # a = np.ones((8, 8)).astype(np.float32)
    a = np.array(range(64)).reshape((8, 8)).astype(np.float32)
    f = dct_handler.dct(a)
    print(f)
    b = dct_handler.idct(f)

    plt.imshow(a)
    plt.show()
    # plt.imshow(f)
    # plt.show()
    plt.imshow(b)
    plt.show()


if __name__ == '__main__':
    main()
