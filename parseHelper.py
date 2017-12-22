import numpy as np


def toOneHot(string):
    """ takes string and returns one_hot matrix from it """

    one_hot_mat = np.zeros((255, len(string))) #255 ascii chars, as long as string

    for count, ch in enumerate(string):
        one_hot_mat[ord(ch)][count] = 1

    return one_hot_mat

def toString(np_array):
    """ takes onehot encoded string and transforms it back to ascii"""

    rows, cols = np.shape(np_array)
    string = ""
    mat_transposed = np.transpose(np_array)

    for col in range(cols):
        string += chr(np.argmax(mat_transposed[col]))
    return string
