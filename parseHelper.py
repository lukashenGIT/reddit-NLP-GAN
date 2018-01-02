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

def encodeComment(comment, wordList):
    """ comment = String; wordList = List """
    """ Predefine dimensions """
    comment_length = 10
    wordList_length = len(wordList)
    comment_mat = np.zeros((wordList_length, comment_length))

    """ Build Matrix """
    for count in range(comment_length):
        try:
            #print(wordList)
            idx = np.where(np.array(wordList) == comment[count])[0]
            comment_mat[idx, count] = 1
            """ uncomment for debug """
            #print("(%s, %s)" % (idx, count))
        except IndexError:
            """ uncomment for debug """
            #print("padding")
            pass

    return comment_mat

def decodeComment(comment_onehot, wordList):
    """ comment_onehot = np_array; wordList = List """
    """ Predefine dimensions """
    comment_length = 10
    comment = ""

    """ Build Comment """
    for col in range(comment_length):
        try:
            idx = np.where(comment_onehot == 1)[0]
            comment += (wordList[idx[col]] + " ")
        except IndexError:
            comment += "§PAD$ "

    comment.rstrip()

    return comment
