import numpy as np
def load():
    W1 = np.transpose(np.loadtxt("W1.txt", unpack = True))
    W2 = np.transpose(np.loadtxt("W2.txt", unpack = True))
    W3 = np.transpose(np.loadtxt("W3.txt", unpack = True))
    W4 = np.transpose(np.loadtxt("W4.txt", unpack = True))
    b1 = np.transpose(np.loadtxt("b1.txt", unpack = True))
    b2 = np.transpose(np.loadtxt("b2.txt", unpack = True))
    b3 = np.transpose(np.loadtxt("b3.txt", unpack = True))
    b4 = np.transpose(np.loadtxt("b4.txt", unpack = True))
    return W1,W2,W3,W4,b1,b2,b3,b4