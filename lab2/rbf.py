import numpy as np 

class RBF():
    def __init__(self, nh=12):
        self.nh = nh
    
    def batch_learning(self, x, f, centers, sigmas, x_test, f_test):
        phi = np.exp(-(x-centers)*2 / 2*sigmas*2)
