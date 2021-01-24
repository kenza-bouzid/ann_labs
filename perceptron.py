import numpy as np
import matplotlib.pyplot as plt

class Neuron:

    def __init__(self,X,y,bias,eta):
        '''X = np.array(num_features, num_samples)
            y = np.array(num_samples)
            bias = boolean
            eta = float'''

        self.num_features = X.shape[0]
        self.num_samples = X.shape[1]
        self.eta = eta
        self.bias = bias
        if self.bias:
            self.X = np.concatenate((X,np.ones((1,self.num_samples))),axis=0)
        else:
            self.X = X
        self.y = y
        self.W = np.random.rand(self.num_features+self.bias)
        self.weights = list()
        self.minx = np.min(self.X[0])
        self.maxx = np.max(self.X[0])


    def update(self):
        '''computes one update step and saves the old weights'''
        self.weights.append(self.W)
        delta = -self.eta*(self.W@self.X-self.y)@self.X.T
        self.W = self.W+delta
        

    #### for now only usefull if we have 2-dim data ####
    def compute_decision_boudary(self,w):
        '''computes the decision boundary given some weight matrix'''
        w1 = w[0]
        w2 = w[1]
        if self.bias:
            b = w[2]
        else:
            b = 0

        y = np.linspace(self.minx,self.maxx,1000)
        z = -(w1/w2)*y - (b / w2)
        return y,z
    #### for now only usefull if we have 2-dim data ####
    def plot_boundarys(self):
        '''plots all boundarys according to the history of the weights'''
        linewidth=0.2
        for w in self.weights[:-1]:
            linewidth += 0.2
            y,z = self.compute_decision_boudary(w)
            plt.plot(y,z,color='darkblue',linewidth=linewidth)
        y,z = self.compute_decision_boudary(self.weights[-1])
        plt.plot(y,z,color='red',linewidth=linewidth)
        plt.scatter(self.X[0],self.X[1],c=self.y)
        plt.show()

def generate_data(sigma1,sigma2,n=100):
    '''    sigma = np.array(2,2)
    creates 2-dim data at mu1=[1,1] and mu2=[-1,-1] with var sigma1,sigma2'''
    n2 = int(n/2)
    X1 = np.random.multivariate_normal([1,1],sigma1,n2)
    X2 = np.random.multivariate_normal([-1,-1],sigma2,n2)
    y1 = np.ones(n2)
    y2 = -np.ones(n2)
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    plt.scatter(X[:,0],X[:,1],c = y)
    return X.T,y

def create_training_data(seperable):
    '''seperable = boolean'''

    if seperable:
        sigma1 = np.array([[0.5,-0.3],[-0.3,0.5]])
        sigma2 = np.array([[0.5,-0.3],[-0.3,0.5]])
        X,y = generate_data(sigma1,sigma2,n=100)
    else:
        sigma1 = np.array([[1,0.3],[0.1,1]])
        sigma2 = np.array([[1,0.2],[0.2,1]])
        X,y = generate_data(sigma1,sigma2,n=100)
    return X,y
