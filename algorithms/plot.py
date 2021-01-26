import numpy as np
import matplotlib.pyplot as plt

def compute_decision_boudary(w, bias):
    '''computes the decision boundary given some weight matrix'''
    w1 = w[0]
    w2 = w[1]
    if bias:
        b = w[2]
    else:
        b = 0

    y = np.linspace(-5,5,1000)
    z = -(w1/w2)*y - (b / w2)
    return y,z
#### for now only usefull if we have 2-dim data ####
def plot_boundarys(weight_history,X,y,typ, lr, bias=1):
    '''plots all boundarys according to the history of the weights'''
    linewidth=0.1
    for w in weight_history[:-1]:
        linewidth += 0.1
        y1,z = compute_decision_boudary(w, bias)
        plt.plot(y1,z,color='darkblue',linewidth=linewidth)
    y1,z = compute_decision_boudary(weight_history[-1], bias)
    plt.plot(y1,z,color='red',linewidth=linewidth)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(f'lr = {lr}, type = {typ}')
    plt.show()