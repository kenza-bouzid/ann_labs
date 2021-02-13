import math
import numpy as np
from matplotlib import pyplot as plt

# Fixing the RBF ndes in the input space for model training


def competetive(data, nodecount, eta, iterations):
    # np.random.shuffle(data)
    RBF = data[np.random.randint(0, len(data), nodecount)]
    print(RBF)
    temp_data = []
    for j in range(iterations):
        rand_id = np.random.randint(0, len(data))
        randvec = data[rand_id]
        # Removing data to ensure all samples are considered while shifting RBF.
        data = np.delete(data, (rand_id), axis=0)
        temp_data.append(randvec)
        distances = []
        for center in RBF:
            distances = np.append(
                distances, (np.linalg.norm(center - randvec)))
        RBF = RBF[np.argsort(distances)]
        # Moving the nodes depending on the sorted distances.
        for i in range(len(RBF)):
            RBF[i] += (eta * (randvec - RBF[i]))/(np.square(i+1))
        # Reappending the data for further iterations.
        if len(data) == 0:
            data = temp_data
            temp_data = []
    print(RBF)
    return RBF

# Competitive learning using the ballist dataset


def gen_data(rbfNodes):
    train_data = np.loadtxt('data_lab2/ballist.dat')
    test_data = np.loadtxt('data_lab2/balltest.dat')
    train_ip = train_data[:, :2]
    train_op = train_data[:, 2:]

    eta = 0.2
    # Iterations for fixing the RBF nodes
    iterations = 800000
    data = train_ip
    rbfMu = competetive(data, rbfNodes, eta, iterations)
    rbfSigma = np.empty((len(rbfMu)))
    # Swapping RBFMu to find to adjust the sigma values
    for i in range(len(rbfMu)):
        for j in range(len(rbfMu)):
            if rbfMu[i][0] < rbfMu[j][0]:
                # print(i,j)
                rbfMu[i] = rbfMu[i] + rbfMu[j]
                rbfMu[j] = rbfMu[i] - rbfMu[j]
                rbfMu[i] = rbfMu[i] - rbfMu[j]
    print(rbfMu)

    # FInding the sigma values such that they are half the distance from the farthest neighbour.
    for ind, nc in enumerate(rbfMu):
        if ind == 0:
            print(math.hypot(nc[0]-0, nc[1]-0))
            rbfSigma[ind] = math.hypot(nc[0]-0, nc[1]-0)/2
        elif ind == len(rbfMu)-1:
            rbfSigma[ind] = math.hypot(
                nc[0]-rbfMu[ind-1][0], nc[1]-rbfMu[ind-1][1])/4
        else:
            rbfSigma[ind] = max(math.hypot(nc[0]-rbfMu[ind+1][0], nc[1]-rbfMu[ind+1][1]),
                                math.hypot(nc[0]-rbfMu[ind-1][0], nc[1]-rbfMu[ind-1][1]))/4

    # Plotting the RBF nodes
    for i in range(len(train_ip)):
        plt.plot(train_ip[i][0], train_ip[i][1], 'r.')
    for i in range(len(rbfMu)):
        my_circ = plt.Circle(rbfMu[i], rbfSigma[i],
                             facecolor='None', edgecolor='blue')
        plt.gcf().gca().add_artist(my_circ)
    plt.show()
    return rbfMu, rbfSigma, train_data, test_data, eta

# Calculating the output of the RBF nodes


def calc_phi(X, rbfMu, rbfSigma):
    Phi = np.zeros((X.shape[0], rbfMu.shape[0]))
    for i in range(X.shape[0]):
        for j in range(rbfMu.shape[0]):
            Phi[i][j] = gaussianTransfer(X[i], rbfMu[j], rbfSigma[j])
    return Phi

# Training the model


def online_delta_rule(epochs, train_ip, train_op, eta, Phi):
    residualError = []
    instantError = []
    W = np.random.rand(2, Phi.shape[1]) * 0.5
    N = Phi.shape[0]

    for iter in range(epochs):
        resError = []
        instError = []
        for k in range(N):
            guess = np.dot(Phi[k], W.T)
            err = train_op[k]-guess
            resError.append(np.abs(err))
            instErr = (1/2)*np.power(err, 2)
            instError.append(instErr)
            deltaW = eta*np.dot(np.reshape(err, (2, 1)),
                                np.reshape(Phi[k], (1, len(Phi[k]))))
            W = W + deltaW
        instantError.append(np.mean(np.abs(instError)))
        residualError.append(np.mean(resError))

    plt.plot(residualError[5:], 'g-', label='Absolute residual error')
    plt.title('Absolute residual error')
    plt.show()
    plt.plot(instantError[50:], 'g-', label='Average instantaneous error')
    plt.title('Average instantaneous error')
    plt.show()
    return residualError, instantError, W


# Output at the RBF nodes
def gaussianTransfer(x, mu, sigma):
    phi_i = np.exp(np.divide(
        (-np.power((x[0]-mu[0]), 2) - np.power((x[1]-mu[1]), 2)), 2*np.power(sigma, 2)))
    return phi_i


rbfMu, rbfSigma, train_data, test_data, eta = gen_data(18)
Phi_train = calc_phi(train_data[:, :2], rbfMu, rbfSigma)
Phi_train = np.concatenate(
    (Phi_train, np.ones((Phi_train.shape[0], 1))), axis=1)
Phi_test = calc_phi(test_data[:, :2], rbfMu, rbfSigma)
Phi_test = np.concatenate((Phi_test, np.ones((Phi_test.shape[0], 1))), axis=1)
residualError, instantError, W = online_delta_rule(
    20000, train_data[:, :2], train_data[:, 2:], 0.001, Phi_train)
test_pred = np.dot(Phi_test, W.T)
print("Actual = ", test_data[:, 2:])
print("Predicted = ", test_pred)
print("Residual error = ", residualError)
