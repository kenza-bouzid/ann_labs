import numpy as np


class TwoLP:

    def __init__(self, nodes_num, features_num, output_dim, seed=42, lr=0.001, alpha=0.9):
        self.nodes_num = nodes_num
        self.features_num = features_num
        self.output_dim = output_dim
        self.hidden_weights = np.random.default_rng(
            seed).normal(0, 0.5, size=[nodes_num, features_num])
        self.output_weights = np.random.default_rng(
            seed).normal(0, 0.5, size=[output_dim, nodes_num+1])
        self.lr = lr
        self.alpha = alpha
        self.output_weights_history = []
        self.output_weights_history.append(self.output_weights)
        self.MSE = []
        self.MSE_val = []

    def activation_function(self, X):
        return 2/(1 + np.exp(-X)) - 1

    def derivative_activation_function(self, phi):
        return np.multiply((1 + phi), (1 - phi)) / 2

    def forward_pass(self, X):
        hin = self.hidden_weights @ X
        hout = np.concatenate((self.activation_function(
            hin), np.ones((1, X.shape[1]))), axis=0)
        oin = self.output_weights @ hout
        out = self.activation_function(oin)

        return hout, out

    def weight_update(self, delta_o, delta_h, d_hidden, d_output, X, hout):
        d_hidden = d_hidden * self.alpha - (delta_h @ X.T) * (1-self.alpha)
        d_output = d_output * self.alpha - (delta_o @ hout.T) * (1-self.alpha)
        self.hidden_weights += d_hidden * self.lr
        self.output_weights += d_output * self.lr
        self.output_weights_history.append(self.output_weights)
        return d_hidden, d_output

    def backward_pass(self, hout, out, y):
        delta_o = np.multiply(
            (out - y), self.derivative_activation_function(out))
        delta_h = np.multiply((self.output_weights.T @ delta_o),
                              self.derivative_activation_function(hout))
        delta_h = delta_h[:self.nodes_num, :]
        return delta_o, delta_h

    def train(self, X, y, X_test=[], y_test=[], verbose=False, epochs=100):
        # add bias
        X = np.c_[X, np.ones(X.shape[0])]
        if X_test != []:
            X_test = np.c_[X_test, np.ones(X_test.shape[0])]
            X_test = X_test.T

        # make row represent one dimension (as in the instruction)
        X = X.T

        d_hidden = np.zeros((self.nodes_num, self.features_num))
        d_output = np.zeros((self.output_dim, self.nodes_num+1))
        for epoch in range(epochs):
            hout, out = self.forward_pass(X)
            if X_test != []:
                _, y_pred = self.forward_pass(X_test)
                self.MSE_val.append(0.5 * np.mean((y_pred - y_test)**2)) 
            mse = 0.5 * np.mean((out - y)**2)
            self.MSE.append(mse)
            delta_o, delta_h = self.backward_pass(hout, out, y)
            d_hidden, d_output = self.weight_update(
                delta_o, delta_h, d_hidden, d_output, X, hout)
            if not verbose:
                print(f'Epoch {epoch}, MSE:{mse}')


class TwoLP_seq:

    def __init__(self, nodes_num, features_num, output_dim, seed=42, lr=0.001, alpha=0.9):
        self.nodes_num = nodes_num
        self.features_num = features_num
        self.output_dim = output_dim
        self.hidden_weights = np.random.default_rng(
            seed).normal(0, 0.5, size=[nodes_num, features_num])
        self.output_weights = np.random.default_rng(
            seed).normal(0, 0.5, size=[output_dim, nodes_num+1])
        self.lr = lr
        self.alpha = alpha
        self.output_weights_history = []
        self.output_weights_history.append(self.output_weights)
        self.MSE = []
        self.MSE_val = []

    def activation_function(self, X):
        return 2/(1 + np.exp(-X)) - 1

    def derivative_activation_function(self, phi):
        return np.multiply((1 + phi), (1 - phi)) / 2

    def forward_pass(self, X):
        hin = self.hidden_weights @ X
        hout = np.concatenate((self.activation_function(
            hin), np.ones(1)), axis=0)
        oin = self.output_weights @ hout
        out = self.activation_function(oin)

        return hout, out

    def weight_update(self, delta_o, delta_h, d_hidden, d_output, X, hout):
        d_hidden = d_hidden * self.alpha - (delta_h[:,None] * X) * (1-self.alpha)
        d_output = d_output * self.alpha - (delta_o * hout) * (1-self.alpha)
        self.hidden_weights += d_hidden * self.lr
        self.output_weights += d_output * self.lr
        self.output_weights_history.append(self.output_weights)
        return d_hidden, d_output

    def backward_pass(self, hout, out, y):
        delta_o = np.multiply(
            (out - y), self.derivative_activation_function(out))
        delta_h = np.multiply((self.output_weights.T @ delta_o),
                              self.derivative_activation_function(hout))
        delta_h = delta_h[:self.nodes_num]
        return delta_o, delta_h
    
    def predict(self, X):
        y_pred = list()
        for x in X.T:
            _, pred = self.forward_pass(x)
            y_pred.append(pred[0])
        return y_pred

    def train(self, X, y, X_test=[], y_test=[], verbose=False, epochs=100):
        # add bias
        X = np.c_[X, np.ones(X.shape[0])]
        if X_test != []:
            X_test = np.c_[X_test,np.ones(X_test.shape[0])]
            X_test = X_test.T

        # make row represent one dimension (as in the instruction)
        #X = X.T

        d_hidden = np.zeros((self.nodes_num, self.features_num))
        d_output = np.zeros((self.output_dim, self.nodes_num+1))
        
        for epoch in range(epochs):
            out_list = np.zeros(X.shape[0])
            for idx,x in enumerate(X):
                hout, out = self.forward_pass(x)
                out_list[idx] = out
                delta_o, delta_h = self.backward_pass(hout, out, y[idx])
                d_hidden, d_output = self.weight_update(
                    delta_o, delta_h, d_hidden, d_output, x, hout)
            mse = 0.5 * np.mean((out_list - y)**2)
            self.MSE.append(mse)
            if X_test != []:
                y_pred = self.predict(X_test)
                self.MSE_val.append(0.5 * np.mean((y_pred - y_test)**2)) 
            if not verbose:
                print(f'Epoch {epoch}, MSE:{mse}')