from util import *
import matplotlib.pyplot as plt
from tqdm import tqdm

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''

    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28, 28], is_top=False, n_labels=10, batch_size=10, name=""):
        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
        self.name = name
        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom
        if is_bottom:
            self.image_size = image_size
        self.is_top = is_top
        if is_top:
            self.n_labels = 10

        self.batch_size = batch_size
        self.delta_bias_v = 0
        self.delta_weight_vh = 0
        self.delta_bias_h = 0
        self.bias_v = np.zeros(self.ndim_visible)#np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))
        self.weight_vh = np.random.normal(
            loc=0.0, scale=0.1, size=(self.ndim_visible, self.ndim_hidden))
        self.bias_h = -4*np.ones((1,self.ndim_hidden))#np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        self.delta_weight_v_to_h = 0
        self.delta_weight_h_to_v = 0
        self.weight_v_to_h = None
        self.weight_h_to_v = None
        self.learning_rate = 0.01
        self.momentum = 0.0
        self.print_period = 500

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 1000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            # pick some random hidden units
            "ids": np.random.randint(0, self.ndim_hidden, 36)
        }

    def cd1(self, visible_trainset, n_iterations=30000):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        recon_loss = {"it":[], "loss":[]}
        print("learning CD1")

        n_samples = visible_trainset.shape[0]
        print("n_samples", n_samples)

        for it in tqdm(range(n_iterations)):
            minibatch_start = it * self.batch_size % n_samples
            minibatch_end = minibatch_start + self.batch_size
            
            v0 = visible_trainset[minibatch_start:minibatch_end,:]
            _, h0 = self.get_h_given_v(v0)
            pvk, vk = self.get_v_given_h(h0)
            phk, _ = self.get_h_given_v(vk)
            self.update_params(v0, h0, pvk, phk)

            # visualize once in a while when visible layer is input images

            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape(
                    (self.image_size[0], self.image_size[1], -1)), it=it, grid=self.rf["grid"])

            #print epoch number: 
            if it % n_samples == 0:
                print(f'Epoch {it//n_samples}')

            # print progress
            if it % self.print_period == 0:
                ph0, h0 = self.get_h_given_v(visible_trainset)
                pvk, vk = self.get_v_given_h(h0)
                recon_loss["it"].append(it)
                recon_loss["loss"].append(
                    np.linalg.norm(visible_trainset - vk))
                print("iteration=%7d recon_loss=%4.4f" %
                      (it, recon_loss["loss"][-1]))
            
        self.plot_loss(recon_loss)
        # self.save_weights()
        return

    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters.
        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """
        vh0 = v_0.T @ h_0
        vh1 = v_k.T @ h_k
        lr = self.learning_rate/self.batch_size

        self.delta_bias_v = self.momentum * self.delta_bias_v + lr * np.sum((v_0 - v_k), axis=0)
        self.delta_weight_vh = self.momentum * self.delta_weight_vh  + lr * (vh0 - vh1)
        self.delta_bias_h = self.momentum * self.delta_bias_h + lr * np.sum((h_0 - h_k), axis=0)

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        assert self.weight_vh is not None
        n_samples = visible_minibatch.shape[0]
        probabilities = sigmoid(self.bias_h + visible_minibatch @ self.weight_vh)
        activations = sample_binary(probabilities)
        
        return probabilities, activations

    def get_v_given_h(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)
        Uses undirected weight "weight_vh" and bias "bias_v"
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        assert self.weight_vh is not None
        n_samples = hidden_minibatch.shape[0]

        if self.is_top:
            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            support = self.bias_v + hidden_minibatch @  self.weight_vh.T
            
            probabilities_labels = softmax(support[:,-self.n_labels:])
            activations_labels = sample_categorical(probabilities_labels)

            probabilities_h2 = sigmoid(support[:,:-self.n_labels])
            activations_h2 = sample_binary(probabilities_h2)
            
            probabilities = np.concatenate(
                (probabilities_h2, probabilities_labels), axis=1)
            activations = np.concatenate(
                (activations_h2, activations_labels), axis=1)

        else:
            probabilities = sigmoid(self.bias_v + hidden_minibatch @ self.weight_vh.T)
            activations = sample_binary(probabilities)

        return probabilities, activations

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):  
        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        assert self.weight_v_to_h is not None
        n_samples = visible_minibatch.shape[0]
        probabilities = sigmoid(self.bias_h + visible_minibatch @  self.weight_v_to_h)
        activations = sample_binary(probabilities)
        
        return probabilities, activations


    def get_v_given_h_dir(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)
        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            raise Exception('Top DBN is a bipartite undirected graph!')

        else:
            probabilities = sigmoid(self.bias_v + hidden_minibatch @  self.weight_h_to_v)
            activations = sample_binary(probabilities)

        return probabilities, activations


    def update_generate_params(self, inps, trgs, preds):
        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, preds):
        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return

    def save_weights(self):
        np.save(f"trained_rbm/{self.name}_weights_{self.ndim_hidden}_{self.batch_size}.npy", self.weight_vh)
        np.save(
            f"trained_rbm/{self.name}_bias_v_{self.ndim_hidden}_{self.batch_size}.npy", self.bias_v)
        np.save(
            f"trained_rbm/{self.name}_bias_h_{self.ndim_hidden}_{self.batch_size}.npy", self.bias_h)

    def plot_loss(self, loss):
        plt.title('Reconstruction loss over training iterations')
        plt.xlabel('iteration')
        plt.ylabel('reconstruction loss')
        plt.plot(loss["it"], loss["loss"])
        plt.savefig(
            f"trained_rbm/{self.name}_loss_{self.ndim_hidden}_{self.batch_size}.png")
        plt.show()
