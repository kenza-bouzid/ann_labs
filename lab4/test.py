from util import *
from rbm import RestrictedBoltzmannMachine
import matplotlib.pyplot as plt
if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    print("\nTesting a Restricted Boltzmann Machine..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
                                     )
    
    rbm.weight_vh = np.load("trained_rbm/weights_100_20.npy")
    rbm.bias_v = np.load("trained_rbm/bias_v_100_20..npy")
    rbm.bias_h = np.load("trained_rbm/bias_h_100_20..npy")


    fig, axs = plt.subplots(10, 2, figsize=(12, 12))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # print("MIN WEIGHT", weights.min(), "MAX WEIGHT", weights.max())
    for  i in range(10):
        image = train_imgs[i]
        v0 = np.random.binomial(1,0.2,(784,))
        vk = image.copy()
        for _ in range(1):
            ph0, h0 = rbm.get_h_given_v(vk)
            pvk, vk = rbm.get_v_given_h(h0)
        axs[i, 0].imshow(image.reshape((28,28)))
        axs[i, 1].imshow(pvk.reshape((28, 28)))
    
    
    plt.savefig("hist/recon.png")
    plt.show()
    plt.close('all')






