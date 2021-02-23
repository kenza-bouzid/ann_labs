from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),   
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }       
        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_gibbs_recog = 20
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 5
        self.print_period = 2000
        return

    def recognize(self,true_img,true_lbl):
        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        n_samples = true_img.shape[0]
        n_labels = true_lbl.shape[1]
        vis = true_img # visible layer gets the image data
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        _, hid = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)
        _, pen = self.rbm_stack['hid--pen'].get_h_given_v_dir(hid)

        vis = np.concatenate((pen, lbl), axis=1)

        for _ in range(self.n_gibbs_recog):
            _, top = self.rbm_stack['pen+lbl--top'].get_h_given_v(vis)
            p_vis, vis = self.rbm_stack['pen+lbl--top'].get_v_given_h(top)

        predicted_lbl = p_vis[:,-n_labels:]
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))  
        return

    def generate(self,true_lbl,name):
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        n_labels = true_lbl.shape[1]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        random_vis = np.random.binomial(
            n=n_sample, p=0.5, size=(n_sample, self.sizes["pen"]))
        vis = np.concatenate((random_vis, lbl), axis=1)
        for _ in range(self.n_gibbs_gener):
            _, top = self.rbm_stack['pen+lbl--top'].get_h_given_v(vis)
            _, vis = self.rbm_stack['pen+lbl--top'].get_v_given_h(top)

            _ , hid = self.rbm_stack['hid--pen'].get_v_given_h_dir(vis[:,:-n_labels])
            _, img = self.rbm_stack['vis--hid'].get_v_given_h_dir(hid)
            vis[:,-n_labels:] = lbl
            records.append([ax.imshow(img.reshape(
                self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)])
            
        anim = stitch_video(fig,records).save("hist/%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :
            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
        except IOError :
            print("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            self.rbm_stack['vis--hid'].cd1(
                visible_trainset=vis_trainset, n_iterations=n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")
        
        self.rbm_stack["vis--hid"].untwine_weights()
        _, h1 = self.rbm_stack['vis--hid'].get_h_given_v_dir(
            vis_trainset)
        try: 
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            
        except IOError :
            print("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """
            # changed ph1 to h1
            self.rbm_stack["hid--pen"].cd1(visible_trainset=h1,
                                           n_iterations=n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")
            
        self.rbm_stack["hid--pen"].untwine_weights()
        ph2, h2 = self.rbm_stack['hid--pen'].get_h_given_v_dir(h1)
        try:
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        
        except IOError :

            print ("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            # p_labels = self.clamp_labels(lbl_trainset)
            h2_labels = np.concatenate((h2, lbl_trainset), axis=1)
            self.rbm_stack["pen+lbl--top"].cd1(visible_trainset=h2_labels,
                                           n_iterations=n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    
    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
