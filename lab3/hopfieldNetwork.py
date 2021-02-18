import numpy as np 
from tqdm import tqdm
from utils import add_noise

class HopfieldNetwork():
    def __init__(self, patterns, seed=42, sparse=False, bias=0):
        np.random.seed(seed)
        self.sparse = sparse
        self.P, self.N = patterns.shape
        self.states = patterns
        self.W = self.compute_wheight_matrix()
        self.max_iter = int(np.log2(self.N))
        self.bias = bias
    
    def zero_self_connection(self):
        for i in range(self.W.shape[0]):
            self.W[i,i] = 0


    def compute_wheight_matrix(self):
        if self.sparse:
            p = 1/(self.P* self.N) * np.sum(self.states)
            return (self.states.T-p) @ (self.states-p)
        else:
            return (self.states.T @ self.states) / self.N


    def check_storage(self):
        for s in self.states:
            self.update_rule(s, self.max_iter)
    

    def check_capacity(self,noise=1):
        count = 0
        for s in self.states:
            if noise:
                temp_s = add_noise(s, noise_frac=0.05)
                _, new_pattern, _ = self.update_rule(s, self.max_iter, verbose=False)
            else:
                _, new_pattern, _ = self.update_rule(s, 1, verbose=False)
            count = count + np.array_equal(new_pattern, s)
        return count   

    def is_in_states(self, pattern):
        for i, s in enumerate(self.states):
            if np.array_equal(s, pattern):
                return i, s
        return None

    def find_attractors(self): 
        attractors = []
        for i in range(2**self.N):
            binary = bin(i)[2:].zfill(self.N)
            pattern = np.array([int(bit) for bit in binary])
            pattern[pattern == 0] = -1
            _, fixed, _ = self.update_rule(pattern, max_iter=self.max_iter, verbose=False)
            try:
                ind, state = self.is_in_states(fixed)
            except:
                attractors.append(pattern)
        attractors = np.array(attractors)
        return attractors

    def print_result(self, iter, pattern):
        print(f"Fixed Point found after {iter} iterations!")
        try:
            ind, state = self.is_in_states(pattern)
            print(f"Convergence towards stored pattern nb {ind}!")
            print(state, "\n")
        except:
            print(f"This pattern is an attractor!")
            print(pattern, "\n")

    def update_rule(self, pattern, max_iter, sync=True, verbose=True):
        old_pattern = pattern.copy()
        new_pattern = pattern.copy()
        inter_patterns = []
        energy = []
        for i in range(max_iter):
            inter_patterns.append(old_pattern)
            if sync:
                new_pattern = np.sign(self.W @ old_pattern-self.bias)
                if self.sparse:
                    new_pattern = 0.5 + 0.5 * new_pattern
            else: 
                ind = np.random.randint(0, self.N, 1)
                new_pattern[ind] = np.sign(self.W[ind,:] @ old_pattern-self.bias)
                if self.sparse:
                    new_pattern[ind] = 0.5 + 0.5 * new_pattern[ind]
                
            new_energy = self.energy(old_pattern)
            # if i > 1 and new_energy == energy[-1]:
                # break 
            energy.append(new_energy)
            old_pattern = new_pattern.copy()
        if verbose:
            self.print_result(i+1, new_pattern)
        inter_patterns = np.array(inter_patterns)
        return inter_patterns, new_pattern, energy

    def energy(self, state):
        return - state @ self.W @ state
    
