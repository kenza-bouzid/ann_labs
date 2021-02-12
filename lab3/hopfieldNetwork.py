import numpy as np 
from tqdm import tqdm

class HopfieldNetwork():
    def __init__(self, patterns, seed=42):
        np.random.seed(seed)
        self.P, self.N = patterns.shape
        self.states = patterns
        self.W = self.states.T @ self.states
        self.max_iter = int(np.log2(self.N))
    
    def check_storage(self):
        for s in self.states:
            self.update_rule(s, self.max_iter)

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
            _, fixed = self.update_rule(pattern, max_iter=self.max_iter, verbose=False)
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
        inter_patterns = []
        for i in range(max_iter):
            inter_patterns.append(pattern)
            if sync:
                pattern = np.sign(self.W @ pattern)
            else: 
                ind = np.random.randint(0, self.N, 1)
                pattern[ind] = np.sign(self.W[ind,:] @ pattern)
        if verbose:
            self.print_result(i, pattern)
        return inter_patterns, pattern


    
