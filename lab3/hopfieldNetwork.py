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
                new_pattern = np.sign(self.W @ old_pattern)
            else: 
                ind = np.random.randint(0, self.N, 1)
                new_pattern[ind] = np.sign(self.W[ind,:] @ old_pattern)
                
            new_energy = self.energy(old_pattern)
            # if i > 1 and new_energy == energy[-1]:
                # break 
            energy.append(new_energy)
            old_pattern = new_pattern.copy()
        if verbose:
            self.print_result(i, new_pattern)
        inter_patterns = np.array(inter_patterns)
        return inter_patterns, new_pattern, energy

    def energy(self, state):
        return - state @ self.W @ state
    
