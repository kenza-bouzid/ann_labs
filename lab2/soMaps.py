import numpy as np
from scipy.spatial import distance_matrix

class SOM:
    def __init__(self,num_features, data, num_nodes, seed, grid = (0,0)):
        self.num_nodes = num_nodes
        self.nodes = self.generate_nodes(num_nodes,num_features,seed)
        self.distMat = distance_matrix(data,self.nodes)
        #create 2dim mapping
        self.grid = grid
        self.tupels, self.mapping, self.reverse_mapping = self.create_mapping(self.grid)

    def generate_nodes(self,num_nodes,num_features,seed):
        np.random.seed(seed)
        return np.random.rand(self.num_nodes,num_features)

    def create_mapping(self,grid):
        x = range(grid[0])
        y = range(grid[1])
        c_product = np.transpose([np.repeat(y, len(x)),np.tile(x, len(y))])
        mapping = {i:np.array(c) for i,c in enumerate(c_product)}
        reverse = {tuple(c):i for i,c in enumerate(c_product)}
        return c_product, mapping, reverse

    def get_closest_node(self,idx):
        argmin = np.argmin(self.distMat[idx])
        return argmin
    
    def get_update_wheight(self, w_id, neighbour_id):
        if w_id == neighbour_id:
            return 1
        w = self.nodes[w_id]
        w_neighbour = self.nodes[neighbour_id]
        wheight = 1/np.linalg.norm(w-w_neighbour,ord=2)
        if 0 <= wheight <= 0.9:
            return wheight
        else:
            return 0.9


    def update_dist(self,data):
        self.distMat = distance_matrix(data,self.nodes)
    
    def update(self,idx, x, eta):
        w = self.nodes[idx]
        delta_w = x-w
        w = w + eta*delta_w
        self.nodes[idx] = w 
    
    def train(self,epochs, eta, data, num_neighbours_start, num_neighbours_end,circular):
        for ep in range(epochs):
            num_neighbours = self.anneal_num_neighbour(num_neighbours_start, num_neighbours_end,epochs,ep)
            for idx, x in enumerate(data):
                idn = self.get_closest_node(idx)
                if self.grid == (0,0):
                    indices = self.get_range(circular,idn, num_neighbours)
                else:
                    indices = self.get_grid_range(idn, num_neighbours)
                for i in indices:
                    wheight = self.get_update_wheight(idn, i)
                    self.update(i,x,wheight * eta)
                self.update_dist(data)

    def get_grid_range(self,idn, num_neighbours):
        idx = self.mapping[idn]
        temp = sorted(self.tupels, key=lambda x: np.linalg.norm(x-idx,ord=2))[:num_neighbours]
        neigbours = [self.reverse_mapping[tuple(n)] for n in temp]
        return neigbours

    def get_range(self, circular, idx,num_neighbours):
        low = idx-round(num_neighbours/2)
        up = idx+round(num_neighbours/2)
        if circular:
            return range(low,up%self.num_nodes)
        else:
            if idx-round(num_neighbours/2) < 0:
                low = 0
            if idx+round(num_neighbours/2) >= self.num_nodes:
                up = self.num_nodes-1
            return range(low,up)


    def anneal_num_neighbour(self,num_neighbours_start, num_neighbours_end,epochs,c_epoch):
        delta = (num_neighbours_start- num_neighbours_end)/epochs * (c_epoch+1)
        return num_neighbours_start-round(delta)