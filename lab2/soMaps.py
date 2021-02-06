import numpy as np
from scipy.spatial import distance_matrix

class SOM:

    def __init__(self,num_features, data, num_nodes, seed, grid = (0,0)):
        '''grid is the format of the grid we want to train, if (0,0) we assume a chain'''
        self.num_nodes = num_nodes
        self.nodes = self.generate_nodes(num_nodes,num_features,seed)
        self.distMat = distance_matrix(data,self.nodes)
        #create 2dim mapping
        self.grid = grid
        self.grid_dist, self.mapping, self.reverse_mapping = self.create_mapping(self.grid)

    def generate_nodes(self,num_nodes,num_features,seed):
        np.random.seed(seed)
        return np.random.rand(self.num_nodes,num_features)

    def create_mapping(self,grid):
        '''reuturns list of 2d coordinates and the mapping plus its inverse'''
        x = range(grid[0])
        y = range(grid[1])
        #this builds the cartesian product
        c_product = np.transpose([np.repeat(y, len(x)),np.tile(x, len(y))])
        grid_dist = distance_matrix(c_product,c_product)
        mapping = {i:np.array(c) for i,c in enumerate(c_product)}
        reverse = {tuple(c):i for i,c in enumerate(c_product)}
        return grid_dist, mapping, reverse

    def get_closest_node(self,idx):
        argmin = np.argmin(self.distMat[idx])
        return argmin
    
    def get_update_wheight(self, w_id, neighbour_id):
        '''returns the extra wheights for the neighouring nodes'''
        if w_id == neighbour_id:
            return 1
        w = self.nodes[w_id]
        w_neighbour = self.nodes[neighbour_id]
        dist = np.linalg.norm(w-w_neighbour,ord=2)
        return self.wheight_function(1/dist)
        # wheight = 1/np.linalg.norm(w-w_neighbour,ord=2)
        # if 0 <= wheight <= 0.9:
        #     return wheight
        # else:
        #     return 0.9


    def update_dist(self,data):
        '''update distance matrix'''
        self.distMat = distance_matrix(data,self.nodes)
    
    def update(self,idx, x, eta):
        '''update wheights at index idx given the datapoint x and learning rate eta'''
        w = self.nodes[idx]
        delta_w = x-w
        w = w + eta*delta_w
        self.nodes[idx] = w 

    def wheight_function(self,x):
        out = x/(1+x) #abs not necessary, since x is distance
        return out
    
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

    def get_grid_range(self,idn, neighbour_dist):
        '''returns all indices which are going to be updated according to num_neighbours in a 2dim grid
        the n neigbour grid is computed by the shortest distance of the position of tuples in the grid
        given some distance. (Note: num_neighbours is now a radius) '''
        row = self.grid_dist[idn]
        neighbours = [i for i,v in enumerate(row) if v <= neighbour_dist]
        # temp = sorted(self.tupels, key=lambda x: np.linalg.norm(x-idx,ord=2))[:num_neighbours]
        # neigbours = [self.reverse_mapping[tuple(n)] for n in temp]
        return neighbours

    def get_range(self, circular, idx,num_neighbours):
        '''returns all indices which are going to be updated according to num_neighbours'''
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
        '''returns the linear decreased num_neighbours value for the current epoch c_epoch'''
        delta = (num_neighbours_start- num_neighbours_end)/epochs * (c_epoch+1)
        return num_neighbours_start-round(delta)