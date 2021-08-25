import numpy as np
import numpy.random as rn
import cv2
import sys
import time

start = time.time()

class SimAnn():
    def __init__(self, images_Normal, images_BW, nc = 3, name = 'Im0_AN.png'):
        
        self.name = name
        
        self.nc = nc
        self.original_Normal = images_Normal
        self.original_BW = images_BW
        self.images_Normal = cv2.resize(images_Normal, (0, 0), fx = 0.5, fy = 0.5)
        self.images_BW = cv2.resize(images_BW, (0, 0), fx = 0.5, fy = 0.5)
        self.interval = (0, 255)
        self.maxsteps = 10000
        self.debug = True
        self.clip = np.vectorize(self.clip)
    
    def annealing(self):
        state = self.random_start()
        cost = self.cost_function(state, False)
        states, costs = [state], [cost]
        for step in range(self.maxsteps):
            fraction = step / float(self.maxsteps)
            T = self.temperature(fraction)
            new_state = self.random_neighbour(state, fraction)
            new_cost = self.cost_function(new_state, False)
           
            if self.acceptance_probability(cost, new_cost, T) > rn.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
        return state, self.cost_function(state, True), states, costs
    def clip(self, x):
        a, b = self.interval
        return int(max(min(x, b), a))
    
    def random_start(self):
        mid = 0
        for i in range(30):
            mid += self.images_BW[np.random.randint(0, len(self.images_BW))][np.random.randint(0, len(self.images_BW))]
        mid = mid // 30
        return np.random.randint(mid-50, mid+50, size = (self.nc, 3))
    
    def cost_function(self, gene, want):
        all_d = []
        for color in gene:
            cl = np.array([[color]*len(self.images_BW)]*len(self.images_BW[0]))
            d = np.sqrt(np.sum(((self.images_Normal - cl) ** 2), axis = 2))
            all_d.append(d)
        if self.nc == 3:
            x = np.minimum(all_d[0], all_d[1])
            y = np.minimum(x, all_d[2])
            fit = np.sum(y) / len(self.images_Normal) ** 2
        if self.nc == 4:
            x = np.minimum(all_d[0], all_d[1])
            y = np.minimum(x, all_d[2])
            z = np.minimum(y, all_d[3])
            fit = np.sum(z) / len(self.images_Normal) ** 2
        if self.nc == 5:
            x = np.minimum(all_d[0], all_d[1])
            y = np.minimum(x, all_d[2])
            z = np.minimum(y, all_d[3])
            w = np.minimum(z, all_d[4])
            fit = np.sum(w) / len(self.images_Normal) ** 2           

        if want:
            image = []
            for i in range(len(self.original_Normal)):
                row = []
                for j in range(len(self.original_Normal[0])):
                    mincl = sys.maxsize
                    for color in gene:   
                        d = np.sqrt(np.sum((self.original_Normal[i][j] - color) ** 2))
                        if d < mincl:
                            p_color = color
                            mincl = d
                    row.append(p_color)
                image.append(row)
            self.image = np.array(image)
            cv2.imwrite(self.name, self.image)
        return fit
    
    def random_neighbour(self, x, fraction=1):
        amplitude = (max(self.interval) - min(self.interval)) * fraction / 10
        delta = np.array([[0] * 3] * self.nc)
        for i in range(self.nc):
            for j in range(3):
                delta[i][j] = (-amplitude/2.) + amplitude * rn.random_sample()
        return self.clip(x + delta)
    
    def acceptance_probability(self, cost, new_cost, temperature):
        if new_cost < cost:
            return 1
        else:
            p = np.exp(- (new_cost - cost) / temperature)
            return p
 
    def temperature(self, fraction):
        return max(0.01, min(1, 1 - fraction))

    def PSNR(self):
        image_grey = np.sum(self.image, axis = 2) // 3
        L = len(np.unique(self.original_BW))
        d = np.sum((self.original_BW - image_grey) ** 2)
        d = d / (len(self.original_BW) ** 2)
        psnr = 10 * np.log10(L ** 2 / d)
        return psnr
    
# images_Normal = cv2.imread('F:\Artificial Intelligence\ALL_IDB2\img\Im040_1.tif', 1)
# images_BW = cv2.imread('F:\Artificial Intelligence\ALL_IDB2\img\Im040_1.tif', 0)
    
# nima = SimAnn(images_Normal, images_BW, nc = 5)
# a, b, c ,d = nima.annealing()
# print(a)
# print("Fitness:", b)
# print("PSNR:", nima.PSNR())

# end = time.time()
# print("Time:", end - start)