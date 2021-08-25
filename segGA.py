import numpy as np
import random
import sys
import cv2

class GA():
    def __init__(self, images_Normal, images_BW, nc = 3, crossover_prob = 0.5, 
                 mutation_prob = 0.8, selction_prob = 0.1, population_size = 20,
                 iterates = 100, name = "Im0_GA.png"):
        
        self.name = name
        
        self.original_Normal = images_Normal
        self.original_BW = images_BW
        self.images_Normal = cv2.resize(images_Normal, (0, 0), fx = 0.5, fy = 0.5)
        self.images_BW = cv2.resize(images_BW, (0, 0), fx = 0.5, fy = 0.5)
        self.nc = nc
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selction_prob = selction_prob
        self.population_size = population_size
        self.iterates = iterates
        self.genes = []
        self.epsilon = 1 - 1 / self.iterates
        
        self.indices = [i for i in range(self.nc * 3)]
        
        self.popFit = [0] * self.population_size        
        self.generate_G()

        

    def evaluate_fit(self, gene, want):
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
            
    def generate_G(self):
        mid = np.array([0, 0, 0])
        for i in range(self.population_size):
            mid += self.images_Normal[np.random.randint(0, len(self.images_Normal))]\
                [np.random.randint(0, len(self.images_Normal))]
        mid = mid // self.population_size
        # min_ = np.min(self.images_Normal[10]) + 50
        # max_ = np.max(self.images_Normal[10])
        for i in range(self.population_size):
            gene = []
            for j in range(self.nc):
                g = [np.random.randint(mid[0]-50, mid[0]+50), np.random.randint(mid[1]-50, mid[1]+50),
                      np.random.randint(mid[2]-50, mid[2]+50)]
                # g = [np.random.randint(min_, max_), np.random.randint(min_, max_),
                #       np.random.randint(min_, max_)]             
                gene.append(g)
            self.genes.append(gene)
            self.popFit[i] = self.evaluate_fit(gene, False)
        self.genes = np.array(self.genes)
        return self.genes                    
    
    def evolve(self):
        mutate = self.epsilon * self.mutation_prob
        cross =  self.crossover_prob
        #(1 - self.epsilon) 
        crossed_count = int(cross * self.nc * 3 - 1)
        mutated_count = int(mutate * self.nc * 3 - 1)

        cross_couple = []
        parents = self.genes.tolist()
        for _ in range(len(self.genes)//2):
            a = random.sample(parents, 2)
            del parents[parents.index(a[0])]
            del parents[parents.index(a[1])]
            cross_couple.append(a)


        self.offspring = []
        self.offFit = []
        for idx in range(len(cross_couple)):
            for i in range(crossed_count):
                try:
                    gene_index = random.choice(self.indices)
                    x = (cross_couple[idx][0][gene_index//3][gene_index%3] + cross_couple[idx][1][gene_index//3][gene_index%3]) // 2
                    cross_couple[idx][0][gene_index//3][gene_index%3] = x
                    cross_couple[idx][1][gene_index//3][gene_index%3] = x
                except:
                    pass
            
            a = cross_couple[idx][0]
            b = cross_couple[idx][1]
            
            self.offspring.append(a)
            self.offspring.append(b)
            
            self.offFit.append(self.evaluate_fit(a, False))
            self.offFit.append(self.evaluate_fit(a, False))
        
        
        
        for i in range(len(self.offspring)):
            for j in range(mutated_count):
                try:
                    gene_index = random.choice(self.indices)                
                    self.offspring[i][gene_index//3][gene_index%3] = 255 - self.offspring[i][gene_index//3][gene_index%3]
                except:
                    pass
        self.offspring = np.array(self.offspring)

        
    def selection(self):
        select = self.selction_prob
        select_count = int(select * self.population_size)
        for i in range(select_count):
            try:
                min_fit_off = min(self.offFit)
                idx_off = self.offFit.index(min_fit_off)
                spring = self.offspring[idx_off]
                
                max_fit_pop = max(self.popFit)
                idx_pop = self.popFit.index(max_fit_pop)
                
                self.genes[idx_pop] = spring
                self.popFit[idx_pop] = min_fit_off
                
                del self.offspring[idx_off]
                del self.offFit[idx_off]
            except:
                pass
            
    def run(self):
        for i in range(self.iterates):
            self.evolve()
            self.selection()
            self.epsilon -= 1/self.iterates
            self.current_score = min(self.popFit)
            self.current_best_gene = self.genes[self.popFit.index(self.current_score)]
            # if i % 20 == 0:
            #     print(self.current_best_gene, "\nBest score:", self.current_score)
        print("----------Final---------")
        print(self.current_best_gene, "\nBest score:", self.current_score)
        self.evaluate_fit(self.current_best_gene, True)
        return self.current_best_gene, self.current_score
    
    def PSNR(self):
        image_grey = np.sum(self.image, axis = 2) // 3
        L = len(np.unique(self.original_BW))
        d = np.sum((self.original_BW - image_grey) ** 2)
        d = d / (len(self.original_BW) ** 2)
        psnr = 10 * np.log10(L ** 2 / d)
        return psnr
        
    
# img = cv2.imread('F:\Artificial Intelligence\ALL_IDB2\img\Im001_1.tif', 1)
# img1 = cv2.imread('F:\Artificial Intelligence\ALL_IDB2\img\Im001_1.tif',0)
# nima = GA(img, img1)
# nima.evaluate_fit([[161, 160, 157],
#  [212  ,89 ,155],
#  [112 ,142 ,119],
#  [134  ,92 ,103]], True)