from imageOpen import image_opener as im

from segGA import GA
from segAN import SimAnn


images_Normal, images_BW = im()


image_number = 8 # From 0 to 48
n_cluster = 3 # 3 or 4 or 5

print("###### Genetic Algorithm ######")
genetic = GA(images_Normal[image_number], images_BW[image_number], nc = n_cluster, 
              crossover_prob = 0.5, mutation_prob = 0.8,
              selction_prob = 0.1, population_size = 20,
              iterates = 100, 
              name = 'Im%.3d_GA_%.d.png'%(image_number + 1, n_cluster))

gene, score = genetic.run()
psnr = genetic.PSNR()
print("PSNR:", psnr)

print("\n###### Simulated Annealing Algorithm ######")
SA = SimAnn(images_Normal[image_number], images_BW[image_number], nc = n_cluster,
            name = 'Im%.3d_AN_%.d.png'%(image_number + 1, n_cluster))

a, b, c ,d = SA.annealing()
print(a)
print("Fitness:", b)
print("PSNR:", SA.PSNR())

SA.cost_function([[150, 175, 87], [128, 173, 31], [140, 175, 133]], True)

# genetic.evaluate_fit([[150, 128, 140],
#   [175  ,173 ,175],
#   [87 ,31 ,133]], True)        