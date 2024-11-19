# The City College of New York, City University of New York
# Written by Olga Chsherbakova
# Date: September, 2023

# GA for solving Traveling Salesman Problem 
from deap import creator, base, tools, algorithms
import random
import gps
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

cities_file = 'cities.csv'
#cities_file = 'cities_v2.csv'

## load city data into a dataframe df
df = pd.read_csv(cities_file)
## generate a weight matrix where the weights are the distances between cities
distance_matrix = gps.generate_distance_matrix(df)

## set chromosome length to the number of cities
CHROM_LENGTH = len(distance_matrix)

if hasattr(creator, "FitnessMin"):
   del creator.FitnessMin
if hasattr(creator, "Individual"):
   del creator.Individual

# create a variable called FitnessMin with the create() function.
# FitnessMin inherits from base.Fitness and it will store whether
# the fitness should be minimized or maximized:
#      if weights == -1.0, then minimize the fitness
#      if weights == 1.0, then maximize the fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# create a variable called Individual that will determine the 
# datastructure used for the chromosome.
# Individual inherits from list and it will store integers.
creator.create("Individual", list, typecode='i', fitness=creator.FitnessMin)

# initialize a new toolbox variable to configure GA parameters:
toolbox = base.Toolbox()

# configure the population of the GA so that each chromosome is initialized
# as a random permutation of the sequence from 0 to CHROM_LENGTH-1
# range(CHROM_LENGTH): creates a list from 0 to CHROM_LENGTH-1
# range: [0,1,...,9] indices is a random permutation of range
toolbox.register("indices", random.sample, range(CHROM_LENGTH), CHROM_LENGTH)
# assign contents of indices as genes in genome
toolbox.register("genome", tools.initIterate, creator.Individual,
                 toolbox.indices)
# repeat chromosome assignment for each chromosome in the population
toolbox.register("population", tools.initRepeat, list, toolbox.genome)

# define the fitness function 
def TSP_fit_func(chromosome):
    ## sum up the total distance of the path stored in chromosome
    ## initialize distance from last element to the first element
    distance = distance_matrix[chromosome[-1]][chromosome[0]]
    ## sum all distances from i to i+1 for all i
    for i in range(len(chromosome)-1):
        distance += distance_matrix[chromosome[i]][chromosome[i+1]]
    ## return the path's distance as the fitness 
    ## (at least one comma is needed in return in deap - stupid rule)
    return distance,
# configure the evaluate parameter by passing the fitness function
toolbox.register("evaluate", TSP_fit_func)

# implementation of crossover oeprator:
# set the reproduction function to partially matched crossover which
# produces children that are also permutations
toolbox.register("mate", tools.cxPartialyMatched)
# set a 5 percent of mutation, which causes genes in the chromosome to
# be randomly swapped and the result of the mutation is also a permutation
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
# set the selection function to tournment selection, which will pick three
# chromosomes at random and select the one with the best fitness
toolbox.register("select", tools.selTournament, tournsize=3)

# set population size to n=100 (n=20 or n=10 see the impact of n)
pop = toolbox.population(n=100)
# have variable best_genome store the chromosome with the best fitness
best_genome = tools.HallOfFame(1)

# run GA to get the solution
# algorithms.eaSimple(pop, toolbox, xo prob, mut prob, gens, store best)
algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 40, halloffame=best_genome)
# store the solution into best_path
best_path = best_genome[0]
# print best_path
print('\nBEST PATH:\n')
for i in range(len(best_path)):
    print(df['city'][best_path[i]])
print(df['city'][best_path[0]])
cost = 0
print('\nBEST PATH COST:\n')
for i in range(len(best_path)-1):
    tempo = distance_matrix[best_path[i]][best_path[i+1]]
    cost+= tempo
    print(df['city'][best_path[i]], tempo)
print(df['city'][best_path[0]],distance_matrix[len(best_path)-1][best_path[0]])
cost += distance_matrix[len(best_path)-1][best_path[0]]
print("TOTAL COST: ",cost)

# PLOTTING BEST PATH

# create a figure to draw the plot on
fig = plt.figure(figsize=(10, 10))

# determine the size and position of the map
# resolution is set to low to reduce run time
m = Basemap(projection='lcc', resolution='l',
            width=5.5E6, height=3.5E6, 
            lat_0=39, lon_0=-96,)

# set ocean color to blue
m.drawmapboundary(fill_color='#DDEEFF')
# set continent color to orange
m.fillcontinents(color="#FFDDCC")
# draw the coastlines on the map
m.drawcoastlines(color='gray')
# draw country border lines on the map
m.drawcountries(color='black')
# draw state border lines on the map
m.drawstates(color='gray')

# create two empty lists to store the x and y positions for the map plot
X = []
Y = []

for i in range(CHROM_LENGTH):
    ## convert the (longitude, latitude) pair of the current city to 
    ## (x, y) coordinates for plotting
    tempX, tempY = m(df['longitude'][best_path[i]], 
                     df['latitude'][best_path[i]])
    ## add tempX and tempY to the end of X and Y respectively
    X.append(tempX)
    Y.append(tempY)

path_color = 'teal'
# draw the shortest path
plt.plot(X,Y,path_color)
# connect the first and last city of the path to close the loop:
# (-1 means the last element of a list)
plt.plot([X[-1],X[0]],[Y[-1],Y[0]],path_color)
## draw points at each city
plt.plot(X,Y,'ok', markersize=7)

for i in range(CHROM_LENGTH):
    # label each city on the map
    plt.text(X[i],Y[i],'  '+df['city'][best_path[i]],fontsize=10,fontweight='bold')

# give the plot a title
Plot_title = 'Best Route Found for TSP Using GA - Total cost: ' + str(cost)
plt.title(Plot_title, fontsize=20)

# display the plot
plt.show()
