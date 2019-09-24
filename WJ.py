# 180602
import random
import operator
import csv
import itertools
import scipy.optimize
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
FEATURE_NUM = 200
CROSS_PROB = 0.6
MUT_PROB = 0.2
GENERATION = 10
SAMPLE_CHOOSE = 1000

with open("trainX.csv") as spambase:
    spamReader = csv.reader(spambase)
    spam = list(list(float(elem) for elem in (row[:FEATURE_NUM]+[row[-1]])) for row in spamReader)
    # print(spam)

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, FEATURE_NUM), bool, "IN")

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)

# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * POPULATION_SIZE, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSpambase(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    spam_samp = random.sample(spam, SAMPLE_CHOOSE)
    result = sum(bool(func(*mail[:FEATURE_NUM])) is bool(mail[FEATURE_NUM]) for mail in spam_samp)
    return result,
    
toolbox.register("evaluate", evalSpambase)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def print_population(pop, hof):
    expr = 0
    for data in [pop, hof]:
        for ind in data:
            print(str(ind))
            expr = ind

    # plotting tree graph
    nodes, edges, labels = gp.graph(expr)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")

def local_search(pop):
    """
    Local Search-> improve avg
    """
    # pop=optimization()
    pop_ro=optimization()

    for i in range(0,POPULATION_SIZE-1):
        # choose better one between pop and pop_ro
        if evalSpambase(pop_ro[i])>evalSpambase(pop[i]):
            pop[i]=pop_ro[i]

    for i in range(0,POPULATION_SIZE-1):
        if i==0:
            # compare with i+1 only
            if evalSpambase(pop[i])<evalSpambase(pop[i+1]):
                # replace with better one
                pop[i]=pop[i+1]

        if i==POPULATION_SIZE-1:
            # compare with i-1 only
            if evalSpambase(pop[i])<evalSpambase(pop[i-1]):
                pop[i]=pop[i-1]
                
        else:
            if evalSpambase(pop[i+1])>max(evalSpambase(pop[i-1]),evalSpambase(pop[i])):
                pop[i]=pop[i+1]
            if evalSpambase(pop[i-1])>max(evalSpambase(pop[i+1]),evalSpambase(pop[i])):
                pop[i]=pop[i-1]


    return pop

def optimization():
    """
    Random Optimization -> improve max
    1. Initialize x with a random position in the search-space.
    2. Until a termination criterion is met
    3. Sample a new position y by adding a normally distributed random vector to the current position x
    If (evalSpambase(y) < evalSpambase(x)) then move to the new position by setting x = y
    This algorithm corresponds to a (1+1) evolution strategy with constant step-size.
    """
    pop = toolbox.population(n=POPULATION_SIZE)

    for i in range(0,POPULATION_SIZE-1):
        ind_x=toolbox.individual()
        ind_y=toolbox.individual()
        if evalSpambase(ind_x)>evalSpambase(ind_y):
            pop[i]=ind_x
        else:
            pop[i]=ind_y

    """
    Luus Jaakola Optimization
    1. Initialize x ~ U(blo,bup) with a random uniform position in the search-space.
    2. Set the initial sampling range to cover the entire search-space (or a part of it): d = bup-blo
    3. Pick a random vector a
    4. Add this to the current position x to create the new potential position y = x + a
    5.If (f(y) < f(x)) then move to the new position by setting x = y, otherwise decrease the sampling-range d
    """
    d=POPULATION_SIZE

    for i in range(0,POPULATION_SIZE-1):
        y=i+random.randint(0,d)
        if y>=POPULATION_SIZE:
            y=POPULATION_SIZE-1
        if evalSpambase(pop[y])>evalSpambase(pop[i]):
            pop[i]=pop[y]
        else:
            d=int(d*0.95)

    return pop

def pop_process():
    pop = toolbox.population(n=POPULATION_SIZE)
    # local search & optimization
    pop=local_search(pop)
    return pop

def main():
    pop=pop_process()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, CROSS_PROB, MUT_PROB, GENERATION, stats, halloffame=hof)

    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()
    #print_population(pop, hof)
