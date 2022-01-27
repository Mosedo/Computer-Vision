#genetic_optimization.py
import math
import random
import numpy as np

best_soln=0

def problem(x):
    return 20*x**2+20*x-55

def fitness(x):
    soln=problem(x)
    if soln==0:
        return 99999
    else:
        return abs(1/soln)

def fit(bs):
    return 20*bs[1]**2+20*bs[1]

population=[]

for npop in range(1000):
    population.append(random.uniform(0,10000))

for g in range(10000):
    rankedSolutions=[]
    for p in population:
        rankedSolutions.append((fitness(p),p))
    
    sort_by=lambda ranked:ranked[0]
    # rankedSolutions.sort(key=sort_by,reverse=True)
    rankedSolutions.sort(reverse=True)


    if rankedSolutions[0][0] > 9999:
        best_soln=rankedSolutions[0]
        break

    pool=rankedSolutions[:100]


    elements=[]


    for elem in pool:
        elements.append(elem[1])
    
    new_solutions=[]

    for n in range(1000):
        new_solutions.append(random.choice(elements)*np.random.uniform(0.99,1.01))
    
    population=new_solutions


    print(f"======= GEN ======= {g}")
    print(rankedSolutions[0])

print("****************Solution****************")
print()
print(fit(best_soln))
