import numpy as np
from functools import *
"""
1. Set k := 0. Generate an initial population P(0).
2. Evaluate P(k).
3. If the stopping criterion is satisfied, then stop.
4. Select M(k) from P(fc).
5. Evolve M(k) to form P(k + 1).
6. Set k := k + 1, go to step 2.
"""


def main():
    c1 = np.array([1, 1])
    c2 = np.array([1, -1])
    c3 = np.array([-1, 1])
    c4 = np.array([-1, -1])
    # print(withinConstraintsOrigin([c1, c2, c3, c4], np.array([0, 0])))
    c1 = np.array([2, 1])
    c2 = np.array([1, 1])
    c3 = np.array([1, 2])
    c4 = np.array([2, 2])
    # print(withinConstraints([c1, c2, c3, c4], np.array([0, 0])))
    # print(initpopulation([c1, c2, c3, c4], 10))
    # print(evalpopulation(lambda x: np.sum(x), initpopulation([c1, c2, c3, c4], 10)))
    # selection(evalpopulation(lambda x: np.sum(x), initpopulation([c1, c2, c3, c4], 10)), 10)
    # print(evolution(selection(evalpopulation(lambda x: np.sum(x), initpopulation([c1, c2, c3, c4], 10)), 10), [c1, c2, c3, c4], .75, .075, 10))
    geneticAlgorithm(lambda x: np.sum(x), [c1, c2, c3, c4], 10, .75, .175, 10**-2)


# Needs: Function, constraints: array of points, population size, pc, pm, tol
# Later: type of crossover and mutation
# For now, random vectors or convex combos
# For now, mutation is vector jump
# For now, constraints are purely convex and squares
# Later, function defining validity
def geneticAlgorithm(f, constraints, pop, pc, pm, tol):
    pk1 = evolution(selection(evalpopulation(lambda x: np.sum(x), initpopulation(constraints, pop)), pop), constraints, pc, pm, pop)
    i = 0
    while i < 20:
        pk1 = evolution(selection(evalpopulation(lambda x: np.sum(x), pk1), pop), constraints, pc, pm, pop)
        i += 1
    print(pk1)
    return pk1


"""perturb the two points above by some random amount: 
z1 = ax + (1 — a)y + W1 and z2 = (1—a)x + ay +W2, 
where W1 and W2 are two randomly generated vectors (with zero mean)"""


def evolution(matingPool, constraints, pc, pm, pop):
    alpha = np.random.random()
    # print(alpha)
    w1 = np.random.rand(constraints[0].size)/4
    w2 = -1 * w1
    # print(w1, w2)
    pk1 = []
    # crossover
    while len(pk1) < len(matingPool):
        for p in matingPool:
            if np.random.random() < pc and len(pk1) < len(matingPool):
                # it mates with random selected member (mating pool has already passed 'selection')
                # print(np.shape(matingPool))
                # print(matingPool)
                x = np.random.randint(0, np.shape(matingPool)[0] - 1)
                # print(x)
                mate = matingPool[x][1]
                alpha = np.random.random()
                z1 = inConstraintCheck(constraints, alpha * p[1] + (1 - alpha) * mate + w1)
                z2 = inConstraintCheck(constraints, (1 - alpha) * p[1] + alpha * mate + w2)
                pk1.append(z1)
                pk1.append(z2)
    for p in pk1:
        if np.random.random() < pm:
            w = np.random.random(np.size(p))
            p = p + w
    # mutation on new pool P(k+1)
    # print(pk1)
    # needs to return a numpy matrix
    # print(np.stack(pk1))
    return np.stack(pk1)


# tournament scheme
def selection(populationScores, pop):
    # f(xi)/sum(f(xi)
    # cumulative sum
    matingPool = []
    for i in range(pop):
        first = populationScores[np.random.randint(0, pop-1)]
        print(len(populationScores))
        print(pop)
        second = populationScores[np.random.randint(0, pop-1)]
        matingPool.append(first if first[0] > second[0] else second)
    # print(matingPool)
    return matingPool


def evalpopulation(f, populationMatrix):
    return [(f(populationMatrix[i]), populationMatrix[i]) for i in range(np.shape(populationMatrix)[0])]


def initpopulation(constraints, pop):
    # find min/max for each dimension
    # numpy rand between/min max n times for each pop vector
    n = np.size(constraints[0])
    constraintmatrix = np.stack(constraints)
    print(constraintmatrix)
    print(constraintmatrix.T)
    minboundvector = np.zeros(np.size(constraints[0]))
    maxboundvector = np.zeros(np.size(constraints[0]))
    for i in range(np.shape(constraintmatrix.T)[0]):
        minboundvector[i] = min(constraintmatrix.T[i])
        maxboundvector[i] = max(constraintmatrix.T[i])
    print(minboundvector)
    print(maxboundvector)
    population = np.zeros((n, pop))
    for j in range(n):
        population[j] = np.random.uniform(minboundvector[j], maxboundvector[j], pop)
    print(population.T)
    return population.T


def inConstraintCheck(constraints, p):
    constraintmatrix = np.stack(constraints)
    minboundvector = np.zeros(np.size(constraints[0]))
    maxboundvector = np.zeros(np.size(constraints[0]))
    for i in range(np.shape(constraintmatrix.T)[0]):
        minboundvector[i] = min(constraintmatrix.T[i])
        maxboundvector[i] = max(constraintmatrix.T[i])
    for j in range(np.size(p)):
        p[j] = max(p[j], minboundvector[j])
        p[j] = min(p[j], maxboundvector[j])
    return p


# False: Not in constraint. True: In.
def withinConstraints(constraints, point):
    # if a point's magnitude is larger or smaller than all, out
    magnitudeArr = []
    for c in constraints:
        # if point is greater, 1. If smaller, 0
        temp = 1 if np.linalg.norm(c) - np.linalg.norm(point) < 0 else 0
        magnitudeArr.append(temp)
    if np.sum(magnitudeArr) == 0 or np.sum(magnitudeArr) == np.size(point):
        return False
    return True


# False: Not in constraint. True: In.
def withinConstraintsOrigin(constraints, point):
    # if a point's magnitude is larger or smaller than all, out
    magnitudeArr = []
    for c in constraints:
        # if point is greater, 1. If smaller, 0
        temp = 1 if np.linalg.norm(c) - np.linalg.norm(point) < 0 else 0
        magnitudeArr.append(temp)
    if np.sum(magnitudeArr) == 0:
        return True
    return False


if __name__ == '__main__':
    main()