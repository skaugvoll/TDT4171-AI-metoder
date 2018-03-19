# coding=utf-8

import I_1 as task_one
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def lossFunctionDerivation_W1(w):
    a = 2 * (task_one.sigmoidLogisticFunction(w, [1,0]) -1) * task_one.sigmoidLogisticFunction(w, [1,0]) * (1 - task_one.sigmoidLogisticFunction(w, [1,0]))
    b = 0
    c = 2 * (task_one.sigmoidLogisticFunction(w, [1,1]) -1) * task_one.sigmoidLogisticFunction(w, [1,1]) * (1 - task_one.sigmoidLogisticFunction(w, [1,1]))
    return a + b + c

def lossFunctionDerivation_W2(w):
    a = 0
    b = 2 * task_one.sigmoidLogisticFunction(w, [0,1]) * task_one.sigmoidLogisticFunction(w, [0,1]) * (1 - task_one.sigmoidLogisticFunction(w, [0,1]))
    c = 2 * (task_one.sigmoidLogisticFunction(w, [1,1]) - 1) * task_one.sigmoidLogisticFunction(w, [1,1]) * (1 - task_one.sigmoidLogisticFunction(w, [1,1]))
    return a + b + c


def updateRule(wOld, learningRate = 0.1):
    w1 = wOld.item(0) - ( learningRate * lossFunctionDerivation_W1(wOld).item(0) )
    w2 = wOld.item(1) - ( learningRate * lossFunctionDerivation_W2(wOld).item(0) )
    return np.matrix(str(w1) + "," + str(w2))


def gradientDecent(init_weight, learningRate, iterations):
    w_current = init_weight.copy()
    weights = [w_current]
    losses = [task_one.lossFunction(init_weight)]

    for it in range(iterations):
        w_current = updateRule(w_current, learningRate)
        weights.append(w_current)
        losses.append(task_one.lossFunction(w_current))

    return w_current, weights, losses

def task(lr, its, printInterval=1):
    init_weights = np.matrix("-6, 3")

    foundWeight, foundWeights, losses = gradientDecent(init_weights, learningRate=lr, iterations=its)

    print(" "*20 + "#_"*5 + "LR: " + str(lr) + " Its: " + str(its) + "_#"*5)
    print("{:9}\t{:^40}\t{}".format("Iteration", "Weights", "Loss"))
    print("_"*72)

    for i,w in enumerate(foundWeights):
        if(i % printInterval == 0):
            print("{:-9d}\t{:^40}\t{:f}".format(i, str(w), losses[i].item(0)))


    # return the w that minimiizes Lsimple
    minWeightIndex = losses.index(min(losses))
    return foundWeights[minWeightIndex], losses


def plot(allLosses, legends=[]):
    lineTypes = ["go--", "bo--", "ro--", "r--", "b--", "g--", "ro-","go-","bo-", "r", "b", "g", "b-", "g-", "r-"]
    print(legends)
    for i in allLosses:
        temp = []
        for l in i:
            temp.append(l.item(0))
        # plt.plot(temp, lineTypes[randint(0,len(lineTypes)-1)])
        plt.plot(temp)

    plt.xlabel("Iterations")
    plt.ylabel("Loss value")

    if legends:
        legends = [unichr(414) + " : " + str(x) for x in legends] # U+019E
        plt.legend(legends, bbox_to_anchor=(1, 1), title="Learning rates")

    plt.show()


def run():
    learningRates = [0.0001, 0.01, 0.1, 1, 10, 100]
    minWeights = []
    allLosses =[]
    for lr in learningRates:
        minWeight, allLss = task(lr, 50, printInterval=10)
        minWeights.append(minWeight)
        allLosses.append(allLss)
        print("\n")
    plot(allLosses, legends=learningRates)




if __name__ == "__main__":
    # print(task(0.1, 20))
    run()
