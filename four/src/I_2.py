# coding=utf-8
'''

'''

import I_1 as task_one
import numpy as np

def lossFunctionDerivation_W1(w):
    a = 2 * (task_one.sigmoidLogisticFunction(w, [1,0]) -1) * task_one.sigmoidLogisticFunction(w, [1,0]) * (1 - task_one.sigmoidLogisticFunction(w, [1,0]))
    b = 0
    c = 2 * (task_one.sigmoidLogisticFunction(w, [1,1]) -1) * task_one.sigmoidLogisticFunction(w, [1,1]) * (1 - task_one.sigmoidLogisticFunction(w, [1,1]))
    return a + b + c

def lossFunctionDerivation_W2(w):
    a = 0
    b = 2 * task_one.sigmoidLogisticFunction(w, [0,1]) * task_one.sigmoidLogisticFunction(w, [0,1]) * (1 - task_one.sigmoidLogisticFunction(w, [0,1]))
    c = 2 * (task_one.sigmoidLogisticFunction(w, [1,1]) - 1) * task_one.sigmoidLogisticFunction(w, [0,1]) * (1 - task_one.sigmoidLogisticFunction(w, [0,1]))
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


def run():
    learningRates = [0.0001, 0.01, 0.1, 1, 10, 100]
    for lr in learningRates:
        task(lr, 50, printInterval=10)
        print("\n")


if __name__ == "__main__":
    # task(0.1, 20)
    run()
