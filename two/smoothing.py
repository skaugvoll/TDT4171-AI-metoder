'''
Smoothing is the process of computing the distribution over past states k, given evidence up to the present t,
that is P(Xk | e1:t), 0 <= k <= t

This computation is split up into 2 parts.

calculate evidence up to k and the evidence from k + 1 to t.

We can calculate the forward message f1:k by filtering forward from 1 to k.

The backward message bk+1:t can be compyted by a RECURSIve process that runs BACKWARD from t.

'''

import filtering as FILTERING
import numpy as np


def backward(T, o, msg):
    return  T * o * msg # is the equation 15.13 found on page 579


def forwardBackward(evidences, init_msg, T, O):
    b_container = []
    b_msg = np.ones_like(init_msg) # a representation of the backward message, initially all 1s : backwards inital message
    b_container.append(b_msg)
    sv = [] # a vector of smoothed estimates for steps 1,...,t

    # fv_messages # a vector of forward messages for steps 0,...,t
    fv_messages = FILTERING.filtering(init_msg, T, O, evidences)

    for evidence in range(len(evidences)-1, -1, -1):
        res = FILTERING.normalizing(np.multiply(fv_messages[evidence+1], b_msg))
        sv.insert(0,res) # add result to begining of list (this insures coorects indexes , since beginning at the end and traversing to the first element)

        if(evidences[evidence]): # get the wanted sensor (probability for did he bring the umbrella or not - given that it rains or not)
            prob = O[0] # it rains
        else:
            prob = O[1] # it does not raine

        b_msg = backward(T, prob, b_msg)
        b_container.append(b_msg)

    return sv, b_container


if(__name__ == "__main__"):
    # evidence is the same as observation, brought umbrella = true.
    evidence1 = [True, True] # umbrella was used both day 1 and day 2
    evidence2 = [True, True, False, True, True]


    ###### HARDKODET SOM SKAL VÆRE HARDKODET
    init_msg = np.matrix("0.5; 0.5")
    T = np.matrix("0.7 0.3; 0.3 0.7") # dynamic transition model
    O = [ np.matrix("0.9 0; 0 0.2"), np.matrix("0.1 0; 0 0.8") ] # observation / sensor model

    ####### Select task
    # smoothedEstimates, b_messages = forwardBackward(evidence1, init_msg, T, O)
    smoothedEstimates, b_messages = forwardBackward(evidence2, init_msg, T, O)

    print("SV: ", smoothedEstimates)
    print()
    print("B msg: ", b_messages)
