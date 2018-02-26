'''
Forwarding = 𝛂 * P(et+1 | Xt+1) * SUM[ (P(Xt+1) | xt) * P(xt | e1:t) ]


forwarding = f1:t+1 = 𝛂 * Ot+1 * T-transposed * f1:t

O = observation model, hva er det vi ser ut ifra den gjemte staten Xt
    - det er flere O,er, 1 Oer for hver type evidence..

= hva sannsynligheten er er for at han har med seg paraply gitt at det regner ideg
  hva er -||=  IKKE  med seg paraply gitt at det regner i dag



f1:t+1 = 𝛂 * Forward(f1:t, et+1)

f1:t = P(Xt | e1:t)   as a " forward message" f1:t
forward message [ sannsynlighet for regn idag , sannsynligheten for ikke regn ]

f1:0 = P(X0)

I denne øvingen er
Xt = { Rain }
Et = { Umbrella }

Rain og Umbrella kan være `true` eller `false`

Forwarding = filtering within inference in temporal model

filtering = forward;
prediction = forward
Smoothing er forward og backward;


FILTERING needs to maintain a current state estimate and updpate it.
Given teh result of filtering up to time t, the agent needs to compute the result for t + 1 from the new evidence et+1

T og O SKAL være hardkodet, da de er en del av modellen vår.
O er diagonal matriser : alt annet en diagonlen er 0

'''

import numpy as np

def normalizing(probs):
    # probs = matrix with probabilities : with, without
    print("Probz: " + str(probs))
    c = np.sum(probs)
    print("C: " + str(c))
    alpha = (1/c)
    print("Alfa: " + str(alpha))
    return probs * alpha

def forwarding(T, o, msg):
    pass


def filtering(init_msg,T, O, evidences):
    messages = [init_msg]
    print("T : " + str(T))
    print("init msg: " + str(init_msg))

    nextMsg = T.T * init_msg
    # print("next msg: " + str(nextMsg))

    # update step BEGIN
    # The update step is multipy the nextMsg with the probability of the evidence for this state, and normalize
    for e in evidences:
        if(e):
            probability = O[0]
        else:
            probability = O[1]

        # proability = O
        # nextMsg = T ?
        msg = probability * nextMsg
        # print("updated msg: " + str(msg))
        nextMsg = normalizing(msg)
        # print("next big R: " + str(nextMsg))
        messages.append(nextMsg)
        nextMsg = T.T * nextMsg

    return messages


def main():
    evidence1 = [True, True] # umbrella was used both day 1 and day 2
    evidence2 = [True, True, False, True, True]


    ###### HARDKODET SOM SKAL VÆRE HARDKODET
    init_msg = np.matrix("0.5;0.5")
    T = np.matrix("0.7 0.3; 0.3 0.7")
    O = [ np.matrix("0.9 0; 0 0.2"), np.matrix("0.1 0; 0 0.2") ]


    messages = filtering(init_msg, T, O, evidence2)
    print("\n\nMessages: \n" + str(messages))



if(__name__ == "__main__"):
    main()
