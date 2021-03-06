'''

Dataset:
Each line describes an object,
the first seven numbers are the attributes,
the last number is the class of that object.
All attributes as well as the class take values 1 or 2


'''
import math
from Queue import PriorityQueue
from Tree import BadAssTree
from copy import deepcopy
import random as r


def dataGenerator(filename, filetype):
    data = []
    with open("data/" + filename + "." + filetype, 'r') as f:
        for line in f:
            data.append([int(x) for x in line.split("\t")])

    return data

def pluralityValues(examples):
    '''
    Selects the most common output value (target class) among a set of examples, breaking ties randomly
    :param examples: 2D array with examples
    :return: class / target
    '''
    count = {}
    for example in examples:
        key = example[-1]
        if count.has_key(key):
            count[key] += 1
        else:
            count[key] = 1

    mostCount = 0
    target = None
    for key in count.keys():
        cnt = count.get(key)
        if (cnt > mostCount):
            mostCount = cnt
            target = key

    return target

def entropy(probability):
    '''
    Entropy of a boolean random variable that is true with probability :param

    :param probability: p / (p + n) , p = positive examples, n = negative examples for a given attribute
    :return: bits of information
    '''
    if(probability == 0):
        return 0
    elif(probability == 1):
        return  - (probability * math.log(probability, 2))

    return - (probability * math.log(probability, 2) + (1 - probability) * math.log(1 - probability, 2))


def remainder(attribute, examples):
    '''
    Expected entropy remaining after testing attribute A

    Attribute A, has d distinct values, which divides the exaples into subsets E1,...,Ed.
    Each subset Ek has pk positive examples and nk negative examples

    :param attribute:
    :param examples:
    :return:
    '''

    # find D (all possible values for this attribute)
    unique_values = set(ex[attribute] for ex in examples)  # find all different targets / classes

    # find pDi and nDi for each d in D

    # sum for each k (pk + nk / p + n) * B(pk / pk + nk)
    total = float(0)
    totalPos = float(0)
    totalNeg = float(0)

    # find number of positve and negative for this value
    for value in unique_values:
        valuePos = float(0)
        valueNeg = float(0)

        for example in examples:
            if example[-1] == 1:
                totalPos += 1
            elif example[-1] == 2:
                totalNeg += 1

            if(example[attribute] == value):
                if(example[-1] == 1):
                    valuePos += 1
                elif(example[-1] == 2):
                    valueNeg += 1

        total += float(((valuePos + valueNeg) / (totalPos + totalNeg)) * entropy( valuePos / (valuePos + valueNeg)))

    return total




def inforamtionGainBinary(attribute, examples):
    '''
    One type of IMPORTANCE-function.

    Attribute test on A, gives expected reduction in entropy

    Information gain equation;
    Gain(A) = B(p/p+n) - Remainer(A) # B(q) = entropy

    :param: attribute: attribute to test inforamtion gain.
    :param: examples: examples to test the attribute on
    :return: bits, the heigher the better.
    '''

    # to find p, n, pk and nk we need the examples and the attribute to test
    # find number of positive and negative examples
    count = {"p":0, "n":0}
    for example in examples:
        target = example[-1]
        if target == 1:
            count["p"] += 1
        elif target == 2:
            count["n"] += 1

    p = float(count.get("p"))
    n = float(count.get("n"))
    prob = p / (p+n)

    entrop = entropy(prob)
    rem = remainder(attribute, examples)

    # return entrop
    return entrop - rem


def randomWeights(attribute, examples, weights):

    entrop = weights[attribute]

    rem = remainder(attribute, examples)

    # return entrop
    return entrop - rem



def decisionTree(examples, attributes, parent_examples, IG=True, weights=[]):
    '''
    bla bla bla
    :param examples:
    :param attributes:
    :param parent_examples:
    :return: Tree
    '''
    # if examples is empty then return Plurality-Values(parent_examples)
    if(len(examples) < 1):
        # print("Empty examples")
        return pluralityValues(parent_examples)

    # else if all exaples have the same classification, return the classification
    unique_classes = set(ex[-1] for ex in examples) # find all different targets / classes
    if(len(unique_classes) == 1):
        # print("All same target")
        return unique_classes.pop()

    # else if attributes is empty then return Plurality-Values(examples)
    if not attributes:
        return pluralityValues(examples)

    # else
    # print("Decisions is about to be made")
    importanceRank = PriorityQueue() # add a set
    for attribute in attributes:
        if IG:
            importanceRank.put((-1 * inforamtionGainBinary(attribute, examples), attribute)) # -1 * informationGain is to reverse the sort order of the priority queue
        else:
            importanceRank.put((randomWeights(attribute, examples, weights), attribute))

    A = importanceRank.get()[1] # get the one with the heighest score:: .get() gives; (value, attribute)

    tree = BadAssTree(test=A)

    unique_values = set(ex[A] for ex in examples)  # find all different values A can have

    for value in unique_values:
        # reduce the dataset of examples to only have examples with the value for A equal to value
        exs = []
        indexOfAttribute = attributes.index(A)
        reducedAtt = attributes[:indexOfAttribute] + attributes[indexOfAttribute+1 :]
        # reducedWeights = weights[:indexOfAttribute] + attributes[indexOfAttribute + 1:]

        for example in examples:
            if example[A] == value:
                exs.append(example)


        subtreeOrValue = decisionTree(exs, reducedAtt, examples, IG, weights)
        tree.addChild(edge=value, child=subtreeOrValue)

    # Now we should have a decision tree
    return tree

def test(decisionTree):
    data = dataGenerator("test", "txt")

    correctClassifications = 0
    wrongClassifications = 0

    orgTree = decisionTree
    for example in data:
        testTree = deepcopy(orgTree)
        test = testTree.getTest()
        correct = example[-1]

        # try and classify the example
        # DFS LIFO -search
        visited = set() # tests we have taken
        stack = testTree.getChildren() # found

        while stack:
            vertex = stack.pop() # get first children
            # print(type(vertex))
            if type(vertex) == type(1):
                if vertex == correct:
                    correctClassifications += 1
                    break
                else:
                    wrongClassifications += 1
                    break
            if vertex not in visited:
                visited.add(vertex)
                stack.extend(vertex.getChildren())

    print("CC: " + str(correctClassifications))
    print("WC: " + str(wrongClassifications))
    print("Total classifications; " + str(correctClassifications + wrongClassifications))

    percentage = (float(correctClassifications) / float(len(data))) * 100
    print("Percentage correct ; " + str(percentage))



def main():
    data = dataGenerator("training", "txt")
    attributes = [0,1,2,3,4,5,6]

    IG = False # turn on information gain for IMPORTANCE or off for random weights
    weights = []
    if not IG:
        weights = [ r.uniform(0,1) for x in range(len(attributes))]


    learnedTree = decisionTree(data, attributes, [], IG=IG, weights=weights)
    print(learnedTree)
    print("\n")

    test(learnedTree)



if __name__ == "__main__":
    main()
