'''

Dataset:
Each line describes an object,
the first seven numbers are the attributes,
the last number is the class of that object.
All attributes as well as the class take values 1 or 2


'''
import math


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
    return - (probability * math.log(probability, 2) + (1 - probability) * math.log(1 - probability, 2))


def remainder(attribute):
    '''
    Expected entropy remaining after testing attribute A

    Attribute A, has d distinct values, which divides the exaples into subsets E1,...,Ed.
    Each subset Ek has pk positive examples and nk negative examples

    :param attribute:
    :return:
    '''

    # find d
    # sum for each k (pk + nk / p + n) * B(pk / pk + nk)




def inforamtionGain(attribute):
    '''
    One type of IMPORTANCE-function.

    Attribute test on A, gives expected reduction in entropy

    :param: attribute: attribute to test inforamtion gain.
    :return: bits, the heigher the better.
    '''




def decisionTree(examples, attributes, parent_examples):
    # if examples is empty then return Plurality-Values(parent_examples)
    if(len(examples) < 1):
        print("Empty examples")
        parent_examples = []
        return pluralityValues(parent_examples)

    # else if all exaples have the same classification, return the classification
    unique_classes = set(ex[-1] for ex in examples) # find all different targets / classes
    if(len(unique_classes) == 1):
        print("All same target")
        return unique_classes.pop()

    # else if attributes is empty then return Plurality-Values(examples)
    if(len(attributes) < 1):
        return pluralityValues(examples)

    # else









def main():
    # data = dataGenerator("empty_data", "txt")
    data = dataGenerator("training", "txt")
    # data = dataGenerator("training", "txt")
    decisionTree(data, [], [])


if __name__ == "__main__":
    main()

