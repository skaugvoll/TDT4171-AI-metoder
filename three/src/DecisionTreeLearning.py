'''

Dataset:
Each line describes an object,
the first seven numbers are the attributes,
the last number is the class of that object.
All attributes as well as the class take values 1 or 2


'''

def dataGenerator(filename, filetype):
    data = []
    with open("data/" + filename + "." + filetype, 'r') as f:
        for line in f:
            data.append([int(x) for x in line.split("\t")])

    return data

def pluralityValues(examples):
    pass


def decisionTree(examples):
    # if examples is empty then return Plurality-Values(parent_examples)
    if(len(examples) < 1):
        print("Empty examples")
        parent_examples = []
        pluralityValues(parent_examples)

    # else if all exaples have the same classification, return the classification
    unique_classes = set(ex[-1] for ex in examples) # find all different targets / classes
    if(len(unique_classes) == 1):
        print("All same target")
        return unique_classes.pop()

    # else if attributes is empty then return Plurality-Values(examples)

    # else









def main():
    # data = dataGenerator("empty_data", "txt")
    data = dataGenerator("all_same_target_data", "txt")
    # data = dataGenerator("training", "txt")
    decisionTree(data)


if __name__ == "__main__":
    main()

