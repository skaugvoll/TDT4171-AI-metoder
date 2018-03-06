
class Tree():
    def __init__(self, parent=None, label=None, value=None):
        self.parent = parent
        self.children = []
        self.test = label
        self.value = value

    def addChild(self, label="", value=None):
        self.children.append(Tree(parent=self, label=label, value=value))

    def __repr__(self):
        return str(self.test) + "\n" + str(self.parent) +'\n' + str(self.children) + "\n" + str(self.value)

