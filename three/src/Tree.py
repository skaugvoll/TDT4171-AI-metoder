# -*- coding: utf-8 -*-


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


class BadAssTree():
    def __init__(self, parent=None, test=None):
        self.parent = parent
        self.test = str(test)
        self.edges = []
        self.children = []

    def addChild(self, edge="", child=None):
        self.edges.append(edge)
        self.children.append(child)

    def __str__(self ):
        s = "\tT:"
        s += str(self.test) + "\nE: "
        for e in self.edges:
            s += str(e) + " \t"

        s += "\nC: "

        for child in self.children:
            s += str(child) + "\t"
        s += "\n"

        return s


    def __repr__(self):
        return '<tree representation>'



#
# Et tre har en test - det er inni nodene [ x ]
# Et tre har også en kant til hvert barn, det er attribut-verdi [ x ]
# et tre har også barn, det er et subtree. eller en verdi  [ x ]
# Et tre (barn) har også en foreldre [ x ]
#

#
# Legger til barn!
# Trenger kanten til barnet
# Trenger selve barnet
# Trenger testen (inni noden) "Det nye Attributtet" ?? --> trenger man dette ? --> Nei tror ikke det, da barnet har det selv - hvis node/subtree
#