"""
Simple decision tree implementation. This is just an example.
The data for each node is saved within the node for simplicity.
"""

import logging
import numpy as np
import ete3
import os


class DecisionTree:
    # How many points are used for the test set (percentage).
    TEST = 0.2

    def __init__(self, data, depth, qmeasure):
        # Complete data set with ground truth.
        self.data = data
        self.maxdepth = depth
        self.qmeasure = qmeasure

        self.root = None
        self.depth = 0

        _, self.dim, _, self.classes = self.datasetinfo(data)

        # Split in training and test set.
        nv = int(DecisionTree.TEST * data.shape[0])
        self.Xs, self.Xt = np.vsplit(self.data, [nv])

        logging.info("Training: " + str(self.Xt.shape[0]))
        logging.info("Test: " + str(self.Xs.shape[0]))

        self.buildtree()

    @staticmethod
    def datasetinfo(data):
        """
        :return: Data points, dimension, label column, number of classes.
        """
        dpoints, col = data.shape
        col -= 1
        classes = max(data[:, col])

        return dpoints, col, col, int(classes) + 1

    def makenode(self, data):
        # Creates node and adds default values for the required attributes.
        node = ete3.TreeNode()

        classdist = self.classdistprob(data)

        # Quality measure for the class distribution.
        qvalue = self.qmeasure(classdist)

        node.add_features(data=data,
                          classdist=classdist,
                          qvalue=qvalue,
                          feature=None,
                          le=None)

        return node

    @staticmethod
    def setsplit(node, f, splitvalue):
        node.feature = f
        node.le = splitvalue

    def classdist(self, data):
        classdist = dict()

        column = self.dim

        for c in range(self.classes):
            classdist[c] = sum(data[:, column] == c)

        return classdist

    def classdistprob(self, data):
        """
        :return: Dictionary with key = class number and value = percentage.
        """
        dist = self.classdist(data)
        points = data.shape[0]

        for k in dist:
            dist[k] = dist[k] / points

        return dist

    @staticmethod
    def di(node, lnode, rnode):
        """
        Improvement of a split.
        """
        lpoints = lnode.data.shape[0]
        rpoints = rnode.data.shape[0]

        lp = lpoints / (lpoints + rpoints)
        rp = rpoints / (lpoints + rpoints)

        return node.qvalue - lp * lnode.qvalue - rp * rnode.qvalue

    def buildtree(self):
        self.depth = 0
        self.root = self.makenode(self.Xt)
        self._build(self.root)

    def _build(self, node):
        """
        DFS
        """
        # TODO: Depth should checked for each node separately.
        if self.depth == self.maxdepth:
            self.addtext(node)
            return

        lnode, rnode = self.bestsplit(node)
        self.addtext(node)

        if lnode is None or rnode is None:
            return

        node.add_child(rnode, 'r')
        node.add_child(lnode, 'l')

        self.depth += 1

        self._build(lnode)
        self._build(rnode)

    def addtext(self, node):
        txt = "Classes: " + str(self.classdist(node.data)) + os.linesep + \
              "Gini: " + str(node.qvalue)

        txt += os.linesep + "Split: " + str(node.le) + ", " + str(node.feature)

        text = ete3.TextFace(txt)

        node.add_face(text, column=0, position="branch-top")

    def bestsplit(self, node):
        """
        Search the best split over all features and possible splits.

        :return: left node and right node or None if there is no improvement.
        """
        data = node.data

        logging.info("Find best split of " + str(data.shape[0]) + " data points.")

        improvement = 0
        feature = -1
        splitvalue = -1
        found = False

        for f in range(self.dim):
            # Sort data based on one feature
            dsorted = data[np.argsort(data[:, f])]

            # Split point
            for i in range(dsorted.shape[0] - 1):
                d1 = dsorted[i + 1, f]
                d2 = dsorted[i, f]

                if d1 == d2:
                    continue

                split = (d1 + d2) / 2

                lnode, rnode = self.split(node, f, split)

                di = self.di(node, lnode, rnode)
                if di > improvement:
                    improvement = di
                    feature = f
                    splitvalue = split
                    found = True

        if found:
            logging.info("Split feature " + str(feature) + " at " + str(splitvalue))
            return self.split(node, feature, splitvalue)
        else:
            return None, None

    def split(self, node, f, split):
        data = node.data

        self.setsplit(node, f, split)

        lnode = self.makenode(data[data[:, f] <= split])
        rnode = self.makenode(data[data[:, f] > split])

        return lnode, rnode

    def classify(self, x):
        return self._classify(self.root, x)

    def _classify(self, node, x):
        feature = node.feature
        splitvalue = node.le

        value = x[feature]

        if value <= splitvalue:
            child = node.get_leaves_by_name('l')
        else:
            child = node.get_leaves_by_name('r')

        if child[0].is_leaf():
            return max(child[0].classdist, key=child[0].classdist.get)
        else:
            return self._classify(child[0], x)

    def accuracy(self):
        points = self.Xt.shape[0]
        correct = 0

        for i in self.Xt:
            c = self.classify(i)
            if c == i[self.dim]:
                correct += 1

        return correct / points

    def show(self):
        self.root.show()
        self.root.render("tree.png")


def gini(classdist):
    """
    :param classdist:
        dictionary with key = class number and
        value = number of points of that class.
    :return: Gini index
    """
    gval = 0
    for c in classdist.values():
        gval += pow(c, 2)
    gval = 1 - gval
    return gval


def loadcsv(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    logging.info("Loaded csv file " + filename + " with: " +
                 "data points: " + str(data.shape[0]) + ", " +
                 "features: " + str(data.shape[1] - 1))
    return data


def main():
    logging.basicConfig(level=logging.INFO)

    data = loadcsv("02_homework_dataset.csv")

    dtree = DecisionTree(data, depth=2, qmeasure=gini)

    # Classify two samples
    xa = np.array([4.1, -0.1, 2.2])
    xb = np.array([6.1, 0.4, 1.3])

    logging.info("xa = " + str(dtree.classify(xa)))
    logging.info("xb = " + str(dtree.classify(xb)))

    logging.info("Accuracy for test set: " + str(dtree.accuracy()))
    dtree.show()


if __name__ == '__main__':
    main()
