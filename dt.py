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
    TEST = 0.3

    def __init__(self, data, depth, qmeasure):
        # Complete data set with ground truth.
        self.data = data
        self.maxdepth = depth
        self.qmeasure = qmeasure

        _, self.dim, _, self.classes = self.datasetinfo(data)

        # Split in training and test set.
        #nv = int(DecisionTree.TEST * data.shape[0])
        #self.Xs, self.Xt = np.vsplit(self.data, [nv])

        #logging.info("Training: " + str(self.Xt.shape[0]))
        #logging.info("Test: " + str(self.Xs.shape[0]))
        self.Xt = self.data

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

    def makenode(self, data, feature, splitvalue):
        # Creates node and adds default values for the required attributes.
        node = ete3.TreeNode()

        classdist = self.classdistprob(data)

        # Quality measure for the class distribution.
        qvalue = self.qmeasure(classdist)

        node.add_features(data=data,
                          feature=feature,
                          le=splitvalue,
                          classdist=classdist,
                          qvalue=qvalue)

        return node

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

    def di(self, node, lnode, rnode):
        """
        Improvement of a split.
        """
        lpoints = lnode.data.shape[0]
        rpoints = rnode.data.shape[0]

        lp = lpoints / (lpoints + rpoints)
        rp = rpoints / (lpoints + rpoints)

        return node.qvalue - lp * lnode.qvalue - rp * rnode.qvalue

    def buildtree(self):
        self.root = self.makenode(self.Xt, None, None)

        self.depth = 0
        self._build(self.root)

    def _build(self, node):
        """
        DFS
        """
        if self.depth == self.maxdepth:
            return
        lnode, rnode = self.bestsplit(node)

        if lnode == None or rnode == None:
            return

        self.addtext(lnode)
        self.addtext(rnode)
        node.add_child(rnode, 'r')
        node.add_child(lnode, 'l')

        self.depth += 1

        self._build(lnode)
        self._build(rnode)

    def addtext(self, node):
        txt = str(self.classdist(node.data)) + os.linesep + str(node.qvalue)
        # txt = np.array_str(node.data)

        text = ete3.TextFace(txt)

        # hola.margin_top = 10
        # hola.margin_right = 10
        # hola.margin_left = 10
        # hola.margin_bottom = 10
        # hola.opacity = 0.5  # from 0 to 1
        # hola.inner_border.width = 1  # 1 pixel border
        # hola.inner_border.type = 1  # dashed line
        # hola.border.width = 1
        # text.background.color = "LightGreen"

        node.add_face(text, column=0, position="branch-right")

    def split(self, node, f, split):
        data = node.data
        lnode = self.makenode(data[data[:, f] <= split], f, split)
        rnode = self.makenode(data[data[:, f] > split], f, split)

        return lnode, rnode

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

    def classify(self, x):
        pass
        # TODO

    def accuracy(self):
        pass
        # TODO

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
    gini = 0
    for c in classdist.values():
        gini += pow(c, 2)
    gini = 1 - gini
    return gini


def loadcsv(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    logging.info("Loaded csv file " + filename +
                 " with: " +
                 "data points: " + str(data.shape[0]) + ", " +
                 "features: " + str(data.shape[1] - 1))
    return data


def main():
    logging.basicConfig(level=logging.INFO)

    data = loadcsv("02_homework_dataset.csv")

    dtree = DecisionTree(data, depth=2, qmeasure=gini)

    dtree.show()


if __name__ == '__main__':
    main()
