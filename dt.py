import logging
import numpy as np


def split(data, feature):
    # Get the best split for all possible splits.

    # Sort data based on one feature
    logging.info("Split feature " + str(feature))
    dsorted = data[np.argsort(data[:, feature])]

    # Split point
    for i in range(dsorted.shape[0] - 1):
        d1 = dsorted[i + 1, feature]
        d2 = dsorted[i, feature]
        split = d1 + (d1 - d2) / 2
        logging.info("Split point: " + str(i) + " : " + str(split))

        # TODO: Calculate how good the split is.


def bestsplit(data):
    # Search the best split over all features and possible splits.

    dpoints, features = data.shape
    features -= 1
    logging.info("Find best split of " + str(dpoints) + " data points.")

    for f in range(features):
        split(data, f)


def buildtree():
    pass


def loadcsv(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    logging.info("Loaded csv file " + filename +
                 " with: " +
                 "data points: " + str(data.shape[0]) +
                 ", features: " + str(data.shape[1] - 1))
    return data


def main():
    logging.basicConfig(level=logging.INFO)
    data = loadcsv("02_homework_dataset.csv")

    bestsplit(data)


if __name__ == '__main__':
    main()
