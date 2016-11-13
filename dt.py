import logging
import numpy as np

def split(data, feature):
    # Sort data based on one feature
    logging.info("Split feature " + str(feature))
    dsorted = data[np.argsort(data[:,feature])]
    print(dsorted)

    # Split point
    for i in range(dsorted.shape[0] - 1):
        d1 = dsorted[i+1,feature]
        d2 = dsorted[i,feature]
        split = d1 + (d1 - d2)/2
        print(split)


def loadcsv(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    logging.info("Loaded csv file " + filename +
                 " with: " +
                 "data points: " + str(data.shape[0]) +
                 " features: " + str(data.shape[1]))
    return data


def main():
    logging.basicConfig(level=logging.INFO)
    data = loadcsv("02_homework_dataset.csv")
    split(data, 0)


if __name__ == '__main__':
    main()
