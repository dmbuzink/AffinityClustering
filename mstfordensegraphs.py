from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from pyspark import RDD, SparkConf, SparkContext
from scipy.spatial import distance

def data_reading():
    pass


def create_mst():
    pass

def main():
    parser = ArgumentParser()
    parser.add_argument('--test', help="Used for smaller dataset and testing", action="store_true")
    args = parser.parse_args()

    print("Start generating MST")
    if args.test:
        print("Test argument given")

    start_time = datetime.now()
    checkpoint_time = start_time
    print("Starting time:", start_time)

    conf = SparkConf().setAppName('MST_Algorithm')
    sc = SparkContext(conf=conf)




if __name__ == '__main__':
    main()