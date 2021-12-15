from typing import List, Tuple

import numpy as np
import networkx as nx
from numpy.core.fromnumeric import size

from sklearn.datasets import make_circles, make_moons, make_blobs
from matplotlib import pyplot as plt

from pyspark import SparkConf, SparkContext, RDD, Broadcast, AccumulatorParam

from helpers.graph import *

class ListAccumulator(AccumulatorParam):
    def zero(self, value: List) -> List:
        return []
    def addInPlace(self, list1: List, list2: List) -> List:
        return list1 + list2

class DictAccumulator(AccumulatorParam):
    def zero(self, value: Dict) -> Dict:
        return {}
    def addInPlace(self, dict1: List, dict2: List) -> Dict:
        dict1.update(dict2)
        return dict1


sparkConf = SparkConf().setAppName('AffinityClustering')
spark = SparkContext(conf=sparkConf)

va = spark.accumulator({}, DictAccumulator())
print(va.value)

def g(x):
    global va
    va += {x[0]: x[1]}

rdd = spark.parallelize([(1, 11), (2, 22), (3, 33)])
rdd.foreach(g)

print(va.value)

spark.stop()