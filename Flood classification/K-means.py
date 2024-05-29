import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids
        squaredDiff = diff ** 2
        squaredDist = np.sum(squaredDiff, axis=1)
        distance = squaredDist ** 0.5
        clalist.append(distance)
    clalist = np.array(clalist)
    return clalist

def classify(dataSet, centroids, k):
    clalist = calcDis(dataSet, centroids, k)
    minDistIndices = np.argmin(clalist, axis=1)
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()
    newCentroids = newCentroids.values
    changed = newCentroids - centroids

    return changed, newCentroids

def kmeans(dataSet, k):
    centroids = random.sample(dataSet, k)
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())
    cluster = []
    clalist = calcDis(dataSet, centroids, k)
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):
        cluster[j].append(dataSet[i])

    return centroids, cluster

def createDataSet():
    return [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]