import networkx as nx
import json
import os.path
import requests
import operator
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import pylab
from collections import OrderedDict
from bisect import bisect_left

def main():
    effectiveDiameter()
    #edgeNovsNodeNo()
    #attackingHighDegrees()

def removeSmallComponents(G):
    for component in list(nx.connected_components(G)):
        if len(component) < 100:
            for node in component:
                G.remove_node(node)
    return G

def effectiveDiameter():
    fileNames = getFileNames()
    effectiveDiameters = []
    for i in range(len(fileNames)):
        data = readFile(fileNames[i][1])
        G = defineGraph(data)
        G = removeSmallComponents(G)
        shortestPathLengths = dict(nx.all_pairs_shortest_path_length(G))
        effDiameter = calculateEffectiveDiameter(shortestPathLengths, G.order())
        effectiveDiameters.append(effDiameter)
        G.clear()


    plt.plot(effectiveDiameters)
    plt.xlabel('Number of days passed since 2019, January 30th')
    plt.ylabel('Effective Diameter')
    pylab.title('Shrinking effective diameter')
    plt.tight_layout()
    plt.show()


def get_val(lst, cn):
    if lst[-1] < cn:
        return "whatever"
    return bisect_left(lst, cn, hi=len(lst) - 1)

def calculateEffectiveDiameter(shortestPathLengths, size):
    nodes = []
    for i in shortestPathLengths:
        nodes.append(i)
    shortestLengths = []
    for i in nodes:
        for j in nodes:
            shortestLengths.append(shortestPathLengths[i][j])
    shortestLengths.sort()
    #g(x) = g(h) + (g(h + 1) − g(h))(x − h).
    three = get_val(shortestLengths, 3) / len(shortestLengths)
    four = get_val(shortestLengths,4)/len(shortestLengths)
    five = get_val(shortestLengths,5)/len(shortestLengths)
    effDiameter = 3+(0.9-three)/(four-three)
    print(effDiameter)
    return effDiameter



def attackingHighDegrees():
    fileNames = getFileNames()
    PP = []  # probability that a random node belongs to the giant component
    p = []  # percolation threshold
    for i in range(len(fileNames)):
        data = readFile(fileNames[i][1])
        G = defineGraph(data)
        #prune smaller components from the graph
        toBeDeleted = []
        for j in range(len(list(nx.connected_components(G)))):
            if len(list(nx.connected_components(G))[j]) < 100:
                print(i, j, len(list(nx.connected_components(G))[j]))
                for k in list(nx.connected_components(G))[j]:
                    toBeDeleted.append(k)
        G.remove_nodes_from(toBeDeleted)
        originalGiantComponentSize = G.order()
        percolationThreshold = 0
        for x in range(1000):
            highestDegrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
            G.remove_node(next(iter(highestDegrees))[0])
            percolationThreshold += 1
            largest_cc = max(nx.connected_components(G), key=len)
            if (len(largest_cc) < originalGiantComponentSize / 100):
                print("Percolation Threshold: ", percolationThreshold, percolationThreshold / originalGiantComponentSize)
                p.append(percolationThreshold/originalGiantComponentSize)
                break

    plt.plot(p)

    plt.xlabel('Number of days passed since 2019, January 30th')
    # Make the y-axis label, ticks and tick labels match the line color.
    plt.ylabel('Percolation Threshold')
    pylab.title('Percolation threshold over the lifetime of LN')

    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')

    plt.tight_layout()
    plt.show()



def edgeNovsNodeNo():
    fileNames = getFileNames()
    edgeNo = []
    nodeNo = []
    for i in range(0,len(fileNames)-2):
        data = readFile(fileNames[i][1])
        G = defineGraph(data)
        edgeNo.append(G.number_of_edges())
        nodeNo.append(G.order())
        print(G.order(),G.number_of_edges())
        G.clear()

    slope, intercept, r_value, p_value, std_err = stats.linregress(nodeNo, edgeNo)
    line = slope * np.asarray(nodeNo) + intercept

    plt.plot(nodeNo, edgeNo, 'o', nodeNo, line)


    plt.xlabel('Number of nodes')
    # Make the y-axis label, ticks and tick labels match the line color.
    plt.ylabel('Number of edges', color='b')
    pylab.title('Linear Fit for LN payment channel growth')


    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')

    plt.tight_layout()
    plt.show()



def averageShortestPathLengths():
    fileNames = getFileNames()
    avgShortestPathLengths=[]
    avgDegree = []
    for i in range(0,len(fileNames)):
        data = readFile(fileNames[i][1])
        G = defineGraph(data)
        #avg degree
        avgDegree.append(G.number_of_edges()/G.order())
        print(i,G.number_of_edges()/G.order())
        #prune smaller components
        toBeDeleted = []
        for j in range(len(list(nx.connected_components(G)))):
            if len(list(nx.connected_components(G))[j]) < 100:
                print(i,j,len(list(nx.connected_components(G))[j]))
                for k in list(nx.connected_components(G))[j]:
                    toBeDeleted.append(k)
        G.remove_nodes_from(toBeDeleted)
        avgShortestPath = nx.algorithms.shortest_paths.generic.average_shortest_path_length(G)
        print(avgShortestPath)
        avgShortestPathLengths.append(avgShortestPath)
        G.clear()

    fig, ax1 = plt.subplots()
    t = np.arange(0, len(fileNames), 1)
    lns1 = ax1.plot(t, avgShortestPathLengths, 'b-', label='Avg Shortest Paths')
    ax1.set_xlabel('Days passed since 2019, January 30th 12 CET')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Average Shortest Path Lengths', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    lns2 = ax2.plot(t, avgDegree, 'r-', label='Average Degree')
    ax2.set_ylabel('Average Out Degree', color='r')
    ax2.tick_params('y', colors='r')

    # added these three lines
    lns = lns1 + lns2# + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')

    fig.tight_layout()
    plt.show()

def defineGraph(data) -> object:
    G = nx.Graph()
    for x in range(len(data)):
        G.add_edge(data[x]['node2_pub'], data[x]['node1_pub'], capacity=data[x]['capacity'])
    return G

def readFile(fileName) -> object:
    with open(fileName) as f:
        data = json.load(f)
    return data['edges']


def getFileNames():
    fileNames = []
    for a, b, c in os.walk('../LNdata/lncaptures/lngraph/2019'):
        if (b == []):
            for i in c:
                fileNames.append(str(a) + '/' + i)
    Files = {}
    for i in fileNames:
        f = open(i, "r")
        timestamp = os.path.basename(i)[:-5]
        Files[timestamp] = i
    sortedFiles = sorted(Files.items(), key=lambda t: t[0])
    return sortedFiles

if __name__ == '__main__':
    main()