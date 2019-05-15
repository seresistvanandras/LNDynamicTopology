import json
import os.path
import requests
import operator
import matplotlib.pyplot as plt
import numpy as np
import csv

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

def main():
    #capacity()
    nodesPerEdges()

def getEdgeData():
    with open('edges_historic.csv', 'r') as f:
        reader = csv.reader(f)
        edgeData = list(reader)
        return edgeData

def getNodeData():
    with open('nodes_historic.csv', 'r') as f:
        reader = csv.reader(f)
        nodeData = list(reader)
        return nodeData

#https://bitcoinvisuals.com/lightning
def nodesPerEdges():
    edgeData = getEdgeData()
    nodeData = getNodeData()

    edgeNumber = []
    nodeNumber = []
    avgDegree = []
    for i in range(24,len(nodeData),10):
        nodeNumber.append(int(nodeData[i][2]))
        edgeNumber.append(int(edgeData[i][2]))
        avgDegree.append(int(edgeData[i][2])/int(nodeData[i][2]))

    print(int(min(nodeNumber)), int(max(nodeNumber)))
    print(nodeNumber)
    print(edgeNumber)


    logNodes = np.log(nodeNumber)
    logEdges = np.log(edgeNumber)

    coeffs = np.polyfit(logNodes, logEdges, 1)
    print(coeffs) ##[ 1.55634117 -2.45480086]

    logpredicted = np.add(np.multiply(logNodes,coeffs[0]),coeffs[1])
    predicted = np.multiply(np.power(nodeNumber,coeffs[0]),np.power(np.exp(1),coeffs[1]))


    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(logNodes)  # or [p(z) for z in x]
    ybar = np.sum(logEdges) / len(logEdges)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((logEdges - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])

    print("Coefficient of determination: ", ssreg / sstot)

    fig, ax = plt.subplots()
    ax.plot(nodeNumber, edgeNumber, nodeNumber, predicted)
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of edges")

    plt.legend(['Empirical','Predicted'], loc='upper left')
    fig.tight_layout()

    offsetbox = TextArea("2018 Jan", minimumdescent=False)

    ab = AnnotationBbox(offsetbox, (500,1750),
                        xybox=(20, 40),
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    offsetbox2 = TextArea("2019 Apr", minimumdescent=False)

    ab = AnnotationBbox(offsetbox2, (4200, 35000),
                        xybox=(20, 40),
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)
    plt.show()



def capacity():
    fileNames = getFileNames()
    basicData = {}
    timestamps = []
    for i in fileNames:
        f = open(i, "r")
        timestamp = os.path.basename(i)[:-5]
        try:
            data = json.load(f)
            basicData[timestamp] = {
                'num_nodes': data['num_nodes'],
                'num_channels': data['num_channels'],
                'total_network_capacity': data['total_network_capacity'],
                'avg_out_degree':data['avg_out_degree']
            }
        except:
            print("bad file :(", f)
        f.close
    sortedData = sorted(basicData.items(), key=operator.itemgetter(0))
    del basicData
    print("Sorting finished")
    noOfNodes = []
    noOfChannels = []
    networkCapacity = []
    avgDegree = []
    density = []
    counter = 0
    for i in sortedData:
        counter = counter + 1
        if counter%180 == 0:
            noOfNodes.append(i[1]['num_nodes'])
            noOfChannels.append(i[1]['num_channels'])
            networkCapacity.append(float(int(i[1]['total_network_capacity'])/100000000))
            density.append(2*i[1]['num_channels']/(i[1]['num_nodes']*(i[1]['num_nodes']-1)))
            avgDegree.append(i[1]['avg_out_degree'])

    del sortedData
    print(len(noOfNodes))

    fig, ax1 = plt.subplots()
    t = np.arange(0, len(noOfNodes), 1)
    lns1 = ax1.plot(t, noOfNodes, 'b-', label='Nodes')
    lns2 = ax1.plot(t, noOfChannels, 'g-', label='Channels')
    ax1.set_xlabel('Hours passed since 2019, January 30th 12 CET')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Nodes&Channels', color='g')
    ax1.tick_params('y', colors='b')

    # ax2 = ax1.twinx()
    # lns3 = ax2.plot(t, networkCapacity, 'r-', label='Capacity in BTC')
    # ax2.set_ylabel('Total Network Capacity', color='r')
    # ax2.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    lns3 = ax2.plot(t, avgDegree, 'r-', label='Average Degree')
    ax2.set_ylabel('Average Out Degree', color='r')
    ax2.tick_params('y', colors='r')

    # ax2 = ax1.twinx()
    # lns3 = ax2.plot(t, density, 'r-', label='Density')
    # ax2.set_ylabel('Density', color='r')
    # ax2.tick_params('y', colors='r')

    start, end = ax2.get_ylim()
    print(start,end)
    #ax2.yaxis.set_ticks(np.arange(start, end, 10))

    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')

    fig.tight_layout()
    plt.show()


def getFileNames():
    myFileNames = []
    for a, b, c in os.walk('../LNdata/lncaptures/lninfo/2019'):
        if (b == []):
            for i in c:
                myFileNames.append(str(a) + '/' + i)
    return myFileNames

if __name__ == '__main__':
    main()