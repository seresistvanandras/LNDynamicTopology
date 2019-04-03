import json
import os.path
import requests
import operator
import matplotlib.pyplot as plt
import numpy as np

def main():
    capacity()

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