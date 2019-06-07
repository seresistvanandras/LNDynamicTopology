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
import numpy.random
from collections import defaultdict

def main():
    #effectiveDiameter()
    #edgeNovsNodeNo()
    #attackingHighDegrees()
    #routingFees()
    #blockHeadersParser()
    #channelsParser()
    networkOverTime()

def networkOverTime():
    channelsData, opens, closes =channelsParser()
    #1008 or 2016 (1 week or 2 weeks measured in blocks)
    LNCreationBlockHeight = 501337
    lastBlock = 576140
    G = nx.MultiGraph()
    seen_nodes = set()
    numberOfNodes= list()
    numberOfEdges = list()
    avgDegree = list()
    effectDiam = list()
    firstConnectedDegrees = []
    for i in range(1500):
        firstConnectedDegrees.append(0)
    for i in range(LNCreationBlockHeight,lastBlock):
        if i in opens:
            createdChannels = opens[i]
            for j in createdChannels:
                if channelsData[j]['from'] not in seen_nodes:
                    G.add_node(channelsData[j]['from'])
                    seen_nodes.add(channelsData[j]['from'])
                if channelsData[j]['to'] not in seen_nodes:
                    G.add_node(channelsData[j]['to'])
                    seen_nodes.add(channelsData[j]['to'])
                if 550000 < i:
                    if type(G.degree(channelsData[j]['from']))==int and type(G.degree(channelsData[j]['to']))==int:
                        firstConnectedDegrees[max(G.degree(channelsData[j]['from']), G.degree(channelsData[j]['to']))] += 1
                    elif type(G.degree(channelsData[j]['from']))==int:
                        firstConnectedDegrees[max(G.degree(channelsData[j]['from']),len(G.degree(channelsData[j]['to'])))] += 1
                    elif type(G.degree(channelsData[j]['to']))==int:
                        firstConnectedDegrees[max(G.degree(channelsData[j]['to']),len(G.degree(channelsData[j]['from'])))] += 1

                G.add_edge(channelsData[j]['from'], channelsData[j]['to'], capacity=channelsData[j]['amt'])
        if i in closes:
            closedChannels = closes[i]
            for j in closedChannels:
                if G.has_edge(channelsData[j]['from'],channelsData[j]['to']):
                    G.remove_edge(channelsData[j]['from'],channelsData[j]['to'])
                else:
                    G.remove_edge(channelsData[j]['to'],channelsData[j]['from'], capacity=channelsData[j]['amt'])
                if G.degree(channelsData[j]['from']) ==0:
                    G.remove_node(channelsData[j]['from'])
                if G.degree(channelsData[j]['to']) ==0:
                    G.remove_node(channelsData[j]['to'])
        if i % 1008 == 0:
            numberOfNodes.append(nx.number_of_nodes(G))
            numberOfEdges.append(nx.number_of_edges(G))
            avgDegree.append(nx.number_of_edges(G)/nx.number_of_nodes(G))

    print(firstConnectedDegrees)
    plt.hist(firstConnectedDegrees,bins=50, density=True)

    # fig, ax1 = plt.subplots()
    # t = np.arange(0, len(numberOfNodes), 1)
    # lns1 = ax1.plot(t, numberOfNodes, 'b-', label='Nodes')
    # lns2 = ax1.plot(t, numberOfEdges, 'g-', label='Edges')
    # ax1.set_xlabel('Weeks passed since 2017, December 22nd 12 CET')
    # # Make the y-axis label, ticks and tick labels match the line color.
    # ax1.set_ylabel('Number of nodes/edges', color='b')
    # ax1.tick_params('y', colors='b')
    #
    # ax2 = ax1.twinx()
    # lns3 = ax2.plot(t, avgDegree, 'r-', label='Average Degree')
    # ax2.set_ylabel('Average Out Degree', color='r')
    # ax2.tick_params('y', colors='r')
    #
    # # added these three lines
    # lns = lns1 + lns2 + lns3
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='best')
    plt.show()

def channelsParser():
    f = open('ln.tsv','r')
    channelsData = dict()
    opens = dict()
    closes = dict()
    counter = 0
    startingBlock = 1000000
    lastBlock = 0
    for line in f:
        fields = line.split('\t')
        data = {}
        data['from'] = '0x'+fields[1][3:]
        data['to'] = '0x'+fields[2][3:]
        data['tx'] = '0x'+fields[3][3:]
        data['input'] = fields[4]
        data['amt'] = fields[5]
        data['opened'] = fields[6]
        if int(fields[6])< startingBlock:
            startingBlock = int(fields[6])
        if lastBlock< int(fields[6]):
            lastBlock = int(fields[6])
        data['closed'] = fields[7]
        channelsData[fields[0]]=data
        counter+=1
        if int(fields[6]) in opens:
            opens[int(fields[6])].append(fields[0])
        else:
            opens[int(fields[6])]=[fields[0]]

        if fields[7]=='\\N\n':
            continue
        if int(fields[7]) in closes:
            closes[int(fields[7])].append(fields[0])
        else:
            closes[int(fields[7])]=[fields[0]]
    print("First LN channel was created at block height",startingBlock)
    print("We have LN history up to block height", lastBlock)

    return channelsData, opens, closes

def routingFees():
    fileNames = getFileNames()
    for i in range(1):
        feebases = []
        feerates = []
        data = readFile(fileNames[i][1])
        feePair = []
        counter = 0
        G = defineFeeGraph(data)
        for i in G.edges():
            feebases.append(int(G.get_edge_data(i[0],i[1])['feebase']))
            feerates.append(int(G.get_edge_data(i[0],i[1])['feerate']))
            feePair.append((int(G.get_edge_data(i[0],i[1])['feebase']),int(G.get_edge_data(i[0],i[1])['feerate'])))
            if(10000 < feebases[-1]):
                print(max(G.degree(i[0]),G.degree(i[1])))
                counter = counter+1
        print("Total", counter)


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

#graph where capacities are tx fees
def defineFeeGraph(data) -> object:
    G = nx.Graph()
    for x in range(len(data)):
        try:
            G.add_edge(data[x]['node2_pub'], data[x]['node1_pub'], feebase=data[x]['node1_policy']['fee_base_msat'])
            G.add_edge(data[x]['node2_pub'], data[x]['node1_pub'], feerate=data[x]['node1_policy']['fee_rate_milli_msat'])
        except:
            G.add_edge(data[x]['node2_pub'], data[x]['node1_pub'], feebase=data[x]['node2_policy']['fee_base_msat'])
            G.add_edge(data[x]['node2_pub'], data[x]['node1_pub'], feerate=data[x]['node2_policy']['fee_rate_milli_msat'])
    return G

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

def blockHeadersParser():
    counter = 0
    allHeaders = []
    timestamps = []
    g = open('blockheadersTimestamps.txt','w')
    f = open('blockheadersraw.json','r')
    for line in f:
        counter += 1
        timestamps.append(json.loads(line)['timestamp'])
    timestamps.sort()
    for i in timestamps:
        g.write(str(i)+'\n')
    f.close()
    g.close()


if __name__ == '__main__':
    main()