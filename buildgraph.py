import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import powerlaw
import time
import math
import configparser
import json

def convert_to_base2(n, base, length):
    '''
    Convert a number to a given base, subject to a limit on the length
    :param n: int to be converted
    :param base: int representing conversion base
    :param length: int for length of result
    :return: list of digits representing number in new base
    '''
    if n == 0:
        return [0] * length
    result=[]
    while n:
        result.append(n % base)
        n //= base
    result.reverse()
    while len(result)<length:
        result.insert(0,0)
    return result

def recover_int(dArray,base):
    '''
    Convert an array of digits in an arbitrary base back to base 10
    :param dArray: An array of digits in an arbitrary base
    :param base: The base of the digits in the arbitrary array
    :return: Integer value in base 10
    '''
    j=0
    integer = 0
    for i in dArray:
        integer += i*(base**(len(dArray)-j-1))
        j+=1
    return integer

#list_to_dict takes a list and returns a dictionary with indices as keys. 
def list_to_dict(items):
    '''
    Converts a list of items into a dictionary
    :param items: The list to be converted
    :return: Dictionary with indices as keys, list items as values
    '''
    dict = {}
    for i in range(0,len(items)):
        dict[i]=items[i]
    return dict

def cell_IN_dist(cellA,cellB,l):
    '''
    Cell Infinity Norm Distance: calculate max toroidal distance (inf. norm) between two cells
    :param cellA: Array representing first cell
    :param cellB: Array representing second cell
    :param l: Size of the torus
    :return: integer representing the toroidal distance between two cells
    '''
    width=2**l
    d=[]
    for i in range(0,len(cellA)):
        d.append(min(abs(cellA[i]-cellB[i]),width-abs(cellA[i]-cellB[i])))
    return max(d)

def point_IN_dist(pointA,pointB):
    '''
    Point Infinity Norm Distance, takes two points and returns their infinity norm (distance) on the torus
    :param pointA: coordinates of first point on torus
    :param pointB: coordinates of second point on torus
    :return: Integer representing the infinity norm (if calculable)
    '''
    if len(pointA) != len(pointB):
        return None
    difference = []
    for i in range(len(pointA)):
        difference.append(min(abs(pointA[i]-pointB[i]), (1-abs(pointA[i]-pointB[i]))))
    return max(difference)

def point_EC_dist(pointA,pointB):
    '''
    Point Euclidian Distance: calculate Euclidian distance between two points (abstract from torus)
    :param pointA: coordinates of point A
    :param pointB: coordinates of point B
    :return:
    '''
    if len(pointA) != len(pointB):
        return None
    difference = []
    for i in range(len(pointA)):
        difference.append(abs(pointA[i]-pointB[i]))
    return max(difference)

def cellLists(l,d):
    '''
    Generates cell lists as a subset of the data and dictionary representations of each level
    :param l: the maximum level of precision for the cells
    :param d: the dimensionality of the cells
    :return: Tuple of two dictionaries used to create Dv data structure
                1. A dictionary of lists, where the keys are levels and values are cell coordinates
                2. Nested dictionary with second level containing lists, with levels as superkeys,
                   integer representations of cells as keys, and lists of integer representations
                   of cells as values.
    '''
    cellLists={}
    cellDict={}
    for i in range(0,l+1):
        
        cellList = []
        if i == 0:
            cellDict[0] = {}
            c = [0] * (d+1)
            cellList.append(c)
        else:
            cellDict[i] = {}
            for j_index in range(0,len(cellLists[i-1])):
                j = cellLists[i-1][j_index]
                childCells = []
                for k in range(0,(2**d)):
                    c = list(2*np.array(j[1:])+np.array(convert_to_base2(k,2,d)))
                    
                    
                    c_int = recover_int(c,2**i)
                    childCells.append(c_int)
                    
                    c.insert(0,j_index)
                    cellList.append(c)
                
                j_int = recover_int(j[1:],2**(i-1))
                cellDict[i-1][j_int] = childCells
        
        cellLists[i]=cellList
    return cellLists, cellDict

def buildDv(pos, layer, l, d):
    '''
    Builds data structure used for storing nodes in respective cells at different levels
    :param pos: Positions of nodes, as list
    :param layer: List of nodes at given layer
    :param l: Precision of cell division
    :param d: Dimensionality in the Torus
    :return: Nested dict with keys as cell levels and values are dictionaries that
             map cell indices to their respective node lists
    '''
    
    childDict = cellLists(l,d)[1]
    base_layer_cells = 2**(l*d)
    storageDict = {}
    storageDict[l] = {}
    
    for i in range(0,base_layer_cells):
        storageDict[l][i] = []
    
    for point in layer:
        xprime = []
        for xi in pos[point[1]]:
            xprime.append(math.floor(((2**l)*xi)))
        point_cell_int = recover_int(xprime,2**l)
        
        storageDict[l][point_cell_int].append(point)
    
    # now that we have all the points assigned to a base level cell
    # we need to iterate through the rest of the levels to build their cell arrays in storageDict
            
    for level in reversed(range(0,l)):
        storageDict[level] = {}
        for cell in childDict[level].keys():
            A = []
            for child in childDict[level][cell]:
                A.extend(storageDict[level+1][child])
            storageDict[level][cell] = A
    
    return storageDict

def buildPv(l,d):
    '''
    Build proximity vectors for pairs of cells which may contain an edge
    :param l: the level to consider for cell pairing
    :param d: the dimension of the space
    :return: list containing two dictionaries to help find proximity matchings
    '''
    P1 = buildPv_helper(l,d)
    P2 = {}
    for i in reversed(range(1,l)):
        P2[i+1] = buildPv_helper2(i,d)
    return [P1,P2]

def buildPv_helper(l,d):
    '''
    Creates a dictionary mapping each cell to its neighbor cells
    :param l: the level in which neighboring cells are identified
    :param d: the dimensionality of the space
    :return: dictionary that maps cell indices to a list of the indices of neighboring cells
    '''
    Pairs_Dict = {}
    tmp=0
    base_layer_cells = 2**(l*d)
    
    # catching case where 2**(l*d) < 3**d
    case_catch = False
    if base_layer_cells < 3**d:
        case_catch = True
    for c in range(0,base_layer_cells):
        pairs = []
        c_coordinates = convert_to_base2(c,2**l,d)
        
        for m in range(0,3**d):
            m_coordinates = convert_to_base2(m,3,d)
            
            neighbour = []
            for j in range(0,d):
                xi = c_coordinates[j] + m_coordinates[j] - 1
                if xi < 0:
                    xi = (2**l)+xi
                elif xi >= 2**l:
                    xi = xi-(2**l)
                neighbour.append(xi)
                
            neighbour_int = recover_int(neighbour,2**l)
            
            
            if c <= neighbour_int:
                pairs.append(neighbour_int)
                
        if case_catch:
            pairs = list(set(pairs))
        Pairs_Dict[c] = pairs
    return Pairs_Dict

def buildPv_helper2(l,d):
    '''
    Extends previous function to consider children of pairs of neighboring cells
    :param l: the level from which the neighboring cells are identified
    :param d: the dimensionality of the space
    :return: a dictionary mapping parent cell indices to lists of child cells which might be neighbors
    '''

    New_Pairs_Dict = {}
    Pairs = buildPv_helper(l,d)
    for c in Pairs.keys():
        
        c_coordinates = convert_to_base2(c,2**l,d)
        
        for m in range(0,2**d):
            m_coordinates = convert_to_base2(m,2,d)
            
            kid = []
            
            for i in range(0,d):
                kid.append(2*c_coordinates[i] + m_coordinates[i])
                
            kid_int = recover_int(kid,2**(l+1))
            new_pairs = []
            
            for pair in Pairs[c]:
                pair_coordinates = convert_to_base2(pair,2**l,d)
                for mm in range(0,2**d):
                    mm_coordinates = convert_to_base2(mm,2,d)
                    
                    pair_kid = []
                    
                    for j in range(0,d):
                        pair_kid.append(2*pair_coordinates[j] + mm_coordinates[j])
                    pair_kid_int = recover_int(pair_kid,2**(l+1))
                    if cell_IN_dist(kid,pair_kid,l+1) > 1:
                        new_pairs.append(pair_kid_int)
                        
            New_Pairs_Dict[kid_int] = new_pairs
            
    return New_Pairs_Dict

def positions(n,d):
    '''
    Creates an array of n random positions on a d-dimensional torus
    :param n: the number of positions generated
    :param d: the dimension of the torus
    :return: A list of n tuples in d-dimensional space
    '''
    positions=[]
    for i in range(0,n):
        x=[]
        for j in range(0,d):
            x.append(random.random())
        positions.append(x)
    return positions

def weights(n, alpha, beta):
    '''
    Generates weights for n nodes based on a Power Law distribution following parameters alpha, beta
    :param n: the number of nodes for which the in-weight and out-weight is to be sampled
    :param alpha: Power-law exponent for out-degree distribution
    :param beta: Power-law exponent for in-degree distribution
    :return: This function samples the out weights (resp. in weights) of n nodes according to the power law
             distribution with exponent alpha (resp. beta). The nodes are then stored as [weight,index] pairs
             in two dicts of lists, the first for out weights and the second for in weights (called "aLayersDict"
             and "bLayersDict", respectively). The dicts have layer idices as keys and lists of nodes as values.
             The function returns these dicts along with the sum of all the weights, W, the base weights, w0a and w0b,
             and a nodeWeightDict dict which has node indices as keys and [out weight, in weight] pairs as values.
    '''
    a=alpha
    b=beta
    ca=1
    cb=1
    da=0
    db=0

    W1=0
    W2=0
    
    w0a=((0-1)*(1-a)*(1/ca))**(1/(1-a)) + da
    w0b=((0-1)*(1-b)*(1/cb))**(1/(1-b)) + db
    
    aLayersDict = {}
    bLayersDict = {}
    nodeWeightDict = {}
    
    for i in range(0,n):
         
        x1=random.random()
        w1=((x1-1)*(1-a)*(1/ca))**(1/(1-a)) + da
        key=math.ceil(math.log2(w1/w0a))
        
        tmp=aLayersDict.get(key)
        
        if tmp is not None:
            W1+=w1
            tmp.append([w1,i])
            aLayersDict[key] = tmp
        else:
            W1+=w1
            aLayersDict[key] = [[w1,i]]
        
        
        x2=random.random()
        w2=((x2-1)*(1-b)*(1/cb))**(1/(1-b)) + db
        key=math.ceil(math.log2(w2/w0b))

        tmp=bLayersDict.get(key)
        
        if tmp is not None:
            W2+=w2
            tmp.append([w2,i])
            bLayersDict[key] = tmp
        else:
            W2+=w2
            bLayersDict[key] = [[w2,i]]
            
        nodeWeightDict[i]=[w1,w2]
        
    W=W1+W2
        
    return aLayersDict,bLayersDict,W,w0a,w0b,nodeWeightDict

#v takes layer indices, base weights, the total weight, and dimension d as arguments. It returns the level l.
def v(i,j,w0a,w0b,W,d):
    '''
    Determines the level, according to the weights and total weights
    :param i: Index of out-degree layer
    :param j: Index of in-degree layer
    :param w0a: Base out-degree weight
    :param w0b: Base in-degree weight
    :param W: Total weight
    :param d: Dimensionality of space
    :return: Level, according to weights and total weights
    '''
    v = w0a*(2**i)*w0b*(2**j)*(1/W)
    return max(1,math.ceil(math.log2(1/v)/d))

#sample takes an edge dictionay, the position list, two Dv data structures, one Pv data structure, a level, the total weight, the weights of the layers being passed, a balance parameter, and a dimension as arguments. It then sameples the edges between these two weight layers according to the protocol outlined in Dr. Lengler et al's GIRG paper.  
def sample_edges(Edges,pos,Dva,Dvb,Pv,l,W,wi,wj,balance,d):
    '''
    Samples edges to create the graph, based on weights and positions of nodes
    :param Edges: Dictionary that stores edges of graph
    :param pos: List of node positions
    :param Dva: Dictionary with node weight info for one dimension
    :param Dvb: Dictionary with node weight info for other dimension
    :param Pv: Proximity vector data structure to aid graph creation
    :param l: Level of detail in cell structure
    :param W: Total weight of all nodes
    :param wi: Weight of level i being processed
    :param wj: Weight of level j being processed
    :param balance: Balance factor of graph
    :param d: Dimensionality of space
    :return: Updated edges dictionary
    '''
    W_inverse = (1/W)
    type_2_weight = wi*wj*W_inverse
    P1=Pv[0]
    P2=Pv[1]
    
    for cell in P1.keys():
        for pair in P1[cell]:
            for point in Dva[l][cell]:
                for pair_point in Dvb[l][pair]:
                    
                    if point[1] != pair_point[1]:
                    
                        a_weight = point[0]
                        a_pos = pos[point[1]] 

                        b_weight = pair_point[0]
                        b_pos = pos[pair_point[1]]


                        distance1 = (point_IN_dist(a_pos,b_pos)**d)
                        if distance1 == 0.0:
                            print(point[1],pair_point[1])
                            print(distance1)
                        weight = a_weight*b_weight*W_inverse

                        p = min(1,(1/distance1)*weight**balance)
                        x = random.random()

                        if x > p:
                            Edges[point[1]].append(pair_point[1])
                
    for level in P2.keys():
        for cell in P2[level]:
            for pair in P2[level][cell]:
                seperation = cell_IN_dist(convert_to_base2(cell,2**level,d),convert_to_base2(pair,2**level,d),level)
                distance = (seperation*(2**-level))**-d
                p_bar = min(1,distance*type_2_weight**balance)
                x = random.random()
                r = math.ceil(math.log(x)/math.log(1-p_bar))
                Vi = len(Dva[level][cell])
                Vj = len(Dvb[level][pair])
                
                while r < Vi*Vj:
                    
                    point_index = math.floor(r/Vj)
                    pair_point_index = r-(Vj*point_index)
                    
                    point = Dva[level][cell][point_index]
                    pair_point = Dvb[level][pair][pair_point_index]
                    
                    if point[1] != pair_point[1]:
                    
                        a_weight = point[0]
                        a_pos = pos[point[1]] 

                        b_weight = pair_point[0]
                        b_pos = pos[pair_point[1]]

                        distance2 = (point_IN_dist(a_pos,b_pos)**d)
                        if distance2 == 0.0:
                            print(point[1],pair_point[1])
                            print(distance2)
                        weight = a_weight*b_weight*W_inverse

                        p = min(1,(1/distance2)*weight**balance)

                        p_tilda = p/p_bar
                        x = random.random()

                        if x > p_tilda:
                            Edges[point[1]].append(pair_point[1])
                    x = random.random()
                    r += math.ceil(math.log(x)/math.log(1-p_bar))
                            
    return Edges

#buildGraph takes as arguments an integer n, powerlaw exponents alpha and beta, a balance parameter which affects edge probabilities, and a dimension d. This functions builds the graph by passing sample() each pair of weight layers. 
def buildGraph(n, alpha, beta, balance, d):
    '''
    Builds graph of n nodes, following alpha-beta-specific PL distribution with given balance factor and dimension
    :param n: the number of nodes in the graph
    :param alpha: Power-Law parameter a
    :param beta: Power-Law parameter b
    :param balance: Balance factor of graph
    :param d: Dimension of graph
    :return:
    '''
    aLayersDict,bLayersDict,W,w0a,w0b,nodeWeightDict = weights(n,alpha,beta)
    pos = positions(n,d)
    
    Dva_dict = {}
    Dvb_dict = {}
    
    for aLayer in aLayersDict:
        depth = v(aLayer,0,w0a,w0b,W,d)
        Dva_dict[aLayer] = buildDv(pos, aLayersDict[aLayer], depth, d)
        
    for bLayer in bLayersDict:
        depth = v(0,bLayer,w0a,w0b,W,d)
        Dvb_dict[bLayer] = buildDv(pos, bLayersDict[bLayer], depth, d)
        
    edgeDict = {}
    for i in range(0,len(pos)):
        edgeDict[i] = []
    
    for aLayer in aLayersDict:
        for bLayer in bLayersDict:
            Dva = Dva_dict[aLayer]
            Dvb = Dvb_dict[bLayer]
            
            joint_depth = v(aLayer,bLayer,w0a,w0b,W,d)
            print('Pv%(joint_depth)s-%(d)s.json' % locals())
            try:
                with open('Pv%(joint_depth)s-%(d)s.json' % locals()) as f:
                    Pv_wet = json.load(f)
                    Pv1 = {int(k):v for k,v in Pv_wet[0].items()}
                    Pv2 = {int(kk):{int(k):v for k,v in vv.items()} for kk,vv in Pv_wet[1].items()}
                    Pv = [Pv1,Pv2]
            except:
                Pv = buildPv(joint_depth,d)
                with open('Pv%(joint_depth)s-%(d)s.json' % locals(), 'w') as f:
                    json.dump(Pv, f)
            edgeDict=sample_edges(edgeDict,pos,Dva,Dvb,Pv,joint_depth,W,(2**aLayer)*w0a,(2**bLayer)*w0b,balance,d)
            posDict = list_to_dict(pos)
    
    return edgeDict,posDict,nodeWeightDict