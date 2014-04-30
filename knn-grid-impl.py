import sys
import random
import math
from operator import itemgetter

# search point's rectangle area to limit the search space
RECT_SIZE = 0.
RECT_SIZE_FOR_EDGE = 0.

class Vertex:
    """
    Each node in a graph, which is either an intersection
    or an end point.
    """

    def __init__(self, x, y):
        self.id = -1
        self.x  = x
        self.y  = y
    
    def __str__(self):
        return '%i %.6f %.6f' % (self.id, self.x, self.y)
    
    def __repr__(self):
        return self.__str__()
    
    # check equal
    def __eq__(self, other):
        return self.id == other.id
    
    # use this class as a dictionary key
    def __hash__(self):
        return hash(self.__str__())
    
class Edge:
    def __init__(self, id, fr, to):
        self.id = id
        self.fr = fr
        self.to = to
    
    def __str__(self):
        return '%i %s %s' % (self.id, self.fr, self.to)
    
    def __repr__(self):
        return self.__str__()
        
    # check equal
    def __eq__(self, other):
        return self.id == other.id
    
    # use this class as a dictionary key
    def __hash__(self):
        return hash(self.__str__())

class Boundary:
    def __init__(self, node1, node2):
        self.minX = min(node1.x, node2.x)
        self.minY = min(node1.y, node2.y)
        self.maxX = max(node1.x, node2.x)
        self.maxY = max(node1.y, node2.y)
    
    def __str__(self):
        return '\"%.6f %.6f %.6f %.6f\"' % (self.minX, self.maxX, self.minY, self.maxY)
    
    def __repr__(self):
        return self.__str__()
    
    def isInside(self, node):
        return self.minX <= node.x <= self.maxX and self.minY <= node.y <= self.maxY

class GridIndex:
    def __init__(self, boundary, numColumns, numRows):
        self.numColumns = numColumns
        self.numRows = numRows
        
        self.boundary = boundary
        # the size of space
        self.width  = boundary.maxX - boundary.minX
        self.height = boundary.maxY - boundary.minY
        # the size of each grid
        self.gridWidth  = self.width/numColumns
        self.gridHeight = self.height/numRows
    
    def getGridIndex(self, v):
        return int((v.x - self.boundary.minX)/self.gridWidth), \
               int((self.boundary.maxY - v.y)/self.gridHeight)
               
    def findCoveringGrids(self, node1, node2):
        x1, y1 = self.getGridIndex(node1)
        x2, y2 = self.getGridIndex(node2)
    
        maxX = max(x1, x2)
        maxY = max(y1, y2)
        minX = min(x1, x2)
        minY = min(y1, y2)

        if maxX > self.numColumns-1: maxX = self.numColumns-1
        if maxY > self.numRows-1:    maxY = self.numRows-1
        if minX < 0: minX = 0
        if minY < 0: minY = 0

        spanX = maxX - minX + 1
        spanY = maxY - minY + 1
    
        return [(minY+j)*self.numColumns + (minX+i) for j in range(spanY) for i in range(spanX)]

class GridIndexForEdges(GridIndex):
    
    def buildIndex(self, edges):
        self.grids = [[] for i in range(self.numRows*self.numColumns)]
        
        for edge in edges:
            for index in self.findCoveringGrids(edge.fr, edge.to):
                self.grids[index].append(edge)
    
    def knn(self, node, k=5):
        centerX = node.x
        centerY = node.y
        
        node1 = Vertex(centerX-RECT_SIZE_FOR_EDGE, centerY-RECT_SIZE_FOR_EDGE)
        node2 = Vertex(centerX+RECT_SIZE_FOR_EDGE, centerY+RECT_SIZE_FOR_EDGE)
        
        closeEdges = []
        for index in self.findCoveringGrids(node1, node2):
            for edge in self.grids[index]:
                closeEdges.append(edge)
        
        # remove duplicates
        list(set(closeEdges))
        
        newCloseEdges = []
        for edge in closeEdges:
            # compute the distance from the target node to the edge
            dist = euclideanDistance(node, pointProjectionOnLine(node, edge.fr, edge.to))
            newCloseEdges.append((dist, edge))
        
        # sort it!
        newCloseEdges.sort(key=itemgetter(0))
        
        return newCloseEdges[0:k]
        
class GridIndexForNodes(GridIndex):
                       
    def buildIndex(self, nodes):
        self.grids = [[] for i in range(self.numRows*self.numColumns)]
        
        for node in nodes:
            x, y = self.getGridIndex(node)
            
            if x < 0: x = 0
            if x > self.numColumns-1: x = self.numColumns-1
            if y < 0: y = 0
            if y > self.numRows-1: y = self.numRows-1
        
            index = y * self.numColumns + x
        
            self.grids[index].append(node)
    
    def knn(self, node, k=5):
        centerX = node.x
        centerY = node.y
        
        node1 = Vertex(centerX-RECT_SIZE, centerY-RECT_SIZE)
        node2 = Vertex(centerX+RECT_SIZE, centerY+RECT_SIZE)
        
        neighbors = []
        for index in self.findCoveringGrids(node1,node2):
            neighbors += [(euclideanDistance(node, neighbor), neighbor) \
                for neighbor in self.grids[index]]
        
        neighbors.sort(key=itemgetter(0))
        
        return neighbors[0:k]

def euclideanDistance(n1, n2):
    """Calculate the euclidean distance.

    :Parameters:
       t : 1d or 2d array_like object ([M,] P)
          test data

    :Returns:
       p : float or 1d numpy array
          predicted response
    """
    return math.sqrt( \
           math.pow(n1.x - n2.x,2) \
         + math.pow(n1.y - n2.y,2))

def isInLineBounds(C, A, B):
    boundary = Boundary(A, B)
    return boundary.isInside(C)

def pointProjectionOnLine(C, A, B):
    Cx, Cy = C.x, C.y
    Ax, Ay = A.x, A.y
    Bx, By = B.x, B.y
    
    L_2 = (Bx-Ax)**2 + (By-Ay)**2
    
    if L_2 == 0:
        return Vertex(Ax, Ay)
    
    r = ((Ay-Cy)*(Ay-By) - (Ax-Cx)*(Bx-Ax))/ L_2

    newX = Ax + r*(Bx-Ax)
    newY = Ay + r*(By-Ay)
    newPoint = Vertex(newX, newY)

    if not isInLineBounds(newPoint, A, B):
        # If this projection node is not in the line,
        # then compute the minimum of distances start node 
        # and end node. 
        distToStart = euclideanDistance(newPoint, A)
        distToEnd   = euclideanDistance(newPoint, B)

        if distToStart < distToEnd: 
            newPoint = Vertex(Ax,Ay)
        else:
            newPoint = Vertex(Bx,By)
            
    return newPoint

def readNodes(filename):
    nodeFile = open(filename)
    
    print 'Number of nodes: ', nodeFile.readline().strip()    
    nodes = []
    
    minX, minY, maxX, maxY = sys.maxint, sys.maxint, -sys.maxint, -sys.maxint
    
    for line in nodeFile.readlines():
        params = line.strip().split()
        
        node = Vertex(float(params[2]), float(params[3]))
        
        node.id = int(params[0])
        
        minX = min(node.x, minX)
        minY = min(node.y, minY)
        maxX = max(node.x, maxX)
        maxY = max(node.y, maxY)
        
        nodes.append(node)
    
    boundary = Boundary(Vertex(minX,minY), Vertex(maxX, maxY))
    
    return nodes, boundary

def readEdges(filename, nodes):
    edgeFile = open(filename)
    
    # remove the line of number of nodes
    edgeFile.readline()
    
    print 'Number of edges: ', edgeFile.readline().strip()
    edges = []
    
    for line in edgeFile.readlines():
        params = line.strip().split()
        
        edge = Edge(int(params[0]), nodes[int(params[1])], nodes[int(params[2])])
        
        edges.append(edge)
    
    return edges

def intersectTriangle(edge, triangle):
    prev, curr, next = triangle
    
    return intersect(edge.fr, edge.to, prev, curr) \
        or intersect(edge.fr, edge.to, curr, next) \
        or intersect(edge.fr, edge.to, prev, next)

def main():
    """Test with finding the k nearest neighbors from a road network
    """
    global RECT_SIZE, RECT_SIZE_FOR_EDGE
    RECT_SIZE = 0.001
    RECT_SIZE_FOR_EDGE = 0.001
    
    print "K nearest neighbors in spatial"
    
    # read nodes and its boundary
    nodes, boundary = readNodes("vertex.txt")
    edges = readEdges("edges.txt", nodes)
    
    # build index
    nodesGridIndex = GridIndexForNodes(boundary, 100, 100)
    nodesGridIndex.buildIndex(nodes)
    
    edgesGridIndex = GridIndexForEdges(boundary, 100, 100)
    edgesGridIndex.buildIndex(edges)
    
    # randomly generate points to test the result
    spanX = boundary.maxX - boundary.minX
    spanY = boundary.maxY - boundary.minY
    pX = boundary.minX + random.random()*spanX
    pY = boundary.minY + random.random()*spanY
    
    print "x:", pX, "y:", pY
    # search k nearest neighbors
    print "### top 5 closest nodes ###"
    for node in nodesGridIndex.knn(Vertex(pX,pY)):
        print node
    print "### top 5 closest edges ###"    
    for edge in edgesGridIndex.knn(Vertex(pX,pY)):
        print edge

if __name__ == "__main__":
    main()

# #####################################################################
# Copyright 2014, Hengfeng Li.
# 
# ExtractGraph.jar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# ExtractGraph.jar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with ExtractGraph.jar.  If not, see http://www.gnu.org/licenses.
# #####################################################################