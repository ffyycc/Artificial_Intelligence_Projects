# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from collections import deque
import heapq
import sys

# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
class MST:
    def __init__(self, objectives):
        #length  = len(objectives)
        #print(objectives.index(objectives[1]))
        self.elements = {key: None for key in objectives}
        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i,j) : get_est_dist(i,j)
                for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def backtrack(start,end,dic):
    path = []
    path.append(end)

    while path[-1] != start:
        child = path[-1]
        path.append(dic[child])
    path.reverse()
    #print(path.reverse)
    return path

def get_est_dist(start, end):
    dist = abs(start[0]-end[0]) + abs(start[1]-end[1])
    return dist

def find_near_cell(start_axis,waypoint_axis):
    # compare the distance and get the nearest waypoint
    dist = 2**32
    near_cell = (0,0)
    for i in range(len(waypoint_axis)):
        temp = get_est_dist(start_axis,waypoint_axis[i])

        # change tuple to list, remove reached point, and change back to tuple
        temp_list = list(waypoint_axis)
        temp_list.remove(waypoint_axis[i])
        temp_list = tuple(temp_list)
        temp_MST  = MST(temp_list)
        temp_weight = MST.compute_mst_weight(temp_MST)
        temp_total = temp + temp_weight

        if (dist > temp_total):
            near_cell = waypoint_axis[i]
            dist = temp_total

    to_return = (near_cell,dist)
    return to_return

def find_near_cell_fast(start_axis,waypoint_axis):
    dist = 2**32
    near_cell = (0,0)
    for i in range(len(waypoint_axis)):
        temp = get_est_dist(start_axis,waypoint_axis[i])
        if (dist > temp):
            near_cell = waypoint_axis[i]
            dist = temp

    to_return = (near_cell,dist)
    return to_return


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start_axis = maze.start
    start_cell = maze[maze.start]

    to_return = []
    visit = []
    visit.append(start_axis)
    q = []
    q.append(start_axis)
    parent = {}

    #bfs while loop start
    while(q):
        temp = q.pop(0)
        cell = maze[temp]
        #print(temp)
        if (cell == maze.legend.waypoint):
            to_return = backtrack(maze.start,temp,parent)
            return to_return

        neibor = maze.neighbors(temp[0],temp[1])
        #define neibor elements as child. Find parents through parent[child]
        for child in neibor:
            if (child not in visit):
                visit.append(child)
                q.append(child)
                parent[child] = temp
    return to_return

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start_axis = maze.start
    start_cell = maze[maze.start]
    end_axis = maze.waypoints
    end_axis = end_axis[0]
    end_cell = maze[end_axis]

    to_return = []
    visit = []
    visit.append(start_axis)
    q = []
    est_dist = get_est_dist(start_axis,end_axis)
    start_comb = (est_dist,start_axis,0)
    heapq.heappush(q,start_comb)
    parent = {}

    #astar while loop
    while(q):
        temp = heapq.heappop(q)
        cell = maze[temp[1]]

        if (cell == maze.legend.waypoint):
            to_return = backtrack(maze.start,temp[1],parent)
            return to_return
        
        neibor = maze.neighbors(temp[1][0],temp[1][1])
        for child in neibor:
            if child not in visit:
                # A* distance calculate
                actual_dist = 1 + temp[2]
                est_dist = get_est_dist(child,end_axis)
                total = actual_dist + est_dist
                comb = (total,child,actual_dist)

                # visit append
                visit.append(child)
                # push q to prior q
                heapq.heappush(q,comb)
                # parent dictory append
                parent[child] = temp[1]

    return to_return

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start_axis = maze.start
    waypoint_axis = maze.waypoints

    est_MST = MST(waypoint_axis)
    weight = MST.compute_mst_weight(est_MST)
    
    to_return = []
    visit = []
    visit.append(start_axis)
    q = []

    # near_cell_info define as (axis,distance)
    near_cell_info = find_near_cell(start_axis,waypoint_axis)
    # print(waypoint_axis)
    # print("start is",start_axis,"waypoints are",waypoint_axis)
    total_est = weight + near_cell_info[1]
    start_comb = (total_est,start_axis,0)
    heapq.heappush(q,start_comb)
    parent = {}

    # multiple astar while loop
    while(q):
        temp = heapq.heappop(q)
        if (temp[1] in waypoint_axis):
            # TODO: do sth after finding the first point
            # change tuple to list, remove reached point, and change back to tuple
            waypoint_list = list(waypoint_axis)
            waypoint_list.remove(temp[1])
            waypoint_axis = tuple(waypoint_list)

            # if waypoints tuple are not empty, continue, else return
            if (len(waypoint_axis) != 0):
                # continue if not empty
                # append path
                part_path = backtrack(start_axis,temp[1],parent)
                to_return.extend(part_path[:-1])
                # print(to_return)
                # refresh the visit, parent, queue list
                start_axis = temp[1]
                parent = {}
                q = []

                visit = []
                visit.append(temp[1])

                near_cell_info = find_near_cell(temp[1],waypoint_axis)
                est_MST = MST(waypoint_axis)
                weight = MST.compute_mst_weight(est_MST)
                total_est = weight + near_cell_info[1]
                start_comb = (total_est,temp[1],0)
                heapq.heappush(q,start_comb)

            else:
                to_return.extend(backtrack(start_axis,temp[1],parent))
                # print(to_return)
                print(maze.validate_path(to_return))
                return to_return

        neibor = maze.neighbors(temp[1][0],temp[1][1])
        for child in neibor:
            if child not in visit:
                # A* distance calculate
                actual_dist = 1 + temp[2]
                near_cell_info = find_near_cell(child,waypoint_axis)
                est_dist = get_est_dist(child,near_cell_info[0])
                total = actual_dist + est_dist + weight
                comb = (total,child,actual_dist)

                # visit append
                visit.append(child)
                # push q to prior q
                heapq.heappush(q,comb)
                #parent dictory append
                parent[child] = temp[1]
    return to_return

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start_axis = maze.start
    waypoint_axis = maze.waypoints

    est_MST = MST(waypoint_axis)
    weight = MST.compute_mst_weight(est_MST)
    
    to_return = []
    visit = []
    visit.append(start_axis)
    q = []

    # near_cell_info define as (axis,distance)
    near_cell_info = find_near_cell(start_axis,waypoint_axis)
    total_est = weight + near_cell_info[1]
    start_comb = (total_est,start_axis,0)
    heapq.heappush(q,start_comb)
    parent = {}

    # multiple astar while loop
    while(q):
        temp = heapq.heappop(q)
        if (temp[1] in waypoint_axis):
            # TODO: do sth after finding the first point
            # change tuple to list, remove reached point, and change back to tuple
            waypoint_list = list(waypoint_axis)
            waypoint_list.remove(temp[1])
            waypoint_axis = tuple(waypoint_list)

            # if waypoints tuple are not empty, continue, else return
            if (len(waypoint_axis) != 0):
                # continue if not empty
                # append path
                part_path = backtrack(start_axis,temp[1],parent)
                to_return.extend(part_path[:-1])
                # refresh the visit, parent, queue list
                start_axis = temp[1]
                parent = {}
                q = []

                visit = []
                visit.append(temp[1])

                near_cell_info = find_near_cell_fast(temp[1],waypoint_axis)
                est_MST = MST(waypoint_axis)
                weight = MST.compute_mst_weight(est_MST)
                total_est = weight + near_cell_info[1]
                start_comb = (total_est,temp[1],0)
                heapq.heappush(q,start_comb)

            else:
                to_return.extend(backtrack(start_axis,temp[1],parent))
                return to_return

        neibor = maze.neighbors(temp[1][0],temp[1][1])
        for child in neibor:
            if child not in visit:
                # A* distance calculate
                actual_dist = 1 + temp[2]
                near_cell_info = find_near_cell_fast(child,waypoint_axis)
                est_dist = get_est_dist(child,near_cell_info[0])
                total = actual_dist + est_dist + weight
                comb = (total,child,actual_dist)

                # visit append
                visit.append(child)
                # push q to prior q
                heapq.heappush(q,comb)
                #parent dictory append
                parent[child] = temp[1]
    return to_return
    
            
