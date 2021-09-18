# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def backtrack(start,end,dic):
    path = []
    path.append(end)

    while path[-1] != start:
        child = path[-1]
        path.append(dic[child])
    path.reverse()
    #print(path.reverse)
    return path

def bfs(maze, ispart1=False):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """
    start_axis = maze.getStart()
    goals = maze.getObjectives()
    # print(goals)
    
    to_return = []
    visit = []
    visit.append(start_axis)
    q = []
    q.append(start_axis)
    parent_dict = {}

    #breakpoint()
    #bfs while loop start
    while(q):
        temp = q.pop(0)
        #print(temp)
        if (temp in goals):
            to_return = backtrack(start_axis,temp,parent_dict)
            # print(to_return)
            if (len(to_return) == 0):
                return None
            return to_return

        neibor = maze.getNeighbors(temp[0],temp[1],temp[2],ispart1)

        #define neibor elements as child. Find parents through parent[child]
        for child in neibor:
            if (child not in visit):
                visit.append(child)
                q.append(child)
                parent_dict[child] = temp
    
    return None