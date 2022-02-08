
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
# from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *
import os
import traceback
import pdb

def start_maze(x,y,z):
    to_return = [[[SPACE_CHAR for k in range(z)] for j in range(y)] for i in range(x)]
    return to_return

def transformToMaze(alien, goals, walls, window,granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            alien (Alien): alien instance
            goals (list): [(x, y, r)] of goals
            walls (list): [(startx, starty, endx, endy)] of walls
            window (tuple): (width, height) of the window

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    offset = [0, 0, 0]
    columns = int(window[0]/granularity + 1)
    rows = int(window[1]/granularity + 1)
    maze = start_maze(columns,rows,3)
    # print(columns,rows,window[0],window[1])

    center = alien.get_centroid()
    shape = get_shape(alien)
    if (shape == 0):
        start = configToIdx([center[0],center[1],'Horizontal'], offset,granularity,alien)
    elif (shape == 1):
        start = configToIdx([center[0],center[1],'Ball'], offset,granularity,alien)
    else:
        start = configToIdx([center[0],center[1],'Vertical'], offset,granularity,alien)

    alien.set_alien_shape('Horizontal')
    for i in range(columns):
        for j in range(rows):
            config = idxToConfig((i,j,0),[0,0,0],granularity,alien)
            alien.set_alien_pos((config[0],config[1]))

            if (is_alien_within_window(alien,window,granularity) == False):
                # alien touches the window with the setting position
                maze[i][j][0] = WALL_CHAR

            elif (does_alien_touch_wall(alien,walls,granularity)):
                # alien touches the wall with the setting position
                maze[i][j][0] = WALL_CHAR

            elif (does_alien_touch_goal(alien,goals) == True):
                # alien touches the goal with the setting position
                maze[i][j][0] = OBJECTIVE_CHAR

    alien.set_alien_shape('Ball')
    for i in range(columns):
        for j in range(rows):
            config = idxToConfig((i,j,1),[0,0,0],granularity,alien)
            alien.set_alien_pos((config[0],config[1]))

            if (is_alien_within_window(alien,window,granularity) == False):
                # alien touches the window with the setting position
                maze[i][j][1] = WALL_CHAR

            elif (does_alien_touch_wall(alien,walls,granularity)):
                # alien touches the wall with the setting position
                maze[i][j][1] = WALL_CHAR

            elif (does_alien_touch_goal(alien,goals) == True):
                # alien touches the goal with the setting position
                maze[i][j][1] = OBJECTIVE_CHAR

    alien.set_alien_shape('Vertical')
    for i in range(columns):
        for j in range(rows):
            config = idxToConfig((i,j,2),[0,0,0],granularity,alien)
            alien.set_alien_pos((config[0],config[1]))

            if (is_alien_within_window(alien,window,granularity) == False):
                # alien touches the window with the setting position
                maze[i][j][2] = WALL_CHAR

            elif (does_alien_touch_wall(alien,walls,granularity)):
                # alien touches the wall with the setting position
                maze[i][j][2] = WALL_CHAR

            elif (does_alien_touch_goal(alien,goals) == True):
                # alien touches the goal with the setting position
                maze[i][j][2] = OBJECTIVE_CHAR

    # col,row,shape start check
    maze[start[0]][start[1]][start[2]] = START_CHAR

    to_return = Maze(maze,alien,granularity,[0,0,0],None)
    return to_return

if __name__ == '__main__':
    import configparser

    def generate_test_mazes(granularities,map_names):
        for granularity in granularities:
            for map_name in map_names:
                try:
                    print('converting map {} with granularity {}'.format(map_name,granularity))
                    configfile = './maps/test_config.txt'
                    config = configparser.ConfigParser()
                    config.read(configfile)
                    lims = eval(config.get(map_name, 'Window'))
                    # print(lis)
                    # Parse config file
                    window = eval(config.get(map_name, 'Window'))
                    centroid = eval(config.get(map_name, 'StartPoint'))
                    widths = eval(config.get(map_name, 'Widths'))
                    alien_shape = 'Ball'
                    lengths = eval(config.get(map_name, 'Lengths'))
                    alien_shapes = ['Horizontal','Ball','Vertical']
                    obstacles = eval(config.get(map_name, 'Obstacles'))
                    boundary = [(0,0,0,lims[1]),(0,0,lims[0],0),(lims[0],0,lims[0],lims[1]),(0,lims[1],lims[0],lims[1])]
                    obstacles.extend(boundary)
                    goals = eval(config.get(map_name, 'Goals'))
                    alien = Alien(centroid,lengths,widths,alien_shapes,alien_shape,window)
                    generated_maze = transformToMaze(alien,goals,obstacles,window,granularity)
                    generated_maze.saveToFile('./mazes/gt_{}_granularity_{}.txt'.format(map_name,granularity))
                except Exception as e:
                    print('Exception at maze {} and granularity {}: {}'.format(map_name,granularity,e))
    def compare_test_mazes_with_gt(granularities,map_names):
        name_dict = {'%':'walls','.':'goals',' ':'free space','P':'start'}
        shape_dict = ['Horizontal','Ball','Vertical']
        for granularity in granularities:
            for map_name in map_names:
                this_maze_file = './mazes/{}_granularity_{}.txt'.format(map_name,granularity)
                gt_maze_file = './mazes/gt_{}_granularity_{}.txt'.format(map_name,granularity)
                if(not os.path.exists(gt_maze_file)):
                    print('no gt available for map {} at granularity {}'.format(map_name,granularity))
                    continue
                gt_maze = Maze([],[],[],filepath = gt_maze_file)
                this_maze = Maze([],[],[],filepath= this_maze_file)
                gt_map = np.array(gt_maze.get_map())
                this_map = np.array(this_maze.get_map())
                difx,dify,difz = np.where(gt_map != this_map)
                if(difx.size != 0):
                    diff_dict = {}
                    for i in ['%','.',' ','P']:
                        for j in ['%','.',' ','P']:
                            diff_dict[i + '_'+ j] = []
                    print('\n\nDifferences in {} at granularity {}:'.format(map_name,granularity))    
                    for i,j,k in zip(difx,dify,difz):
                        gt_token = gt_map[i][j][k] 
                        this_token = this_map[i][j][k]
                        diff_dict[gt_token + '_' + this_token].append(noAlienidxToConfig((i,j,k),granularity,shape_dict))
                    for key in diff_dict.keys():
                        this_list = diff_dict[key]
                        gt_token = key.split('_')[0]
                        your_token = key.split('_')[1]
                        if(len(this_list) != 0):
                            print('Ground Truth {} mistakenly identified as {}: {}'.format(name_dict[gt_token],name_dict[your_token],this_list))
                    print('\n\n')
                else:
                    print('no differences identified  in {} at granularity {}:'.format(map_name,granularity))
    ### change these to speed up your testing early on! 
    granularities = [2,5,8,10]
    map_names = ['Test1','Test2','Test3','Test4','NoSolutionMap']
    generate_test_mazes(granularities,map_names)
    compare_test_mazes_with_gt(granularities,map_names)