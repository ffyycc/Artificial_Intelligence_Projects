# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien
import pdb

def dist_two_point(a,b):
    d = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return d

# a,b are two endpoints of lines, and c is a point; 0 middle 1 left 2 right
def left_or_right(a,b,c):
    formula = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    if (formula > 0):
        return 1
    elif (formula < 0):
        return 2
    else: 
        return 0

def dist_line_point(a,b,c):
    p1 = np.array(list(a))
    p2 = np.array(list(b))
    p3 = np.array(list(c))
    distance = np.linalg.norm(np.cross(p2-p1,p3-p1))/np.linalg.norm(p2-p1)
    return distance

# Horizontal - 0  Ball - 1 Vertical - 2
def get_shape(alien):
    if (alien.is_circle()):
        return 1
    head_tail = alien.get_head_and_tail()
    # x same
    if (head_tail[0][0] == head_tail[1][0]):
        return 2
    else:
        return 0

# two line intersection check
def ccw(a,b,c):
    to_return = (c[1]-a[1])*(b[0]-a[0]) >= (b[1]-a[1])*(c[0]-a[0])
    return to_return

def intersect(a,b,c,d):
    if ((ccw(a,c,d) != ccw(b,c,d)) & (ccw(a,b,c) != ccw(a,b,d))):
        return True
    else:
        return False

def circle_touch(center,width,walls,away_dist):
    for wall in walls:
        p1 = (wall[0],wall[1])
        p2 = (wall[2],wall[3])
        # print(p1,p2,center,away_dist,width)
        tangent_dist = dist_line_point(p1,p2,center)

        p1_to_center = dist_two_point(center,p1)
        p2_to_center = dist_two_point(center,p2)
        
        p1_part = np.sqrt(abs(p1_to_center**2-tangent_dist**2)) #修改，不确定
        p2_part = np.sqrt(abs(p2_to_center**2-tangent_dist**2))

        # print(p1_part,p2_part)
        wall_length  = dist_two_point(p1,p2)

        if ((p1_part <= wall_length) & (p2_part <= wall_length)):
            # tangent point in the line
            if (tangent_dist <= (width+away_dist)):
                return True

        else:
            # tangent point out the line
            temp = min(p1_to_center,p2_to_center)
            if (temp <= (width+away_dist)):
                return True
    return False

def horizontal_touch(alien,walls,away_dist):
    width = alien.get_width()
    #  p1-top_left p2-top_right p3-bot_left p4-bot_right
    head_tail = alien.get_head_and_tail()
    if (head_tail[0][0] >= head_tail[1][0]):
        head  = head_tail[1]            
        tail = head_tail[0]
    else:
        head  = head_tail[0]            
        tail = head_tail[1]
    p1 = (head[0],head[1]-width-away_dist)
    p2 = (tail[0],tail[1]-width-away_dist)
    p3 = (head[0],head[1]+width+away_dist)
    p4 = (tail[0],tail[1]+width+away_dist)
    # print(p1,p2,p3,p4,alien.get_width(),head,tail)
    # print(intersect(p1,p2,(16,114.5),(55,110)))

    for wall in walls:
        wall_start = (wall[0],wall[1])
        wall_end = (wall[2],wall[3])
        if (intersect(p1,p2,wall_start,wall_end)):
            return True
        if (intersect(p3,p4,wall_start,wall_end)):
            return True

    left_center = (head[0],head[1])
    right_center = (tail[0],tail[1])
    if ((circle_touch(left_center,width,walls,away_dist)) | (circle_touch(right_center,width,walls,away_dist))):
        return True
    return False

def vertical_touch(alien,walls,away_dist):
    width = alien.get_width()
    #  p1-top_left p2-top_right p3-bot_left p4-bot_right
    head_tail = alien.get_head_and_tail()
    if (head_tail[0][1] >= head_tail[1][1]):
        head  = head_tail[1]            
        tail = head_tail[0]
    else:
        head  = head_tail[0]            
        tail = head_tail[1]
    p1 = (head[0]-width-away_dist,head[1])
    p2 = (head[0]+width+away_dist,head[1])
    p3 = (tail[0]-width-away_dist,tail[1])
    p4 = (tail[0]+width+away_dist,tail[1])
    # print(p1,p2,p3,p4,alien.get_width(),head,tail)
    # print(intersect(p1,p2,(16,114.5),(55,110)))

    for wall in walls:
        wall_start = (wall[0],wall[1])
        wall_end = (wall[2],wall[3])
        if (intersect(p1,p3,wall_start,wall_end)):
            return True
        if (intersect(p2,p4,wall_start,wall_end)):
            return True

    top_center = (head[0],head[1])
    bot_center = (tail[0],tail[1])
    # print(circle_touch(top_center,width,walls,away_dist))
    # print(top_center,bot_center,width,walls,away_dist)
    if ((circle_touch(top_center,width,walls,away_dist)) | (circle_touch(bot_center,width,walls,away_dist))):
        return True
    return False

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    away_dist = granularity/math.sqrt(2)

    # print(left_or_right((0,0),(1,0),(2,2)))
    # print(dist_line_point((0,0),(2,2),(1,2)))
    if (alien.is_circle()):
        # top, bot, left, right
        if (circle_touch(alien.get_centroid(),alien.get_width(),walls,away_dist)):
            return True
        else:
            return False

    # horizontal
    if (get_shape(alien) == 0):
        if (horizontal_touch(alien,walls,away_dist)):
            return True
        else:
            return False
    
    # vertical
    if (get_shape(alien) == 2):
        if (vertical_touch(alien,walls,away_dist)):
            return True
        else:
            return False
    
    return False

def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    center = alien.get_centroid()
    if (alien.is_circle()):
        # top, bot, left, right
        for goal in goals:
            goal_axis = (goal[0],goal[1])
            radius  = goal[2]
            distance  = dist_two_point(goal_axis,center)
            if (distance <= (radius + alien.get_width())):
                return True
        return False

    width = alien.get_width()
    head_tail = alien.get_head_and_tail()   
    head = head_tail[0]
    tail = head_tail[1]
    for goal in goals:
        goal_axis = (goal[0],goal[1])
        radius = goal[2]
        d_head = dist_two_point(goal_axis,head)
        d_tail = dist_two_point(goal_axis,tail)

        if (d_head <= (radius + alien.get_width())):
            return True
        if (d_tail <= (radius + alien.get_width())):
            return True

        # assume top and bot segments are walls
        walls = []
        # horizontal
        if (get_shape(alien) == 0):
            walls = [(head[0], head[1]-width,tail[0],tail[1]-width),
                     (head[0], head[1]+width,tail[0],tail[1]+width)]

        # vertical
        if (get_shape(alien) == 2):
            walls = [(head[0]-width,head[1],tail[0]-width,tail[1]),
                     (head[0]+width,head[1],tail[0]+width,tail[1])]

        if (circle_touch(goal_axis,radius,walls,0)):
            return True
    return False


def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    away_dist = granularity/math.sqrt(2)
    hori_length = window[0]
    verti_length = window[1]
    windows = [(0,0,0,verti_length),(0,0,hori_length,0),(hori_length,0,hori_length,verti_length),(0,verti_length,hori_length,verti_length)]
    # breakpoint()
    if (alien.is_circle()):
        # top, bot, left, right
        if (circle_touch(alien.get_centroid(),alien.get_width(),windows,away_dist)):
            return False
        else:
            return True

    # horizontal
    if (get_shape(alien) == 0):
        if (horizontal_touch(alien,windows,away_dist)):
            return False
        else:
            return True
    
    # vertical
    if (get_shape(alien) == 2):
        if (vertical_touch(alien,windows,away_dist)):
            return False
        else:
            return True
    return True

if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)	
    assert not does_alien_touch_wall(alien_ball, walls, 0), f'does_alien_touch_wall(alien, walls) with alien config {alien_ball.get_config()} returns True, expected: False'
    assert not does_alien_touch_goal(alien_ball, goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_ball.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_ball, window, 0), f'is_alien_within_window(alien, walls) with alien config {alien_ball.get_config()} returns False, expected: True'

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    assert not does_alien_touch_wall(alien_horz, walls, 0), f'does_alien_touch_wall(alien, walls) with alien config {alien_horz.get_config()} returns True, expected: False'
    assert not does_alien_touch_goal(alien_horz, goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_horz.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_horz, window, 0), f'is_alien_within_window(alien, walls) with alien config {alien_horz.get_config()} returns False, expected: True'

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    assert does_alien_touch_wall(alien_vert, walls, 0),f'does_alien_touch_wall(alien, walls) with alien config {alien_vert.get_config()} returns False, expected: True'
    assert not does_alien_touch_goal(alien_vert, goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_vert.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_vert, window, 0), f'is_alien_within_window(alien, walls) with alien config {alien_vert.get_config()} returns False, expected: True'

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()
        assert does_alien_touch_wall(alien, walls, 0) == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {not truths[0]}, expected: {truths[0]}'
        assert does_alien_touch_goal(alien, goals) == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {not truths[1]}, expected: {truths[1]}'
        assert is_alien_within_window(alien, window, 0) == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {not truths[2]}, expected: {truths[2]}'

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")