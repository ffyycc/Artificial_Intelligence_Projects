import numpy as np
import utils
import pdb

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.iteration = 0
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None # state
        self.a = None # action

    def take_action(self,Q_table,N_table,state):
        # priority order RIGHT > LEFT > DOWN > UP
        # if (self.iteration >= 200):
        #     breakpoint()


        if (N_table[state][utils.RIGHT] < self.Ne):
            return utils.RIGHT,True
        elif (N_table[state][utils.LEFT] < self.Ne):
            return utils.LEFT,True
        elif (N_table[state][utils.DOWN] < self.Ne):
            return utils.DOWN,True
        elif (N_table[state][utils.UP] < self.Ne):
            return utils.UP,True
    
        action_list = Q_table[state]
        max_value = max(Q_table[state])
        unique_num = len(np.unique(action_list))
        if (unique_num == 4):
            action_list = list(action_list)
            return action_list.index(max_value),False
        else:
            # priority order RIGHT > LEFT > DOWN > UP
            if (action_list[utils.RIGHT] == max_value):
                return utils.RIGHT,False
            elif (action_list[utils.LEFT] == max_value):
                return utils.LEFT,False
            elif (action_list[utils.DOWN] == max_value):
                return utils.DOWN,False
            else:
                return utils.UP,False

    def update_N_Q_table(self,N_table,Q_table,state_new,award):
        # update N-table and Q-table
        old_state = self.s 
        old_action = self.a
        N_table[old_state][old_action] += 1

        alpha = self.C/(self.C+N_table[old_state][old_action])

        future_value = max(Q_table[state_new])
        Q_table[old_state][old_action] = Q_table[old_state][old_action] + alpha*(award+self.gamma*future_value-Q_table[old_state][old_action])
        return N_table,Q_table

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO: write your function here
        self.iteration += 1
        state = s_prime
        self.train()
        N_table = self.N
        Q_table = self.Q

        if (dead == True):
            award = -1
        elif (points > self.points):
            # breakpoint()
            award = 1
            self.points = points
        else:
            award = -0.1
        
        # cal N and Q and update table
        self.N, self.Q = self.update_N_Q_table(N_table,Q_table,s_prime,award)

        if (dead == True):
            self.reset()
            return utils.DOWN


        new_action,explore = self.take_action(Q_table,N_table,s_prime)
        # create new environment and state
        # environment(snake_head_x,snake_head_y, body, food_x, food_y)
        env_new = environment

        if (new_action == utils.LEFT):
            env_new[0] = environment[0]-utils.GRID_SIZE
            env_new[1] = environment[1]
        elif (new_action == utils.RIGHT):
            env_new[0] = environment[0]+utils.GRID_SIZE
            env_new[1] = environment[1]
        elif (new_action == utils.UP):
            env_new[1] = environment[1]-utils.GRID_SIZE
            env_new[0] = environment[0]
        elif (new_action == utils.DOWN):
            env_new[1] = environment[1]+utils.GRID_SIZE
            env_new[0] = environment[0]

        state_new = self.generate_state(env_new)
        
        # update previous state and action
        self.s = state_new
        self.a = new_action
        return new_action

    # return (food_dir_x, food_dir_y)
    def food_dir(self,snake_head_axis,food_axis):
        if (food_axis[0] < snake_head_axis[0]):
            x = 1
        elif (food_axis[0] > snake_head_axis[0]):
            x = 2
        else:
            x = 0
        
        if (food_axis[1] < snake_head_axis[1]):
            y = 1
        elif (food_axis[1] > snake_head_axis[1]):
            y = 2
        else:
            y = 0
        return (x,y)

    def adjoining_wall(self,snake_head_axis):
        num_block = utils.DISPLAY_SIZE/utils.GRID_SIZE
        snake_head_x = snake_head_axis[0]/utils.GRID_SIZE
        snake_head_y = snake_head_axis[1]/utils.GRID_SIZE
        if (snake_head_x - 1 == 0):
            x = 1
        elif (snake_head_x + 1 == num_block-1):
            x = 2
        else:
            x = 0
        
        if (snake_head_y - 1 == 0):
            y = 1
        elif(snake_head_y + 1 == num_block-1):
            y = 2
        else: 
            y = 0
        return (x,y)

    def adj_body(self,snake_head_axis,snake_body):
        adj_body_top, adj_body_bot,adj_body_left,adj_body_right = 0,0,0,0
        x_head = snake_head_axis[0]
        y_head = snake_head_axis[1]
        grid_size = utils.GRID_SIZE
        if (len(snake_body)!=0):
            for i in range(len(snake_body)):
                element = snake_body[i]
                x_body = element[0]
                y_body = element[1]
                if (y_body == y_head - grid_size):
                    adj_body_top = 1
                if (y_body == y_head + grid_size):
                    adj_body_bot = 1
                if (x_body == x_head - grid_size):
                    adj_body_left = 1
                if (x_body == x_head + grid_size):
                    adj_body_right = 1
        return adj_body_top, adj_body_bot,adj_body_left,adj_body_right


    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        # tuple (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        # environment(snake_head_x,snake_head_y, body, food_x, food_y)
        snake_head_axis = (environment[0],environment[1])
        snake_body = environment[2]
        food_axis = (environment[3],environment[4])

        food_dir_axis = self.food_dir(snake_head_axis,food_axis)
        food_dir_x = food_dir_axis[0]
        food_dir_y = food_dir_axis[1]

        adjoining_wall_axis = self.adjoining_wall(snake_head_axis)
        adj_wall_x = adjoining_wall_axis[0]
        adj_wall_y = adjoining_wall_axis[1]

        adj_body_top, adj_body_bot,adj_body_left,adj_body_right = self.adj_body(snake_head_axis,snake_body)
        to_return = (food_dir_x, food_dir_y, adj_wall_x, adj_wall_y, adj_body_top, adj_body_bot, adj_body_left, adj_body_right)
        return to_return