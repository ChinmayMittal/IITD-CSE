import numpy as np
import time
from collections import defaultdict
import heapq 
import matplotlib.pyplot as plt
import seaborn as sns
from plot import *

class environment:
    def __init__(self, filename, moves,move_move_map, p = 0.8, living_reward = -0.9, hole_reward = -1 , goal_reward = 1):
        self.living_reward = living_reward
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.p = p
        self.move_move_map = move_move_map
        self.move_encoding = {moves[i] : i for i in range(len(moves))}
        self.move_decoding = {i : moves[i] for i in range(len(moves))}
        self.moves = moves
        self.filename = filename
        self.data = self.read_file()
        self.num_rows = len(self.data)
        self.num_cols = len(self.data[0])
        self.state_encoding = {}
        state_id = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.state_encoding[(i,j)] = state_id
                state_id += 1
        self.state_decoding = {v: k for k, v in self.state_encoding.items()}
        self.holes = set([self.state_encoding[(i,j)] for i,j in zip(np.where(self.data == 'H')[0],np.where(self.data == 'H')[1])])
        self.terminal = self.holes.union(set([self.state_encoding[(i,j)] for i,j in zip(np.where(self.data == 'G')[0],np.where(self.data == 'G')[1])]))
        self.goal = ([self.state_encoding[(i,j)] for i,j in zip(np.where(self.data == 'G')[0],np.where(self.data == 'G')[1])][0])
        self.trans_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.reward_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.trans_mat = np.zeros((self.num_rows * self.num_cols, len(moves), self.num_rows * self.num_cols))
        self.reward_mat = np.zeros((self.num_rows * self.num_cols, len(moves), self.num_rows * self.num_cols))
        self.map_left_x = 0
        self.map_right_x = self.num_cols - 1
        self.map_top_y = 0
        self.map_bottom_y = self.num_rows - 1
        self.construct_transition_map()
        self.construct_reward_map()
        self.construct_transition_matrix()
        self.construct_reward_matrix()

    def read_file(self):
        with open(self.filename, 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip().split(',')
        data = np.array(data)
        return data
    
    def check_legal(self, current_pos, move):
        if move == 0: # up
            new_pos = (current_pos[0]-1, current_pos[1])
            if new_pos[0] < self.map_top_y :
                return False, new_pos
        elif move == 1: # down
            new_pos = (current_pos[0]+1, current_pos[1])
            if new_pos[0] > self.map_bottom_y:
                return False, new_pos
        elif move == 2: # left
            new_pos = (current_pos[0], current_pos[1]-1)
            if new_pos[1] < self.map_left_x :
                return False, new_pos
        elif move == 3: # right
            new_pos = (current_pos[0], current_pos[1]+1)
            if new_pos[1] > self.map_right_x :
                return False, new_pos
        return True, new_pos
    
    def construct_transition_map(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.state_encoding[(i,j)] in self.terminal:
                    for move in self.moves:
                        self.trans_map[self.state_encoding[(i,j)]][move][self.state_encoding[(i,j)]] = 1          
                else:     
                    for move in self.moves:
                        legality, updated_pos = self.check_legal((i,j), move)
                        if legality:
                            self.trans_map[self.state_encoding[(i,j)]][move][self.state_encoding[updated_pos]] = self.p
                        else:
                            self.trans_map[self.state_encoding[(i,j)]][move][self.state_encoding[(i,j)]] = self.p
                        for move2 in self.move_move_map[move]:
                            if move != move2:
                                legality2, updated_pos2 = self.check_legal((i,j), move2)
                                if legality2:
                                    self.trans_map[self.state_encoding[(i,j)]][move][self.state_encoding[updated_pos2]] += (1-self.p)/len(self.move_move_map[move])
                                else:
                                    self.trans_map[self.state_encoding[(i,j)]][move][self.state_encoding[(i,j)]] += (1-self.p)/len(self.move_move_map[move])

    def construct_reward_map(self):
        for state1 in self.trans_map.keys():
            for moves in self.trans_map[state1].keys():
                for state2 in self.trans_map[state1][moves].keys():
                    state2_pos = self.state_decoding[state2]
                    if state1 in self.terminal and state1 not in self.holes:
                        self.reward_map[state1][moves][state2] = 0
                    elif self.data[state2_pos[0]][state2_pos[1]] == 'H':
                        self.reward_map[state1][moves][state2] = self.hole_reward
                    elif self.data[state2_pos[0]][state2_pos[1]] == 'G':
                        self.reward_map[state1][moves][state2] = self.goal_reward
                    else:
                        self.reward_map[state1][moves][state2] = self.living_reward

    def construct_transition_matrix(self):
        for state1, state1_id in self.state_encoding.items():
            for move,move_id in self.move_encoding.items():
                for state2, state2_prob in self.trans_map[state1_id][move].items():
                    self.trans_mat[state1_id][move_id][state2] = state2_prob

    def construct_reward_matrix(self):
        for state1 in self.reward_map:
            for move,move_id in self.move_encoding.items():
                for state2, state2_reward in self.reward_map[state1][move_id].items():
                    self.reward_mat[state1][move_id][state2] = state2_reward

    def plot(self,policy, value, move_map,cmap = 'viridis', figsize = None, show_values = True, show_policy = True, save_path = None):
        values_round = value.round(5)
        # change value of the goal states to 0
        value[np.where(self.data.reshape(-1) == 'G')] = 1
        heat_data = None
        if show_policy and show_values:
            heat_data = np.array([str(values_round[i]) + ' \n ' + str(move_map[policy[i]]) for i in range(self.num_cols * self.num_rows)])
        elif show_policy:
            heat_data = np.array([str(move_map[policy[i]]) for i in range(self.num_cols * self.num_rows)])
        elif show_values:
            heat_data = np.array([str(values_round[i]) for i in range(self.num_cols * self.num_rows)])
        heat_data[np.where(self.data.reshape(-1) == 'G')] = 'G'
        heat_data[list(self.holes)] = 'H'
        update = self.num_cols
        if self.num_rows == self.num_cols:
            update = self.num_cols + 1
        if figsize == None:
            size = (2 * update, 2 * self.num_rows)
        else:
            size = figsize
        fig, ax = plt.subplots(figsize=size)         # Sample figsize in inches
        sns.heatmap(value.reshape(self.num_rows,self.num_cols), annot = heat_data.reshape(self.num_rows,self.num_cols).tolist(), fmt = '',ax = ax,cmap = cmap)
        if save_path != None:
            fig.savefig(save_path)
        plt.title('Value Function and Policy')
        plt.show()

class MDPSolver :

    def __init__(self, environment, gamma):
        self.environment = environment
        self.gamma = gamma
        self.num_states = (self.environment.num_rows * self.environment.num_cols)
        self.num_moves = len(self.environment.moves)
        
    
    def evaluate_policy(self, policy ,max_iter=1000, tol=1e-7, print_iters = False):
        T_policy = np.stack([self.environment.trans_mat[s,policy[s]] for s in (self.environment.state_encoding.values())])
        R_policy = np.array([self.environment.reward_mat[s,policy[s]] for s in (self.environment.state_encoding.values())])
        R_exp_policy = np.sum(np.multiply(R_policy, T_policy),axis=1)
        V_policy = np.zeros(self.num_states)
        iter = 0
        while iter < max_iter:
            V_policy_new = R_exp_policy + self.gamma * T_policy.dot(V_policy)
            if np.linalg.norm(V_policy_new - V_policy) < tol:
                break
            V_policy = V_policy_new
            iter += 1
        
        if print_iters:
            print('Number of iterations: ', iter)
            
        return V_policy

    def policy_iteration(self, max_iter=1000, tol=1e-11, print_iters = False):
        print("Policy Iteration")
        policy = np.random.randint(0,self.num_moves,self.num_states, dtype=int)
        iter = 0
        t = time.time()
        V_policy = self.evaluate_policy(policy)
        history = [V_policy[0]]
        while iter < max_iter:
            for s in range(self.num_states):
                if s not in self.environment.terminal:
                    policy[s] = np.argmax([ np.sum([self.environment.trans_mat[s,a,s1] * (self.environment.reward_mat[s,a,s1] + self.gamma*V_policy[s1]) for s1 in (self.environment.trans_map[s][a])]) for a in range(self.num_moves)])
            iter += 1
            V_policy_new = self.evaluate_policy(policy)
            if np.linalg.norm(V_policy_new - V_policy,ord = np.inf) < tol:
                print(np.linalg.norm(V_policy_new - V_policy,ord = np.inf))
                break
            V_policy = V_policy_new
            history.append(V_policy[0])
        if print_iters:
            print(f'Time elapsed: {time.time() - t} seconds')
            print('Number of iterations: ', iter)
            print('Policy Value: ', V_policy)
        return policy, V_policy, history
    
    def value_iteration(self, tol = 1e-4, max_iters = 1000, print_iters = False):
        print("Plain Value Iteration")
        value = np.zeros(self.num_states)
        iter = 0
        delta = -np.inf
        history = [value[0]]
        t = time.time()
        while iter < max_iters:
            val_policy_new = np.zeros(self.num_states)
            for s in range(self.num_states):
                if s not in self.environment.terminal:
                    val_policy_new[s] = np.max([np.sum([self.environment.trans_mat[s,a,s1] * (self.environment.reward_mat[s,a,s1] + self.gamma*value[s1]) for s1 in self.environment.trans_map[s][a]]) for a in range(self.num_moves)])
            delta = np.linalg.norm(value - val_policy_new, ord = np.inf)
            value = val_policy_new
            history.append(value[0])
            if delta < tol * (1-self.gamma) / self.gamma:
                break
            iter += 1
        if print_iters:
            print(f'Time taken: {time.time() - t} seconds')
            print('Number of iterations: ', iter)
            print('Number of state updates = ',iter * self.num_states)
            print('Value: ', value)
            print('Delta: ', delta)
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            policy[s] = np.argmax([np.sum([self.environment.trans_mat[s,a,s1] * (self.environment.reward_mat[s,a,s1] + self.gamma*value[s1]) for s1 in self.environment.trans_map[s][a]]) for a in range(self.num_moves)])
        return policy, value, history

    def row_major_value_iteration(self, tol = 1e-7, max_iters = 10000, print_iters = False):
        print("Row Major Value Iteration")
        value = np.zeros(self.num_states)
        iter = 0
        t = time.time()
        history = [value[0]]
        while iter < max_iters:
            delta = -np.inf
            for i in range(self.environment.num_rows):
                for j in range(self.environment.num_cols):
                    s = self.environment.state_encoding[(i,j)]
                    if s not in self.environment.terminal:
                        prev_val =  value[s]
                        value[s] = np.max([np.sum([self.environment.trans_mat[s,a,s1] * (self.environment.reward_mat[s,a,s1] + self.gamma*value[s1]) for s1 in self.environment.trans_map[s][a]]) for a in range(self.num_moves)])
                        delta = max(delta, np.abs(prev_val - value[s]))
            history.append(value[0])
            if delta < tol :
                break
            iter += 1
        if print_iters:
            print(f'Time taken: {time.time() - t} seconds')
            print('Number of iterations: ', iter)
            print('Number of state updates = ',iter * self.num_states)
            print('Value: ', value)
            print('Delta: ', delta)
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            policy[s] = np.argmax([np.sum([self.environment.trans_mat[s,a,s1] * (self.environment.reward_mat[s,a,s1] + self.gamma*value[s1]) for s1 in self.environment.trans_map[s][a]]) for a in range(self.num_moves)])
        return policy, value, history
        
    def heap_value_iteration(self, tol = 1e-9, max_state_updates = 1000, print_iters = False):
        print("Prioritized Sweep Value Iteration")
        value = np.zeros(self.num_states)
        value[-1] = 8
        iter = 0
        pq = []
        history = [value[0]]
        t = time.time()
        while iter < max_state_updates:
            for s in range(self.num_states):
                if s not in self.environment.terminal:
                    heapq.heappush(pq, (-s, s))
            val_val = value
            t2 = time.time()
            iter_local = 0
            while pq:
                state_to_update = heapq.heappop(pq)
                s = state_to_update[1]
                val_old = value[s]
                value[s] = np.max([np.sum([self.environment.trans_mat[s,a,s1] * (self.environment.reward_mat[s,a,s1] + self.gamma*value[s1]) for s1 in self.environment.trans_map[s][a]]) for a in range(self.num_moves)])
                if (np.abs(val_old - value[s])) > tol  *  (1-self.gamma) / self.gamma:
                    heapq.heappush(pq,(-np.abs(val_old - value[s]),s))
                iter += 1
                iter_local += 1
                if s == 0:
                    history.append(value[0])
            
            print(f'Middle Heap Step Done in : {time.time() - t2} seconds and {iter_local} state updates')
            if np.linalg.norm(val_val - value, ord = np.inf) < tol * (1-self.gamma) / self.gamma:
                break
        print(f'Heap Step Done in : {time.time() - t} seconds')
        print(f'Heap update steps = {iter}')
        print('Value: ', value)
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            policy[s] = np.argmax([np.sum([self.environment.trans_mat[s,a,s1] * (self.environment.reward_mat[s,a,s1] + self.gamma*value[s1]) for s1 in self.environment.trans_map[s][a]]) for a in range(self.num_moves)])
        return policy, value, history
    
def visualise_policy(policy, env, move_map):
    map = []
    for i in range(env.num_rows):
        moves = []
        for j in range(env.num_cols):
                if env.data[i][j] == 'H':
                    moves.append('O')
                elif env.data[i][j] == 'G':
                    moves.append('X')
                else:
                    moves.append(move_map[policy[env.state_encoding[(i,j)]]])
        map.append(moves)
    file = ''
    for i in map:
        file += ' '.join(i) + '\n'
    print(file)
    return file

def animate_policy(policy):
    plcy = policy.reshape((small_environment.num_rows,small_environment.num_cols))
    for i in range(4):
        for j in range(4):
            if  small_environment.data[i][j] == 'G' or small_environment.data[i][j] == 'H':
                plcy[i][j] = -1
    frames = get_traj_frames(small_environment.data, plcy)
    plot(frames,cell_size=15,frame_delay=1)    

def experiments_for_partA(map_path,figsize = None,show_values = True,prefix = 'small', max_state_updates = 1000):
    small_environment = environment(map_path, move_list,move_move_map, living_reward = living_reward, hole_reward = hole_reward, goal_reward = goal_reward)
    small_MDP = MDPSolver(small_environment, 0.9)
    PI_pol, PI_val, PI_hist = small_MDP.policy_iteration(print_iters = True, tol = tol)
    print()
    VI_pol, VI_val, VI_hist = small_MDP.value_iteration(print_iters = True, tol = tol)
    print()
    VIH_pol, VIH_val, VIH_hist = small_MDP.heap_value_iteration(print_iters = True, tol = tol, max_state_updates=max_state_updates)
    print()
    VIR_pol, VIR_val, VIR_hist = small_MDP.row_major_value_iteration(print_iters = True, tol = tol)

    plt.figure(figsize=(16,12))
    plt.plot([i for i in range(len(VI_hist))],VI_hist)
    plt.plot([i for i in range(len(VIR_hist))],VIR_hist)
    plt.plot([i for i in range(len(VIH_hist))],VIH_hist)
    plt.plot([i for i in range(len(PI_hist))],PI_hist)
    plt.legend(['Value Iteration',  'Row Major Value Iteration','Heap Value Iteration', 'Policy Iteration'])
    plt.xlabel('Iterations')
    plt.ylabel('Value of Policy')
    plt.title('Convergence of Different Algorithms')
    plt.savefig(f'plots/{prefix}_convergence_all.png')
    print()
    plt.close()
    small_environment.plot(PI_pol,PI_val,move_map,cmap = 'viridis',show_values = show_values, show_policy = True, save_path = f'plots/{prefix}_map_PI_policy.png', figsize=figsize)
    small_environment.plot(VI_pol,VI_val,move_map,cmap = 'viridis',show_values = show_values, show_policy = True, save_path = f'plots/{prefix}_map_VI_policy.png', figsize=figsize)
    small_environment.plot(VIR_pol,VIR_val,move_map,cmap = 'viridis',show_values = show_values, show_policy = True, save_path = f'plots/{prefix}_map_VIR_policy.png', figsize=figsize)
    small_environment.plot(VIH_pol,VIH_val,move_map,cmap = 'viridis',show_values = show_values, show_policy = True, save_path = f'plots/{prefix}_map_VIH_policy.png', figsize=figsize)    
    
if __name__ == "__main__":
    # Part - A
    living_reward = 0
    hole_reward = 0
    goal_reward = 1
    p = 0.8
    move_list = [0,1,2,3]
    move_move_map = {0:[1,2,3],1:[0,2,3],2:[0,1,3],3:[0,1,2]}
    tol = 1e-9
    move_map = {0: "\u25B2", 1: "\u25BC", 2: "\u25C4", 3: "\u25BA"}
    # experiments_for_partA('small_map.csv',prefix = 'small',show_values=True)
    # experiments_for_partA('large_map.csv',prefix = 'large',show_values=False,figsize = (50,50))
    print('meow')
    p = 1
    living_reward = -0.1
    small_environment = environment('small_map.csv', move_list,move_move_map,p = p, living_reward = living_reward, hole_reward = hole_reward, goal_reward = goal_reward)
    small_MDP = MDPSolver(small_environment, 0.9)
    VI_pol, VI_val, VI_hist = small_MDP.value_iteration(print_iters = True, tol = tol)
    small_environment.plot(VI_pol,VI_val,move_map,cmap = 'viridis',show_values = True, show_policy = True, save_path = f'plots/partb1_petrol_1.png', figsize=None)
    
    living_reward = -0.9
    small_environment = environment('small_map.csv', move_list, move_move_map,p = p, living_reward = living_reward, hole_reward = hole_reward, goal_reward = goal_reward)
    small_MDP = MDPSolver(small_environment, 0.9)
    VI_pol, VI_val, VI_hist = small_MDP.value_iteration(print_iters = True, tol = tol)
    small_environment.plot(VI_pol,VI_val,move_map,cmap = 'viridis',show_values = True, show_policy = True, save_path = f'plots/partb1_petrol_2.png', figsize=None)
    
    living_reward = 0.001
    small_environment = environment('small_map.csv', move_list, move_move_map,p = p, living_reward = living_reward, hole_reward = hole_reward, goal_reward = goal_reward)
    small_MDP = MDPSolver(small_environment, 0.999)
    VI_pol, VI_val, VI_hist = small_MDP.value_iteration(print_iters = True, tol = tol)
    small_environment.plot(VI_pol,VI_val,move_map,cmap = 'viridis',show_values = True, show_policy = True, save_path = f'plots/partb1_electric.png', figsize=None)

    living_reward = 0.01
    small_environment = environment('small_map.csv', move_list, move_move_map,p = p, living_reward = living_reward, hole_reward = hole_reward, goal_reward = goal_reward)
    small_MDP = MDPSolver(small_environment, 0.999)
    VI_pol, VI_val, VI_hist = small_MDP.value_iteration(print_iters = True, tol = tol)
    small_environment.plot(VI_pol,VI_val,move_map,cmap = 'viridis',show_values = True, show_policy = True, save_path = f'plots/partb1_electric_2.png', figsize=None)

    living_reward = 0
    hole_reward = 0
    goal_reward = 1
    p = 1/3
    move_list = [0,1,2,3]
    move_move_map = {0:[2,3],1:[2,3],2:[0,3],3:[0,1]}
    tols = [1e-9]
    move_map = {0: "\u25B2", 1: "\u25BC", 2: "\u25C4", 3: "\u25BA"}
    # 0 - up , 1 - down , 2 - left , 3 - right
    small_environment = environment('small_map.csv', move_list,move_move_map=move_move_map ,p = p,living_reward = living_reward, hole_reward = hole_reward, goal_reward = goal_reward)
    small_MDP = MDPSolver(small_environment, 0.9)
    VI_pol, VI_val, VI_hist = small_MDP.policy_iteration(print_iters = True, tol = tol)
    small_environment.plot(VI_pol,VI_val,move_map,cmap = 'viridis',show_values = True, show_policy = True, save_path = f'plots/partb1_p3_PI.png', figsize=None)

    small_environment = environment('small_map.csv', move_list,move_move_map=move_move_map ,p = p,living_reward = living_reward, hole_reward = hole_reward, goal_reward = goal_reward)
    small_MDP = MDPSolver(small_environment, 0.9)
    VI_pol, VI_val, VI_hist = small_MDP.value_iteration(print_iters = True, tol = tol)
    small_environment.plot(VI_pol,VI_val,move_map,cmap = 'viridis',show_values = True, show_policy = True, save_path = f'plots/partb1_p3.png', figsize=None)