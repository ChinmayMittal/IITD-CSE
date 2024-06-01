import car_gym
from utils import SMALL_ENV, LARGE_ENV
from utils import plot_rewards
import wandb
import matplotlib.pyplot as plt
import time
import seaborn as sns
#Example USAGE
'''
env can be SMALL_ENV or LARGE_ENV

reset the environment: 
        observation, info = env.reset()
Next state on taking an action: 
        observation, reward, terminated, truncated, info = env.step(action)
Render the current_state:
    rgb_image = env.render

For more information about the gym API please visit https://gymnasium.farama.org/api/env/
New to OpenAI gym ?  Get started here https://gymnasium.farama.org/content/basic_usage/
'''

## Keep your Tabular Q-Learning implementatio here

#TODO
import gymnasium as gym
import numpy as np
#from gym.wrappers.record_video import RecordVideo

# Define the Q-learning function
def q_learning(env,Nn = 100, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
	# Initialize Q-table with zeros
	# env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: episode_number%100 == 0)
	Q = np.zeros((env.observation_space.n, env.action_space.n))

	eps_initial = 1
	eps_final = epsilon
	N = Nn
	model_name = f"tabular_agent_N_{N}_alpha_{alpha}_eps_init_{eps_initial}_eps_final_{eps_final}"
	wandb.init(project = "COL778_A3", name = f"{model_name}")
	wandb.run.log_code(".")
	report_dict = dict();time.sleep(10);rewards_all = []; first_state_value = []
	for episode in range(num_episodes):
		state, _ = env.reset()
		done = False
		r = max((N - episode)/N, 0);eps = (eps_initial - eps_final)*r + eps_final
		while not done:
			# Choose action using epsilon-greedy policy
			if np.random.rand() < eps:
				action = env.action_space.sample()  # Explore
			else:
				action = np.argmax(Q[state, :])  # Exploit

			next_state, reward, done, _, _ = env.step(action)

			# Update Q-value using Q-learning equation
			Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

			state = next_state

		total_reward = evaluate_policy(env, Q);rewards_all.append(total_reward); first_state_value.append(Q[0].max())
		print(f"Episode: {episode}, reward: {total_reward}, epsilon: {eps}")
		report_dict['total_reward'] = total_reward
		report_dict['epsilon'] = eps 
		report_dict['episodes'] = episode 
		wandb.log(report_dict) 

	return Q, rewards_all, first_state_value

# Function to evaluate the learned policy
def evaluate_policy(env, Q):
	state, info = env.reset()
	done = False
	total_reward = 0

	max_step = 500
	step = 0

	while step < max_step:
		action = np.argmax(Q[state, :])
		state, reward, done, _, _ = env.step(action)
		total_reward += reward
		step += 1
		if done:
			break

	return total_reward

def run_policy(env, Q):
	# env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: True)
	state, info = env.reset()
	step = 0
	done = False
	total_reward = 0
	while not done:
		action = np.argmax(Q[state, :])
		state, reward, terminated, truncated, info = env.step(action)
		total_reward += reward
		if terminated:
			done = True
		step += 1
	
	env.close()
	return reward

def plot_actions(action_map,terminal_states,num_rows, num_cols, Q_table,ret_fig = False,cmap = 'viridis', figsize = None, show_values = False, show_policy = True, save_path = None):
        values_round = (Q_table.max(axis = 1).copy()).round(1)
        policy = Q_table.argmax(axis = 1)
        values_round[-1] = 1
        heat_data = None
        if show_policy and show_values:
            heat_data = np.array([str(values_round[i]) + ' \n ' + str(action_map[policy[i]]) for i in range(num_cols * num_rows)])
        elif show_policy:
            heat_data = np.array([str(action_map[policy[i]]) for i in range(num_cols * num_rows)])
        elif show_values:
            heat_data = np.array([str(values_round[i]) for i in range(num_cols * num_rows)])
        values = values_round.reshape(num_rows , num_cols)
        heat_data = heat_data.reshape(num_rows , num_cols)
        for state in terminal_states:
             values[state] = -100
             heat_data[state] = 'T'
        fig, ax = plt.subplots(figsize=(20,12))  
        sns.heatmap(values, annot = heat_data.tolist(), fmt = '',ax = ax,cmap = cmap)
        if save_path != None:
            fig.savefig(save_path)
            plt.close()
        if ret_fig:
            return fig

def run():
	env = RecordVideo(SMALL_ENV, './video',  episode_trigger = lambda episode_number: True)

	state, info = env.reset()

	step = 0
	for _ in range(1000):
		action = env.action_space.sample()  # agent policy that uses the observation and info
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			print("step: ", step)
			observation, info = env.reset()
		step += 1

	env.close()

if __name__ == "__main__":
	# env = RecordVideo(SMALL_ENV, './video',  episode_trigger = lambda episode_number: True)
	env = LARGE_ENV;Nn = 30000;num_episodes = 50000;epsilon = 0.6;t = time.time()
	Q , rewards_all, first_state_value= q_learning(env, Nn = Nn,num_episodes=num_episodes, alpha = 0.4, gamma = 0.99, epsilon=epsilon);print(f'Training time: {time.time() -t}')
	num_rows = env.nrow
	num_cols = env.ncol
	terminal_state_value = []
	terminal_states = [];action_map = { 0: "\u25C4", 1: "\u25BC", 2: "\u25BA",3: "\u25B2"}
	V = [[0 for j in range(num_cols)] for i in range(num_rows)]
	for i in range(num_rows):
		for j in range(num_cols):
			newletter = env.desc[i, j]
			terminated = bytes(newletter) in b"GH"
			s = i * num_cols + j
			V[i][j] = np.max(Q[s, :])

			if terminated:
				terminal_states.append((i, j))
				terminal_state_value.append(V[i][j])
	
	print(terminal_states); plt.plot(first_state_value); plt.xlabel('episode'), plt.ylabel('value'); plt.savefig(f'first_large_{num_episodes}_eps_{int(epsilon*100)}_N_{Nn}.png'); plt.close()
	print(terminal_state_value); plot_rewards(rewards_all, path = f'rewards_large_{num_episodes}_eps_{int(epsilon*100)}_N_{Nn}.png');plt.close()
	plt.imshow(V, cmap='hot', interpolation='nearest')
	plt.colorbar()  # Add color bar to show value range
	plt.title('Heat Map of MDP Values')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')

	plt.savefig(f'heatmap_large_{num_episodes}_eps_{int(epsilon*100)}_N_{Nn}.png');plot_actions(action_map,terminal_states,num_rows, num_cols, Q, save_path=f'action_large_{num_episodes}_eps_{int(epsilon*100)}_N_{Nn}.png')

	# print(Q)
	# run_policy(env, Q)