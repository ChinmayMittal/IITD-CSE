from Q1 import *
from collections import defaultdict
from itertools import permutations
from utils import *
from hungarian import hungarianAlgorithm
import time

def calculate_pdf(x, mean, cov):
    return (1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))) * np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))

class DataAssociation:

    def __init__(self, agents, strategy = "factorial_metric_prob"):
        self.num_agent = len(agents)
        self.agents = agents
        self.association_strategy = strategy

    def association_perfect(self, observations):
        return observations,1
    
    def hungarian_association_with_manhattan(self, observations):
        # assuming that the belief as been propagated by the control
        cost_matrix = np.zeros((self.num_agent,self.num_agent))
        for i in range(self.num_agent):
            for j in range(self.num_agent):
                cost_matrix[i,j] = np.linalg.norm(observations[j] - self.agents[i].belief['mu'][:3], ord = 1)
        row_ind = hungarianAlgorithm(cost_matrix, False)
        # print(np.array(observations)[row_ind], sum([cost_matrix[i][j] for i, j in enumerate(row_ind)]))
        return np.array(observations)[row_ind], sum([cost_matrix[i][j] for i, j in enumerate(row_ind)])
    
    def hungarian_association_with_euclidean(self, observations):
        # assuming that the belief as been propagated by the control
        cost_matrix = np.zeros((self.num_agent,self.num_agent))
        for i in range(self.num_agent):
            for j in range(self.num_agent):
                cost_matrix[i,j] = np.linalg.norm(observations[j] - self.agents[i].belief['mu'][:3])
        row_ind = hungarianAlgorithm(cost_matrix, False)
        return np.array(observations)[row_ind], -cost_matrix[row_ind, row_ind].sum()

    def hungarian_association_with_prob(self, observations):
        # assuming that the belief as been propagated by the control
        cost_matrix = np.zeros((self.num_agent,self.num_agent))
        for i in range(self.num_agent):
            for j in range(self.num_agent):
                cost_matrix[i,j] = -np.log(1e-30 + calculate_pdf(observations[j], mean=self.agents[i].belief['mu'][:3], cov=self.agents[i].belief['sigma'][:3,:3]))
        row_ind = hungarianAlgorithm(cost_matrix, False)
        return np.array(observations)[row_ind], -cost_matrix[row_ind, row_ind].sum()
    
    def factorial_association_using_probs(self, observations):
        # assuming that the belief as been propagated by the control
        max_prob = -np.inf
        best_perm = None
        for perm in permutations(observations):
            prob = 0
            for i in range(len(perm)):
                prob += np.log(1e-30 + calculate_pdf(perm[i], mean=self.agents[i].belief['mu'][:3], cov=self.agents[i].belief['sigma'][:3,:3]))
            if prob > max_prob:
                max_prob = prob
                best_perm = perm
        return best_perm, max_prob
    
    def factorial_euclidean_association(self, observations):
        max_score = np.inf
        best_perm = None
        for perm in permutations(observations):
            score = 0
            for i in range(len(perm)):
                score += np.linalg.norm(perm[i] - self.agents[i].belief['mu'][:3])
            if score < max_score:
                max_score = score
                best_perm = perm
        return best_perm, max_score
    
    def factorial_manhattan_association(self, observations):
        max_score = np.inf
        best_perm = None
        for perm in permutations(observations):
            score = 0
            for i in range(len(perm)):
                score += np.linalg.norm(perm[i] - self.agents[i].belief['mu'][:3], ord = 1)
            if score < max_score:
                max_score = score
                best_perm = perm
        return best_perm, max_score
        

    def associate(self,observations):
        if self.association_strategy == "factorial_metric_prob":
            return self.factorial_association_using_probs(observations)
        elif self.association_strategy == "factorial_metric_euclidean":
            return self.factorial_euclidean_association(observations)
        elif self.association_strategy == "factorial_metric_manhattan":
            return self.factorial_manhattan_association(observations)
        elif self.association_strategy == "perfect":
            return self.association_perfect(observations)
        elif self.association_strategy == "hungarian_metric_prob":
            return self.hungarian_association_with_prob(observations)
        elif self.association_strategy == "hungarian_metric_euclidean":
            return self.hungarian_association_with_euclidean(observations)
        elif self.association_strategy == "hungarian_metric_manhattan":
            return self.hungarian_association_with_manhattan(observations)
        else:
            raise ValueError("Association strategy not implemented")
    
def getIncrement(t = 0):
    return -np.array([np.sin(t), np.cos(t), np.sin(t)])

def simulate_multi_agents(agents,agents_filter, association_strategy = "factorial_metric_prob", action_update_first = True, simulation_iterations = 500, print_probs = True):
    data_association = DataAssociation(agents, association_strategy)
    true_state_list = defaultdict(list)
    observed_list = defaultdict(list)
    estimated_list = defaultdict(list)
    belief_covariances = defaultdict(list)
    for i in range(len(agents)):
        true_state_list[i] = [agents[i].state]
        observed_list[i] = [agents[i].sensor_outputs()]
        estimated_list[i] = [agents_filter[i].belief['mu']]
        belief_covariances[i] = [agents_filter[i].belief['sigma']]

    for i in range(1,simulation_iterations+1):
        observations = []
        increment = getIncrement(i)
        for agent in agents:
            # random_increments.append(getIncrement(i) * np.random.uniform(-4,4))
            agent.updateState(increment)

        #updating belief
        for j,agent in enumerate(agents_filter):
            agent.updateBelief(u = increment)

        for agent in agents:
            observations.append(agent.sensor_outputs())
        best_obs, metric_value = data_association.associate(observations)
        if  print_probs:
            print("Best permutation: ", best_obs, " with metric_value: ", metric_value)

        for agent,obs in zip(agents_filter, best_obs):
            agent.updateBelief(z = obs)
        for j,agent in enumerate(agents):
            true_state_list[j].append(agent.state)
            observed_list[j].append(best_obs[j])
            estimated_list[j].append(agents_filter[j].belief['mu'])
            belief_covariances[j].append(agents_filter[j].belief['sigma'])
        
    return true_state_list, observed_list, estimated_list, belief_covariances

def simulator(A,B,C,Qt_list,Rt_list,mu0_list,sigma0_list,methods_list, print_probs = True):
    results = defaultdict(lambda : defaultdict(list))
    for method in (methods_list):
        print('Simulating method: ', method)
        t = time.time()
        agents_to_sim = []
        agents_filter = []
        for i in range(len(mu0_list)):
            agents_to_sim.append(AeroplaneAgent(mu0_list[i],A,B,C,Qt_list[i],Rt_list[i]))
            agents_filter.append(KalmannFilter(agents_to_sim[i], {'mu': mu0_list[i], 'sigma': sigma0_list[i]}))
        true_state_list, observed_list, estimated_list, belief_covariances = simulate_multi_agents(agents_to_sim,agents_filter, association_strategy = method, action_update_first = True, simulation_iterations = 1000, print_probs = print_probs)
        results[method] = {'true_state_dict': true_state_list, 'observed_dict': observed_list, 'estimated_dict': estimated_list, 'belief_covariances': belief_covariances, 'agents' : agents_to_sim, 'agents_filter': agents_filter}
        print('Time taken: ', time.time() - t)
        print('---------------------------------')
        print()
    return results

def error_summary(results):

    avg_dict = defaultdict(lambda : defaultdict(lambda : 0))
    for i in range(len(results[list(results.keys())[0]]['true_state_dict'])):
        print('Agent: ', i)
        print()
        min_method = None
        min = np.inf
        for method in ((results.keys())):
            print('Method: ', method)
            true_state = np.array(results[method]['true_state_dict'][i])
            estimated_state = np.array(results[method]['estimated_dict'][i])
            observed_state = np.array(results[method]['observed_dict'][i])
            print('MSE between true and observed: ', distance_metric(true_state[:,:3], observed_state[:,:3]))
            print('MSE between true and estimated: ', distance_metric(true_state[:,:3], estimated_state[:,:3]))
            print()
            avg_dict[method]['true_observed'] += distance_metric(true_state[:,:3], observed_state[:,:3])
            avg_dict[method]['true_estimated'] += distance_metric(true_state[:,:3], estimated_state[:,:3])
            if distance_metric(true_state[:,:3], estimated_state[:,:3]) < min:
                min = distance_metric(true_state[:,:3], estimated_state[:,:3])
                min_method = method
        print('Best method: ', min_method)
        print('---------------------------------')
    print()
    print('Average MSE across agents\n')
    min_method = None
    min = np.inf
    for method in avg_dict.keys():
        print('Method: ', method)
        print('Average MSE between true and observed: ', avg_dict[method]['true_observed']/4)
        print('Average MSE between true and estimated: ', avg_dict[method]['true_estimated']/4)
        if avg_dict[method]['true_estimated']/4 < min:
            min = avg_dict[method]['true_estimated']/4
            min_method = method
        print('---------------------------------')
    print('Best method: ', min_method)

if __name__ == "__main__":
    mu = np.random.uniform(-20,20,(1,3))
    mu = np.concatenate([mu, np.random.uniform(50,100,(1,3))], axis = 0)
    mu = np.concatenate([mu, np.random.uniform(-200,-150,(1,3))], axis = 0)
    mu = np.concatenate([mu, np.random.uniform(200,250,(1,3))], axis = 0)
    mu = np.concatenate([mu, np.random.uniform(-4,4,(4,3))], axis = 1)
    print("Mu", mu)
    A = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    B = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    C = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
    rx, ry, rz = 1.2,1.2,1.2 
    rvx, rvy, rvz  = 0.01, 0.01, 0.01
    Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    Rt = np.eye(3)*(7**2)
    sigma0 = np.eye(6)*((0.01)**2) 

    Rt_list = [np.eye(3)*(7**2) for i in range(4)]
    Qt_list = [np.diag([rx, ry, rz, rvx, rvy, rvz])**2 for i in range(4)]
    sigma0_list = [np.eye(6)*((0.01)**2)*np.random.uniform(0,1) for i in range(4)]
    methods_list = [ "hungarian_metric_manhattan", "hungarian_metric_euclidean", "hungarian_metric_prob", "factorial_metric_manhattan", "factorial_metric_euclidean", "factorial_metric_prob"]
    results = simulator(A,B,C,Qt_list,Rt_list,mu,sigma0_list,methods_list, print_probs = False)
    for method in methods_list:
        plot_list = []
        for i in range(4):
            plot_list.append([results[method]['true_state_dict'][i], 'True_'+method + "_" + str(i+1)])
            plot_list.append([results[method]['observed_dict'][i], 'Observed_'+method+"_"+str(i+1)])
            plot_list.append([results[method]['estimated_dict'][i], 'Estimated_'+method+"_"+str(i+1)])
        plt = plot_trajectories(plot_list,  0, 1,2)
        plt.write_html('Plots/HTML/Q2_'+method+'.html')
        plt.write_image('Plots/PNG/Q2_'+method+'.png')

    error_summary(results)
