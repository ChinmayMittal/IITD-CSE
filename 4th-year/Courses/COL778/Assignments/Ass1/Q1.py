import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
import time

class KalmannFilter :
     
    def __init__(self, agent, init_belief):
        agent.belief = init_belief
        self.agent = agent
        self.belief = init_belief

    def predictionUpdate(self, u):
        # Prediction update of the state of the aeroplane
        # u is the control input

            self.belief['mu'] = np.matmul(self.agent.At, self.belief['mu']) + np.matmul(self.agent.Bt, u) 
            self.belief['sigma'] = np.matmul(self.agent.At, np.matmul(self.belief['sigma'], self.agent.At.T)) + self.agent.Qt
        
    def measurementUpdate(self, z):
    
        # Measurement update of the state of the aeroplane
        # z is the measurement
    
        Kt = np.matmul(np.matmul(self.belief['sigma'], self.agent.Ct.T), np.linalg.inv(np.matmul(self.agent.Ct, np.matmul(self.belief['sigma'], self.agent.Ct.T)) + self.agent.Rt))
        self.belief['mu'] = self.belief['mu'] + np.matmul(Kt, z - np.matmul(self.agent.Ct, self.belief['mu']))
        self.belief['sigma'] = np.matmul(np.identity(self.belief['sigma'].shape[0]) - np.matmul(Kt, self.agent.Ct), self.belief['sigma'])
    
    def updateBelief(self, u = None, z = None):
                
        # Update the state of the aeroplane at time t
        # u is the control input
        # z is the measurement
                
        if u is not None:
            self.predictionUpdate(u)
        if z is not None:
            self.measurementUpdate(z)
        if u is None and z is None:
            print("No input given")
        self.agent.belief = self.belief

class AeroplaneAgent :

    # Kalmann Filter for estimating the position of the aeroplane

    def __init__(self, init_pos, At, Bt, Ct, Qt, Rt):
        self.belief = None
        self.state = init_pos
        self.true_state = init_pos
        self.At = At
        self.Bt = Bt
        self.Ct = Ct
        self.Qt = Qt # covariance matrix for the noise in control
        self.Rt = Rt # covariance matrix for the noise in measurement

    def fetchState(self):

        # Fetch the current state of the aeroplane at time t (returns belief distribution)

        return self.state
    
    def updateState(self, u):
         
        self.state = np.matmul(self.At, self.state) + np.matmul(self.Bt, u) + np.random.multivariate_normal(np.zeros(self.Qt.shape[0]),self.Qt)

    def sensor_outputs(self):
        return np.matmul(self.Ct, self.state) + np.random.multivariate_normal(np.zeros(self.Rt.shape[0]),self.Rt)

                
def getIncrement(t = 0):
     return np.array([np.sin(t), np.cos(t), np.sin(t)])

def simulate(Agent, mu0, sigma0, simulation_iterations = 500, leave_obs_cond = lambda i : i < -1):
    Agent_estimator = KalmannFilter(Agent, {'mu': mu0, 'sigma': sigma0})
    true_state_list = [Agent.state]
    observed_list = [Agent.sensor_outputs()]
    estimated_list = [mu0]
    belief_covariances = [sigma0]
    for i in range(1,simulation_iterations+1):
        Agent.updateState(u = getIncrement(i))

        if leave_obs_cond(i):
            Agent_estimator.updateBelief(u = getIncrement(i))
        else:
            Agent_estimator.updateBelief(u = getIncrement(i), z = Agent.sensor_outputs())
        Agent.belief = Agent_estimator.belief
        true_state_list.append(Agent.state)
        observed_list.append(Agent.sensor_outputs())
        estimated_list.append(Agent_estimator.belief['mu'])
        belief_covariances.append(Agent_estimator.belief['sigma'])
    return true_state_list, observed_list, estimated_list, belief_covariances

if __name__ == '__main__':
    # os.mkdir('Plots')
    # os.mkdir('Plots/HTML')
    # os.mkdir('Plots/PNG')
    # os.mkdir('viva')
    A = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    B = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    C = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    rx, ry, rz = 1.2,1.2,1.2 
    rvx, rvy, rvz  = 0.01,0.01,0.01
    Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    Rt = np.eye(6)*(7**2)
    mu0 = np.array([0,0,0,0,0,0])
    mu1 = np.array([0,0,0,0.5,0,0])
    sigma0 = np.eye(6)*((0.01)**2)

    Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu1, sigma0, 500)
    #ac_obs_traj.show()
    all_traj = plot_trajectories([[true_state_list, 'True'], [estimated_list, 'Estimated']],  3,4,5)
    all_traj.show()
    fig_c, ax_c = plt.subplots(figsize=(60,40))
    ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    ax_c.set_xlabel('X')
    ax_c.set_ylabel('Y')
    fig_c.savefig("viva/high_velocity_noise.png")
    print(belief_covariances[-1])
    # ac_obs_traj.write_html("Plots/HTML/q1a_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1a_trajectory.png")

    # ### PART A ###
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [observed_list, 'Observed']],  0, 1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1a_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1a_trajectory.png")

    # ### PART C ###
    # all_traj = plot_trajectories([[true_state_list, 'True'], [observed_list, 'Observed'], [estimated_list, 'Estimated']],  0, 1,2)
    # # all_traj.show()
    # all_traj.write_html("Plots/HTML/q1c_trajectory.html")
    # all_traj.write_image("Plots/PNG/q1c_trajectory.png")
    # print("MSE for the estimated position: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # velocity_error = distance_metric(np.array(estimated_list)[3:], np.array(true_state_list)[3:])
    # # plt.show()
    # fig_c.savefig("Plots/PNG/q1c_uncertainity.png")

    # ### PART D ###
    # rx, ry, rz = 10,10,10 # increasing noise in position update
    # rvx, rvy, rvz  = 0.01, 0.01, 0.01
    # Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    # Rt = np.eye(3)*(7**2)
    # mu0 = np.array([0,0,0,0,0,0])
    # sigma0 = np.eye(6)*((0.01)**2)
    # print('\nPART D EXPERIMENTS\n')
    # print('Varying Position Noise\n')
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [observed_list, 'Observed'], [estimated_list, 'Estimated']]
    #                                 ,  0, 1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1d_pos_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1d_pos_trajectory.png")
    # print("MSE for the estimated position with higher position update noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # fig_c.savefig("Plots/PNG/q1d_pos_uncertainity.png")

    # rx, ry, rz = 0.01,0.01,0.01 # increasing noise in position update
    # rvx, rvy, rvz  = 0.01, 0.01, 0.01
    # Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    # Rt = np.eye(3)*(7**2)
    # mu0 = np.array([0,0,0,0,0,0])
    # sigma0 = np.eye(6)*((0.01)**2)
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [observed_list, 'Observed'], [estimated_list, 'Estimated']],  0, 1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1d_posL_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1d_posL_trajectory.png")
    # print("MSE for the estimated position with lower position update noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # fig_c.savefig("Plots/PNG/q1d_posL_uncertainity.png")

    # print('\nVarying Velocity Noise\n')
    # rx, ry, rz = 1.2,1.2,1.2 # increasing noise in position update
    # rvx, rvy, rvz  = 5,5,5
    # Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    # Rt = np.eye(3)*(7**2)
    # mu0 = np.array([0,0,0,0,0,0])
    # sigma0 = np.eye(6)*((0.01)**2)
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list[:50], 'Actual'], [observed_list[:50], 'Observed'], [estimated_list[:50], 'Estimated']],  0, 1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1d_velH_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1d_velH_trajectory.png")
    # print("MSE for the estimated position with higher velovity update noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # fig_c.savefig("Plots/PNG/q1d_velH_uncertainityFull.png")

    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list[:50], belief_covariances[:50], ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # fig_c.savefig("Plots/PNG/q1d_velH_uncertainity0_50.png")

    # rx, ry, rz = 1.2,1.2,1.2 # increasing noise in position update
    # rvx, rvy, rvz  = 0.0001, 0.0001, 0.0001
    # Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    # Rt = np.eye(3)*(7**2)
    # mu0 = np.array([0,0,0,0,0,0])
    # sigma0 = np.eye(6)*((0.01)**2)
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [observed_list, 'Observed'], [estimated_list, 'Estimated']],  0, 1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1d_velL_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1d_velL_trajectory.png")
    # print("MSE for the estimated position with much lower velocity update noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # fig_c.savefig("Plots/PNG/q1d_velL_uncertainity.png")

    # print('\nVarying Sensor Noise\n')
    # rx, ry, rz = 1.2,1.2,1.2 # increasing noise in position update
    # rvx, rvy, rvz  = 0.01, 0.01, 0.01
    # Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    # Rt = np.eye(3)*(100**2)
    # mu0 = np.array([0,0,0,0,0,0])
    # sigma0 = np.eye(6)*((0.01)**2)
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [observed_list, 'Observed'], [estimated_list, 'Estimated']],  0, 1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1d_sensH_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1d_sensH_trajectory.png")
    # print("MSE for the estimated position with higher sensor noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # print("MSE between actual and observed with higher sensor  noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # fig_c.savefig("Plots/PNG/q1d_sensH_uncertainity.png")

    # rx, ry, rz = 1.2,1.2,1.2 # increasing noise in position update
    # rvx, rvy, rvz  = 0.01, 0.01, 0.01
    # Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    # Rt = np.eye(3)*(0.01**2)
    # mu0 = np.array([0,0,0,0,0,0])
    # sigma0 = np.eye(6)*((0.01)**2)
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [observed_list, 'Observed'], [estimated_list, 'Estimated']],  0, 1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1d_sensL_trajectory.html")
    # ac_obs_traj.write_image("Plots/PNG/q1d_sensL_trajectory.png")
    # print("MSE for the estimated position with lower sensor  noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # print("MSE between actual and observed with lower sensor  noise: ", distance_metric(np.array(estimated_list)[:3], np.array(true_state_list)[:3]))
    # fig_c, ax_c = plt.subplots(figsize=(60,40))
    # ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    # ax_c.set_xlabel('X')
    # ax_c.set_ylabel('Y')
    # fig_c.savefig("Plots/PNG/q1d_sensL_uncertainity.png")

    # print()
    # ### PART E ###
    # rx, ry, rz = 1.2,1.2,1.2 
    # rvx, rvy, rvz  = 0.01, 0.01, 0.01
    # Qt = np.diag([rx, ry, rz, rvx, rvy, rvz])**2
    # Rt = np.eye(3)*(7**2)
    # mu0 = np.array([0,0,0,0,0,0])
    # sigma0 = np.eye(6)*((0.01)**2)

    # leave_cond = lambda i : 1<=i<=50 or 100<=i<=150 or 200<=i<=250 or 300<=i<=350 or 400<=i<=450
    # AgentE = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_listE, observed_listE, estimated_listE, belief_covariancesE = simulate(AgentE, mu0, sigma0, 500, leave_cond)
    # traj_e = plot_trajectories([[true_state_listE, 'Actual'], [estimated_listE, 'Estimated']],  0, 1,2)
    # fig_e, ax_e = plt.subplots(figsize=(60,40))
    # ax_e = uncertainity_ellipse(estimated_listE, belief_covariancesE, ax_e,0,1)
    # traj_e.write_html("Plots/HTML/q1e_trajectory.html")
    # traj_e.write_image("Plots/PNG/q1e_trajectory.png")
    # fig_e.savefig('Plots/PNG/q1e_uncertainity.png')
    # print("MSE for the estimated position with radio silence: ", distance_metric(np.array(estimated_listE)[:3], np.array(true_state_listE)[:3]))
    # # fig_e.show()
    # # traj_e.show()

    # ### PART F ###
    # # traj_vel_f = plot_trajectories([[true_state_list, 'Actual'], [estimated_list, 'Estimated']],  3, 4,5)
    # # traj_vel_f.write_html("Plots/HTML/q1f_trajectory_all_obs.html")
    # # traj_vel_f.write_image("Plots/PNG/q1f_trajectory_all_obs.png")
    # # print("MSE for velocities with all observations: ",velocity_error )
    # # traj_vel_f.show()

    # traj_vel_fE = plot_trajectories([[true_state_listE, 'Actual'], [estimated_listE, 'Estimated']],  3, 4,5)
    # traj_vel_fE.write_html("Plots/HTML/q1f_trajectory_radio_silece.html")
    # traj_vel_fE.write_image("Plots/PNG/q1f_trajectory_radio_silece.png")
    # print("MSE for velocities with radio silence: ", distance_metric(np.array(estimated_listE)[3:], np.array(true_state_listE)[3:]))
    # # traj_vel_f.show()

    # mu1 = np.array([0,0,0,10,0,0])
    # Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    # true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu1, sigma0, 500)
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [estimated_list, 'Estimated']],  3, 4,5)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1f_trajectory_diff.html")
    # ac_obs_traj.write_image("Plots/PNG/q1f_trajectory_diff.png")
    # ac_obs_traj = plot_trajectories([[true_state_list, 'Actual'], [estimated_list, 'Estimated']],  0,1,2)
    # # ac_obs_traj.show()
    # ac_obs_traj.write_html("Plots/HTML/q1f_trajectory_loc_diff.html")
    # ac_obs_traj.write_image("Plots/PNG/q1f_trajectory_loc_diff.png")
    
