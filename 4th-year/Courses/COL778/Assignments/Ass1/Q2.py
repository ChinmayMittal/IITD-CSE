from utils import *
from matplotlib import pyplot as plt
from Q1 import AeroplaneAgent, simulate
import plotly.graph_objects as go
class LandmarkAeroplane():

    def __init__(self, init_pos, At, Bt, Ct, Qt, Rt, Rt_landmark, landmark_pos):

        self.state = init_pos
        self.At = At
        self.Bt = Bt
        self.Ct = Ct
        self.Qt = Qt
        self.Rt = Rt
        self.R_landmark = Rt_landmark
        self.landmarks = landmark_pos
        self.belief = None

    def fetchState(self):
        return self.state
    
    def updateState(self, u):
        self.state = np.matmul(self.At, self.state) + np.matmul(self.Bt, u) + np.random.multivariate_normal(np.zeros(self.Qt.shape[0]),self.Qt)

    def sensor_outputs(self):
        return np.matmul(self.Ct, self.state) + np.random.multivariate_normal(np.zeros(self.Rt.shape[0]),self.Rt)

    def getObservation(self):
        observation, landmark = self.evalH(self.state)
        observation = observation + np.random.multivariate_normal(np.zeros(self.R_landmark.shape[0]),self.R_landmark)
        assert abs(observation[0]) != np.inf or abs(observation[1]) != np.inf or abs(observation[2]) != np.inf
        return observation, landmark
    
    def evalH(self, x):
        dist = [np.linalg.norm(landmark - self.state[:3]) for landmark, limit in (self.landmarks)]
        best_landmark = np.argmin(dist)
        if dist[best_landmark] > self.landmarks[best_landmark][1]:
            return x[:4], -1
        else:
            return np.concatenate([x[:3], np.array([np.linalg.norm(self.landmarks[best_landmark][0] - x[:3])])]), best_landmark
    
    def getJacobian(self, x):
        H = np.zeros((4,6))
        H[:3,:3] = np.eye(3)
        dist = [np.linalg.norm(landmark - self.state[:3]) for landmark, limit in (self.landmarks)]
        best_landmark = np.argmin(dist)
        H[3,:3] = (x[:3] - self.landmarks[best_landmark][0])/np.linalg.norm(x[:3] - self.landmarks[best_landmark][0])
        return H
    
class ExtendedKalmannFilter():

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

    def extendedMeasurementUpdate(self):
        z, landmark = self.agent.getObservation()
        if landmark == -1: #No landmark within limits
            z = self.agent.sensor_outputs()
            self.measurementUpdate(z[:3])
        else: #Landmark within limits
            H = self.agent.getJacobian(self.belief['mu'])
            inter1 = np.linalg.inv(np.matmul(H, np.matmul(self.belief['sigma'], H.T)) + self.agent.R_landmark)
            Kt = np.matmul(np.matmul(self.belief['sigma'], H.T), inter1)
            self.belief['mu'] = self.belief['mu'] + np.matmul(Kt, z - self.agent.evalH(self.belief['mu'])[0])
            self.belief['sigma'] = np.matmul(np.identity(self.belief['sigma'].shape[0]) - np.matmul(Kt, H), self.belief['sigma'])

    def extendedPredictionUpdate(self, u):
        self.predictionUpdate(u)
    
    def extendedUpdateBelief(self, u = None, z = None):
        if u is not None:
            self.extendedPredictionUpdate(u)
        if z is not None:
            self.extendedMeasurementUpdate()
        else:
            print("No update provided")

def getIncrementLandmark(t):
    return np.array([-0.128*np.cos(0.032*t), -0.128*np.sin(0.032*t), 0.01])

def simulateLandmark(Agent,mu0, sigma0, simulation_iterations = 500, leave_obs_cond = lambda i : i < -1):
    AgentEstimator = ExtendedKalmannFilter(Agent, {'mu':mu0, 'sigma':sigma0})
    true_state_list = [Agent.state]
    observed_list = [Agent.getObservation()]
    estimated_list = [mu0]
    belief_covariances = [sigma0]
    beliefs = [mu0]
    for i in range(1,simulation_iterations+1):
        try:
            Agent.updateState(u = getIncrementLandmark(i))

            if leave_obs_cond(i):
                AgentEstimator.extendedUpdateBelief(u = getIncrementLandmark(i))
            else:
                AgentEstimator.extendedUpdateBelief(u = getIncrementLandmark(i), z = 1)
        except:
            print(beliefs)
            raise
        Agent.belief = AgentEstimator.belief
        true_state_list.append(Agent.state)
        observed_list.append(Agent.getObservation())
        estimated_list.append(Agent.belief['mu'])
        belief_covariances.append(Agent.belief['sigma'])
    return true_state_list, observed_list, estimated_list, belief_covariances

if __name__ == "__main__":
    A = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    B = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    C = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
    Qt = np.eye(6) * (0.01**2)
    Rt = np.eye(3)*(10**2)
    Rt_landmark = np.eye(4)*(10**2)
    Rt_landmark[3,3] = 1
    mu0 = np.array([100,0,0,0,4,0])
    sigma0 = np.eye(6)*((0.01)**2)
    simulation_iterations = 200

    landmarks = [(np.array((150, 0, 100)),90), (np.array((-150, 0, 100)),90), (np.array((0, 150, 100)),90) , (np.array((0, -150, 100)),90)]
    Agent = LandmarkAeroplane(mu0, A, B, C, Qt, Rt, Rt_landmark, landmarks)
    true_state_list, observed_list, estimated_list, belief_covariances = simulateLandmark(Agent, mu0, sigma0, simulation_iterations)
    fig = plot_trajectories([(true_state_list, 'Actual'), (estimated_list, 'Estimated')], 0, 1, 2)
    fig.add_trace(go.Scatter3d(mode='markers',x=[150],y=[0],z=[100],
                               name = 'Landmark 1',marker=dict(color=px.colors.sequential.YlOrBr,size=[180],sizemode='diameter')))
    fig.add_trace(go.Scatter3d(mode='markers',x=[-150],y=[0],z=[100],
                               name = 'Landmark 2',marker=dict(color=px.colors.qualitative.Light24,size=[180],sizemode='diameter')))
    fig.add_trace(go.Scatter3d(mode='markers',x=[0],y=[150],z=[100],
                               name = 'Landmark 3',marker=dict(color=px.colors.plotlyjs.RdBu,size=[180],sizemode='diameter')))
    fig.add_trace(go.Scatter3d(mode='markers',x=[0],y=[-150],z=[100],
                               name = 'Landmark 4',marker=dict(color=px.colors.plotlyjs.Viridis,size=[180],sizemode='diameter')))
    fig.write_html('Plots/HTML/Q3_c.html')
    fig.write_image('Plots/PNG/Q3_c.png')
    fig_c, ax_c = plt.subplots(figsize=(60,40))
    ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    ax_c.set_xlabel('X')
    ax_c.set_ylabel('Y')
    fig_c.savefig('Plots/PNG/Q3_c_uncertainty.png')
    print("MSE between true and estimated states with landmark observations: ", distance_metric(np.array(true_state_list), np.array(estimated_list)))

    Agent1 = AeroplaneAgent(mu0, A, B, C, Qt, Rt)
    true_state_list, observed_list, estimated_list, belief_covariances = simulate(Agent1, mu0, sigma0, simulation_iterations)
    print("MSE between true and estimated states without landmark: ", distance_metric(np.array(true_state_list), np.array(estimated_list)))

    ## PART D##
    print('\nExperiments for Part D\n')
    mu0 = np.array([100,0,0,0,4,0])
    sigma0 = np.eye(6)*((0.01)**2)
    Rt_landmark[3][3] = 0.01
    print("Standard Deviation = 0.1")
    Agent = LandmarkAeroplane(mu0, A, B, C, Qt, Rt, Rt_landmark, landmarks)
    true_state_list, observed_list, estimated_list, belief_covariances = simulateLandmark(Agent, mu0, sigma0, simulation_iterations)
    fig = plot_trajectories([(true_state_list, 'Actual'), (estimated_list, 'Estimated')], 0, 1, 2)
    fig.write_html('Plots/HTML/Q3_d1.html')
    fig_c, ax_c = plt.subplots(figsize=(60,40))
    ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    ax_c.set_xlabel('X')
    ax_c.set_ylabel('Y')
    fig_c.savefig('Plots/PNG/Q3_d1.png')
    print("MSE between true and estimated states with landmark observations: ", distance_metric(np.array(true_state_list), np.array(estimated_list)))

    print()
    Rt_landmark[3][3] = 400
    mu0 = np.array([100,0,0,0,4,0])
    sigma0 = np.eye(6)*((0.01)**2)
    print("Standard Deviation = 20")
    Agent = LandmarkAeroplane(mu0, A, B, C, Qt, Rt, Rt_landmark, landmarks)
    true_state_list, observed_list, estimated_list, belief_covariances = simulateLandmark(Agent, mu0, sigma0, simulation_iterations)
    fig = plot_trajectories([(true_state_list, 'Actual'), (estimated_list, 'Estimated')], 0, 1, 2)
    fig.write_html('Plots/HTML/Q3_d2.html')
    fig_c, ax_c = plt.subplots(figsize=(60,40))
    ax_c = uncertainity_ellipse(estimated_list, belief_covariances, ax_c,0,1)
    ax_c.set_xlabel('X')
    ax_c.set_ylabel('Y')
    fig_c.savefig('Plots/PNG/Q3_d2.png')
    print("MSE between true and estimated states with landmark observations: ", distance_metric(np.array(true_state_list), np.array(estimated_list)))

'''
MSE between true and estimated states with landmark observations:  3.055037127929626
MSE between true and estimated states without landmark:  3.547325035841461

Experiments for Part D

Standard Deviation = 0.1
MSE between true and estimated states with landmark observations:  1.8586983496417018

Standard Deviation = 20
MSE between true and estimated states with landmark observations:  3.293268142549564
'''