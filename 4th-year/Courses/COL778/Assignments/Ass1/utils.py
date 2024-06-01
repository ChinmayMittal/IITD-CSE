import numpy as np
import plotly.express as px
def uncertainity_ellipse(mean, covariances, ax, ax1 = 0, ax2 = 2):
    for i in range(1,len(covariances)):
        theta = np.linspace(0, 2 * np.pi, 10000)
        eigenvalues, eigenvectors = np.linalg.eig(covariances[i][ax1:ax2+1,ax1:ax2+1])
        eigenvectors = eigenvectors.T
        a = 1/np.sqrt(1/eigenvalues[0])
        b = 1/np.sqrt(1/eigenvalues[1])
        rotation_angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        center = mean[i][ax1:ax2+1]
        ellipse_points = a * np.cos(theta)[:, np.newaxis] * eigenvectors[:, 0] + b * np.sin(theta)[:, np.newaxis] * eigenvectors[:, 1]
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
        rotated_points = np.dot(rotation_matrix, ellipse_points.T).T
        rotated_points += center
        ax.plot(rotated_points[:, 0], rotated_points[:, 1], 'b-')
    mean_temp = np.array(mean)
    ax.plot(mean_temp[:,ax1], mean_temp[:,ax2], 'r-', linewidth = 5)
    return ax

def plot_trajectories(trajs_to_plot,  ax1 = 0, ax2 = 1, ax3 = 2):
    combined_pos = []
    label_list = []
    for i in range(len(trajs_to_plot)):
        combined_pos.extend((trajs_to_plot[i][0]))
        label_list.extend([trajs_to_plot[i][1]]*len(trajs_to_plot[i][0]))
    dataFram = {'x': [i[ax1] for i in combined_pos], 'y': [i[ax2] for i in combined_pos], 'z': [i[ax3] for i in combined_pos], 'label': label_list}
    fig = px.line_3d(dataFram, x = 'x', y = 'y', z = 'z', color = 'label')
    return fig

def distance_metric(estimated, true):
    return (np.linalg.norm(estimated - true)/np.sqrt(len(estimated)))