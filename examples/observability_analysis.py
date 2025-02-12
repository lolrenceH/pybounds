# usage: python3 /src/tools/pybounds/examples/observability_analysis.py /src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_wind_0.01_train_std/eval/plume_7999_b1efe0c993b1c07badde7ee2b516ae04/noisy3x5b5.pkl
import gc
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pybounds import Simulator, SlidingEmpiricalObservabilityMatrix, FisherObservability, SlidingFisherObservability, ObservabilityMatrixImage, colorline
import jax
import jax.numpy as jnp
import pickle
import numpy as np
import tamagotchi.eval.log_analysis as log_analysis

def log_shapes(pytree):
    def log_leaf(path, leaf=()):
        if isinstance(leaf, jnp.ndarray) or isinstance(leaf, np.ndarray):
            print(f"Path: {path}, Shape: {leaf.shape}")
    jax.tree_util.tree_map_with_path(log_leaf, pytree)
    

def f(X, U):
    '''
    Return Xdot given X and U
    
    X: state vector
        v_para: ground velocity in the direction parallel to head direction (egocentric frame) [m/s]
        v_perp: ground velocity perpendicular to head direction (egocentric frame) [m/s]
        phi: heading [rad]
        w: wind speed [m/s]
        zeta: wind angle [rad]
    U: input vector assuming actions have been squashed and scaled
        u_para: translational speed [m/s]
        u_phi: angular velocity [rad/s]
        u_para_dot: translational acceleration [m/s^2]
    '''
    # States
    # v_para, v_perp, phi, w, zeta = X
    x, y, v_para, v_perp, phi, w, zeta = X
    
    # Inputs
    u_para, u_phi, u_para_dot, u_zeta_dot, u_w_dot = U # keep u_para because it is in the observation matrix

    # Dynamics
    # w_dot = 0*w # wind speed is constant
    w_dot = u_w_dot # wind speed is constant
    zeta_dot = u_zeta_dot # for discontinuous wind direction change
    phi_dot = u_phi # angular velocity is controlled by agent
    v_perp_dot = -w * np.cos(phi - zeta) * u_phi + w * np.cos(phi - zeta) * zeta_dot - w_dot * np.sin(phi - zeta) 
    v_para_dot = -w * np.sin(phi - zeta) * u_phi + w * np.sin(phi - zeta) * zeta_dot + w_dot * np.cos(phi - zeta) + u_para_dot
    x_dot = v_para * np.cos(phi) - v_perp * np.sin(phi)
    y_dot = v_para * np.sin(phi) + v_perp * np.cos(phi)

    # Package and return Xdot
    X_dot = [x_dot, y_dot, v_para_dot, v_perp_dot, phi_dot, w_dot, zeta_dot]

    return X_dot


def h(X, U):
    '''
    Measurement functions - input is the state and control input; output is the measurement
    Assuming control signals are squashed and scaled
    '''
    # States
    x, y, v_para, v_perp, phi, w, zeta = X
    # v_para, v_perp, phi, w, zeta = X
    
    # Inputs
    u_para, u_phi, u_para_dot, u_zeta_dot, u_w_dot = U
    
    # Measurements
    # Heading
    phi = phi # heading is directly observable 
    # Apparent wind
    appWind = - u_para # equal and opposite to translational speed which is in line with head direction and thus have only a parallel component
    # Course direction in fly reference frame
    psi = np.arctan2(v_perp, v_para) # drift angle / egocentric course angle  # TODO numerically check if this is actually the drift angle - yes according to Ben
    
    # Unwrap the angles s.t. they are continuous - no more snapping back to 0; this is important for the observability analysis
    if np.array(phi).ndim > 0:
        if np.array(phi).shape[0] > 1:
            phi = np.unwrap(phi)
            psi = np.unwrap(psi)

    # Measurements
    Y  = [phi, appWind, psi]

    # Return measurement
    return Y



def squash_and_scale_actions(raw_actions, dt):
    '''
    See /src/JH_boilerplate/dev_test/env/explore_pkl.ipynb
    
    Logged actions are the raw outputs of the agent. They need to be translated into the control inputs for the simulator.
    1. Squash the actions to [0, 1]
    2. Scale the actions to the fly's capabilities
    3. Calculate the translational acceleration - new need to be checked
    4. Return a dictionary of control inputs for the simulator
    
    Note at agent_angle_rad_t0, this is obtained by the following steps:
      - t = -1: env generates an obs by env.reset()
      - t = 0: policy creates an action based on the obs
      - t = 0: env.step(action)
          - t = 0: obs_t0 = obs_t(-1) + action_t0  (obs_t-1 based on how the angle was initialized which is NOT documented in the eps logs)
      - t = 1: env generates an obs
    '''
    
    def squash_action(x):
      return np.clip((np.tanh(x) + 1)/2, 0.0, 1.0) # squash action, center and scale to [0, 1], per action treatment 

    if type(raw_actions) is list:
      raw_actions = np.stack(raw_actions)
      
    # Vectorize the function # TODO sanity check
    vsquash_action = np.vectorize(squash_action)

    # Apply the vectorized function to the array
    actions = vsquash_action(raw_actions)

    # Scale actions by fly capabilities 
    actions[:, 0] = actions[:, 0] * 2.0 # Max agent speed in m/s
    actions[:, 1] = (actions[:, 1] - 0.5) * 6.25*np.pi # Max agent CW/CCW turn per second

    # Calculate translational acceleration
    acc = np.diff(actions[:, 0]) / dt # checked - this is correct
    # acc = np.insert(acc, 0, 0) # first acceleration is 0
    
    # Omit the first action. See function description.
    u_sim = {'u_para': actions[1:, 0], 'u_phi': actions[1:, 1], 'u_para_dot': acc} 
    # print('u_sim shapes', u_sim['u_para'].shape, u_sim['u_phi'].shape, u_sim['u_para_dot'].shape)
    
    return u_sim


# set up the simulator
state_names = [
                'x',  # x position [m]
                'y',  # y position [m]
                'v_para',  # parallel ground velocity [m/s]
                'v_perp',  # perpendicular ground velocity [m/s]
                'phi', # heading [rad]
                'w',  # ambient wind speed [m/s]
                'zeta',  # ambient wind angle [rad]
                ]

input_names = [
                'u_para',  # translational speed [m/s]
                'u_phi',  # angular velocity [rad/s]
                'u_para_dot',  # translational acceleration [m/s^2] 
                'u_zeta_dot',
                'u_w_dot'  
                ]
measurement_names = ['phi', 'appWind', 'psi'] # heading, apparent wind parallel component, drift angle/egocentric course angle
dt = 0.04  # [s]

import sys, os
# load the episode logs
log_fname = sys.argv[1]
# log_fname = '/src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_wind_0.001_debug_yes_vec_norm_train_actor_std/eval/plume_3492_45513dd8ac9d9cdbb3a34f957436f7af/noisy3x5b5.pkl'
# load pkl file
with open(log_fname, 'rb') as f_handle:
    episode_logs = pickle.load(f_handle)
print('Loaded episode logs from', log_fname)
print('Number of episodes:', len(episode_logs))
print('Episodes contain:', episode_logs[0].keys())
# print('For more info on pkl content see /src/JH_boilerplate/dev_test/env/explore_pkl.ipynb')

# load the selected_df 
number_of_eps = 240 # pull all episodes
dataset = 'noisy3x5b5' # TODO set by the user
exp_folder = 'eval' # TODO set by the user
eval_folder = os.path.dirname(log_fname) + '/'
selected_df = log_analysis.get_selected_df(eval_folder, [dataset],
                                        n_episodes_home=40,
                                        n_episodes_other=0,  
                                        balanced=True,
                                        oob_only=False,
                                        verbose=True)

traj_df_stacked, stacked_neural_activity = log_analysis.get_traj_and_activity_and_stack_them(selected_df, 
                                                                                            obtain_neural_activity = True, 
                                                                                            obtain_traj_df = True, 
                                                                                            get_traj_tmp = True,
                                                                                            extended_metadata = True) # get_traj_tmp 
analysis_results = []
for eps_idx in traj_df_stacked['ep_idx'].unique():
    start_time = time.time()
    simulator = Simulator(f, h, dt=dt, state_names=state_names, input_names=input_names, measurement_names=measurement_names)
    # load the action data
    raw_actions = episode_logs[eps_idx]['actions']
    # stack a list of actions into a 2D array
    raw_actions = np.stack(raw_actions)
    u_sim = squash_and_scale_actions(raw_actions, dt) # can be pulled from traj_df_stacked['step'] and 'turn'; just need to scale but already squashed

    # load the trajectory data
    epoch_traj_df = traj_df_stacked[traj_df_stacked['ep_idx'] == eps_idx]
    epoch_latent_activity = stacked_neural_activity[epoch_traj_df.index]

    gt_dict = {'x':[], 'y':[], 'v_para': [], 'v_perp': [], 'phi': [], 'w': [], 'zeta': [], 
            'psi_ego_course_dir': [], 'v_allo': []}
    gt_dict['x'] = epoch_traj_df['loc_x'].values
    gt_dict['y'] = epoch_traj_df['loc_y'].values
    gt_dict['phi'] = np.angle(epoch_traj_df['agent_angle_x'] + 1j*epoch_traj_df['agent_angle_y'], deg=False)
    gt_dict['w'] = np.round(epoch_traj_df['wind_speed_ground'].values, 3)
    gt_dict['zeta'] = epoch_traj_df['wind_angle_ground_theta'].values # normalized by pi and then shifted to 0-1
    gt_dict['psi_ego_course_dir'] = epoch_traj_df['ego_course_direction_theta'].values # normalized by pi and then shifted to 0-1
    gt_dict['v_allo'] = np.stack(epoch_traj_df['allo_ground_velocity'].values) # PEv3: (np.array(self.agent_location) - self.agent_location_last)/self.dt; calc'd in get_eval_dfs_and_stack_them
    gt_dict['v_allo_dt'] = np.diff(gt_dict['v_allo']) / dt # acceleration
    # scale angles from 0-1 to -pi to pi
    gt_dict['zeta'] = np.pi * (2*gt_dict['zeta'] - 1)
    gt_dict['psi_ego_course_dir'] = np.pi * (2*gt_dict['psi_ego_course_dir'] - 1)

    gt_dict['v_para'] = np.cos(gt_dict['psi_ego_course_dir']) * np.linalg.norm(np.stack(gt_dict['v_allo']), axis=1) # cos(psi) * g = v_para
    gt_dict['v_perp'] = np.sin(gt_dict['psi_ego_course_dir']) * np.linalg.norm(np.stack(gt_dict['v_allo']), axis=1) # sin(psi) * g = v_perp

    gt_dict['phi'] = np.unwrap(gt_dict['phi'])

    u_sim['u_zeta_dot'] = np.diff(gt_dict['zeta']) / dt
    u_sim['u_w_dot'] = np.diff(gt_dict['w']) / dt


    # simulate the episode to get the ground truth states and measurements
    x0 = {'x': gt_dict['x'][0], 'y': gt_dict['y'][0], 'v_para': gt_dict['v_para'][0], 'v_perp': gt_dict['v_perp'][0], 'phi': gt_dict['phi'][0], 'w': gt_dict['w'][0], 'zeta': gt_dict['zeta'][0]}
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=x0, mpc=False, u=u_sim, return_full_output=True)

    # Choose sensors to use from O
    o_sensors = ['phi', 'appWind', 'psi']

    # Chose states to use from O
    o_states = [
                    # 'x',  # x position [m]
                    # 'y',  # y position [m]
                    'v_para',  # parallel ground velocity [m/s]
                    'v_perp',  # perpendicular ground velocity [m/s]
                    # 'phi', # heading [rad]
                    'w',  # ambient wind speed [m/s]
                    'zeta',  # ambient wind angle [rad]
                    ]


    # Choose time-steps to use from O
    window_size = 10 # TODO set by the user
    o_time_steps = np.arange(0, window_size, step=1)
    sensor_noise = {'phi': 0, 'appWind': 0, 'psi': 0} 
    # Construct O in sliding windows
    SEOM = SlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=window_size, eps=1e-6)
    # Compute Fisher information matrix & inverse for each sliding window
    SFO = SlidingFisherObservability(SEOM.O_df_sliding, time=SEOM.t_sim, lam=1e-6, R=0.1, #sensor_noise_dict=sensor_noise,
                                    states=o_states, sensors=o_sensors, time_steps=o_time_steps, w=None)
    # Pull out minimum error variance, 'time' column is the time vector shifted forward by w/2 and 'time_initial' is the original time
    EV_aligned = SFO.get_minimum_error_variance()
    EV_no_nan = EV_aligned.fillna(method='bfill').fillna(method='ffill')

    # Save analysis results
    analysis_results.append([EV_no_nan, t_sim, x_sim, window_size, eps_idx])
    # NOTE: zeta is in degrees
    # clear the simulator
    del simulator
    # garbage collect
    
    gc.collect()
    print('Analysis complete for episode', eps_idx, 'in', time.time() - start_time, 'seconds')

print('Analysis complete')

# Save the analysis results
analysis_results_fname = log_fname.replace('.pkl', '_observability_test.pkl')
with open(analysis_results_fname, 'wb') as f_handle:
    pickle.dump(analysis_results, f_handle)
print('Saved analysis results to', analysis_results_fname)

# # plot the minimum error variance on trajectory # TODO wrap into a function and plot with gridspec
# states = list(SFO.FO[0].O.columns)
# n_state = len(states)

# fig, ax = plt.subplots(n_state, 2, figsize=(6, n_state*2), dpi=150)
# ax = np.atleast_2d(ax)

# cmap = 'inferno_r'

# min_ev = np.min(EV_no_nan.iloc[:, 2:].values)
# max_ev = np.max(EV_no_nan.iloc[:, 2:].values)

# log_tick_high = int(np.ceil(np.log10(max_ev)))
# log_tick_low = int(np.floor(np.log10(min_ev)))
# cnorm = mpl.colors.LogNorm(10**log_tick_low, 10**log_tick_high)

# for n, state_name in enumerate(states):
#     # colorline(t_sim, x_sim[state_name], EV_no_nan[state_name].values, ax=ax[n, 0], cmap=cmap, norm=cnorm)
#     colorline(x_sim['x'], x_sim['y'], EV_no_nan[state_name].values, ax=ax[n, 0], cmap=cmap, norm=cnorm)
#     colorline(t_sim, EV_no_nan[state_name].values, EV_no_nan[state_name].values, ax=ax[n, 1], cmap=cmap, norm=cnorm)

#     # Colorbar
#     cax = ax[n, -1].inset_axes([1.03, 0.0, 0.04, 1.0])
#     cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cax,
#                         ticks=np.logspace(log_tick_low, log_tick_high, log_tick_high-log_tick_low + 1))
#     cbar.set_label('min. EV: ' + state_name, rotation=270, fontsize=7, labelpad=8)
#     cbar.ax.tick_params(labelsize=6)
    
#     ax[n, 0].set_ylim(np.min(x_sim['y']) - 0.01, np.max(x_sim['y']) + 0.01)
#     ax[n, 0].set_xlim(np.min(x_sim['x']) - 0.01, np.max(x_sim['x']) + 0.01)
#     ax[n, 0].set_ylabel('y', fontsize=7)
#     ax[n, 0].set_xlabel('x', fontsize=7)
#     ax[n, 0].set_aspect(1.0)

#     ax[n, 1].set_ylim(10**log_tick_low, 10**log_tick_high)
#     ax[n, 1].set_yscale('log')
#     ax[n, 1].set_ylabel('min. EV: ' + state_name, fontsize=7)
#     ax[n, 1].set_yticks(np.logspace(log_tick_low, log_tick_high, log_tick_high-log_tick_low + 1))


# for a in ax.flat:
#     a.tick_params(axis='both', labelsize=6)
    
# for a in ax[:, 1]:
#     a.set_xlabel('time (s)', fontsize=7)
#     a.set_xlim(-0.1, t_sim[-1] + 0.1)
    
# # for a in ax[:, 1]:
# #     a.set_xlim(-0.1, t_sim[-1] + 0.1)

# fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)

# plt.show()