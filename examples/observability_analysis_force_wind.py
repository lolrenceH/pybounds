# Refactor of observability_analysis.py using the force-physics dynamical system of
# observability_analysis_force_wind.ipynb (PlumeEnvironment_v3, action_physics == 'force').
#
# usage: python3 /src/tools/pybounds/examples/observability_analysis_force_wind.py <eval_log.pkl> <window_size>
# e.g.:  python3 observability_analysis_force_wind.py /src/data/.../eval/plume_XXXX/noisy3x5b5.pkl 10
#
# Differences from observability_analysis.py:
# - Dynamics: body-frame rigid-body force model (thrust + linear drag on airspeed, torque + yaw
#   damping) equivalent to the env's world-frame semi-implicit Euler integrator; phi_dot is a
#   true state (initialized from the logged ang_vel) instead of a control input.
# - Measurements: heading, ego course direction, and the apparent-wind vector the agent actually
#   observes (appW_para/appW_perp = -air velocity in the body frame); appWind = -u_para no longer
#   holds under force physics.
# - Actions: 3-dim [T_par, T_perp, tau] decoded per env.step() (squash then scale by
#   T_para_max / T_perp_max / tau_max from conf/physics), instead of the kinematic speed/turn scaling.
# - Always fits all trials (n_home = n_other = 1000); dataset name comes from the eval file name.
import gc
import time
import os
import sys
import pickle
import numpy as np
from pybounds import Simulator, SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability
import tamagotchi.eval.log_analysis as log_analysis


def log_shapes(pytree):
    # print shapes of arrays in a dict
    for key, leaf in pytree.items():
        if isinstance(leaf, np.ndarray):
            print(f"Path: {key}, Shape: {leaf.shape}")


# ----------------------------------------------------------------------------------------------
# Physics coefficients (conf/physics/fly.yaml, SI units) - MUST match the physics config the
# eval logs were generated with. Loaded from the tamagotchi package when importable so the
# analysis stays in sync with the env; hardcoded fallback otherwise.
# ----------------------------------------------------------------------------------------------
force_physics = {
    'T_para_max': 2.90e-6,  # [N] parallel thrust max; action[0] in [0,1] -> [0, T_para_max]
    'T_perp_max': 1.30e-6,  # [N] perpendicular thrust max; action[1] -> [-T_perp_max, T_perp_max]
    'tau_max':    1.41e-11, # [N*m] yaw torque max; action[2] -> [-tau_max, tau_max]
    'mass':       1.0e-6,   # [kg]
    'drag':       1.93e-6,  # [N*s/m] isotropic linear drag on airspeed
    'inertia':    5.2e-13,  # [kg*m^2] yaw moment of inertia
    'k_rot':      1.61e-12, # [N*m*s] yaw damping
}
try:
    import yaml
    from pathlib import Path
    import tamagotchi
    yaml_path = Path(tamagotchi.__file__).parent / 'conf' / 'physics' / 'fly.yaml'
    with open(yaml_path) as f_yaml:
        loaded = yaml.safe_load(f_yaml)
    force_physics.update({k: loaded[k] for k in force_physics if k in loaded})
    print(f"Loaded force physics coefficients from {yaml_path}")
except Exception as e:
    print(f"Using hardcoded fly.yaml coefficients ({e})")

M      = force_physics['mass']
C_PARA = force_physics['drag']   # isotropic in PEv3: C_para = C_perp = drag
C_PERP = force_physics['drag']
I_YAW  = force_physics['inertia']
C_PHI  = force_physics['k_rot']


def f(X, U):
    '''
    Return Xdot given X and U.

    Body-frame form of PlumeEnvironment_v3 action_physics == 'force' (see env.py step()):
        m*v_dot    = R(phi)[u_para, u_perp] - drag*(v - wind)   (world frame)
        I*phi_ddot = u_phi - k_rot*phi_dot

    X: state vector
        x, y: position [m]
        v_para: ground velocity parallel to head direction (egocentric frame) [m/s]
        v_perp: ground velocity perpendicular to head direction (egocentric frame) [m/s]
        phi: heading [rad]
        phi_dot: angular velocity [rad/s] - integrator state under force physics
        w: wind speed [m/s]
        zeta: wind angle [rad]
    U: input vector assuming actions have been squashed and scaled (see squash_and_scale_actions)
        u_para: parallel thrust force [N]
        u_perp: perpendicular thrust force [N]
        u_phi: yaw torque [N*m]
        u_zeta_dot: wind direction rate [rad/s] - tracks logged wind changes
        u_w_dot: wind speed rate [m/s^2]
    '''
    # States
    x, y, v_para, v_perp, phi, phi_dot, w, zeta = X

    # Inputs
    u_para, u_perp, u_phi, u_zeta_dot, u_w_dot = U

    # Air velocity (body frame)
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)

    # Acceleration (thrust + linear drag on airspeed + rotating-frame coupling)
    v_para_dot = ((u_para - C_PARA * a_para) / M) + (v_perp * phi_dot)
    v_perp_dot = ((u_perp - C_PERP * a_perp) / M) - (v_para * phi_dot)

    # Angular acceleration
    phi_ddot = (u_phi - C_PHI * phi_dot) / I_YAW

    # Wind dynamics driven by inputs (constant wind: both zero). Unlike the kinematic system,
    # no wind-rate coupling terms in v_para_dot/v_perp_dot: ground velocity is a true integrator
    # state here, and wind enters only through the drag force.
    w_dot = u_w_dot
    zeta_dot = u_zeta_dot

    # Position dynamics
    x_dot = v_para * np.cos(phi) - v_perp * np.sin(phi)
    y_dot = v_para * np.sin(phi) + v_perp * np.cos(phi)

    # Package and return Xdot
    X_dot = [x_dot, y_dot, v_para_dot, v_perp_dot, phi_dot, phi_ddot, w_dot, zeta_dot]

    return X_dot


def h(X, U):
    '''
    Measurement function - input is the state and control input; output is the measurement.

    Matches the PEv3 observation channels (env.py sense_environment()):
        phi: heading (obs head_x/y)
        psi: egocentric course direction / drift angle (obs course_x/y)
        apparent wind (obs wind_x/y = -air velocity rotated into the body frame), in two
        equivalent parameterizations selectable via o_sensors:
            gamma, a: apparent airflow angle & magnitude (fly_wind_example convention)
            appW_para, appW_perp = -(a_para, a_perp): the vector components the agent receives
    '''
    # States
    x, y, v_para, v_perp, phi, phi_dot, w, zeta = X

    # Inputs
    u_para, u_perp, u_phi, u_zeta_dot, u_w_dot = U

    # Air velocity (body frame)
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    a = np.sqrt(a_para ** 2 + a_perp ** 2)
    gamma = np.arctan2(a_perp, a_para)  # air velocity angle

    # Apparent wind vector sensed by the agent (egocentric) = -air velocity
    appW_para = -a_para
    appW_perp = -a_perp

    # Course direction in fly reference frame (drift angle)
    psi = np.arctan2(v_perp, v_para)

    # Unwrap the angles s.t. they are continuous - no more snapping back to 0; this is important
    # for the observability analysis
    if np.array(phi).ndim > 0:
        if np.array(phi).shape[0] > 1:
            phi = np.unwrap(phi)
            psi = np.unwrap(psi)
            gamma = np.unwrap(gamma)

    # Measurements
    Y = [phi, psi, gamma, a, appW_para, appW_perp]

    # Return measurement
    return Y


def squash_and_scale_actions(raw_actions, pc):
    '''
    Decode logged raw policy outputs into the physical control inputs of the force model.
    Mirrors env.py step() under action_physics == 'force':
      1. squash: (tanh(x) + 1) / 2, then clip to [0, 1]
      2. scale:  u_para = a0 * T_para_max              in [0, T_para_max]
                 u_perp = (a1 - 0.5) * 2 * T_perp_max  in [-T_perp_max, T_perp_max]
                 u_phi  = (a2 - 0.5) * 2 * tau_max     in [-tau_max, tau_max]

    Timing (same convention as observability_analysis.py): infos[i] logs the state AFTER
    action[i] was applied, so the transition s_i -> s_{i+1} uses action[i+1]. We simulate from
    the first logged state, so the first action is omitted.
    '''
    if type(raw_actions) is list:
        raw_actions = np.stack(raw_actions)
    assert raw_actions.shape[1] == 3, (
        f"Expected 3-dim force-physics actions [T_par, T_perp, tau], got {raw_actions.shape[1]}-dim. "
        "These logs are not from an action_physics='force' agent.")

    actions = np.clip((np.tanh(raw_actions) + 1) / 2, 0.0, 1.0)  # squash, per env step()

    # Omit the first action. See function description.
    u_sim = {'u_para': actions[1:, 0] * pc['T_para_max'],
             'u_perp': (actions[1:, 1] - 0.5) * 2.0 * pc['T_perp_max'],
             'u_phi':  (actions[1:, 2] - 0.5) * 2.0 * pc['tau_max']}
    # print('u_sim shapes', u_sim['u_para'].shape, u_sim['u_perp'].shape, u_sim['u_phi'].shape)

    return u_sim


# set up the simulator
state_names = [
                'x',        # x position [m]
                'y',        # y position [m]
                'v_para',   # parallel ground velocity [m/s]
                'v_perp',   # perpendicular ground velocity [m/s]
                'phi',      # heading [rad]
                'phi_dot',  # angular velocity [rad/s]
                'w',        # ambient wind speed [m/s]
                'zeta',     # ambient wind angle [rad]
                ]

input_names = [
                'u_para',     # parallel thrust force [N]
                'u_perp',     # perpendicular thrust force [N]
                'u_phi',      # yaw torque [N*m]
                'u_zeta_dot', # wind direction rate [rad/s]
                'u_w_dot',    # wind speed rate [m/s^2]
                ]
measurement_names = ['phi', 'psi', 'gamma', 'a', 'appW_para', 'appW_perp']
dt = 0.1  # [s] - must match the env step (config.env dt) the logs were generated with

# load the episode logs
log_fname = sys.argv[1]

window_size = int(sys.argv[2])

# Always fit on all trials
n_home = 1000  # pull all
n_other = 1000  # pull all
print('Fitting ALL trials')

# load pkl file
with open(log_fname, 'rb') as f_handle:
    episode_logs = pickle.load(f_handle)
print('Loaded episode logs from', log_fname)
print('Number of episodes:', len(episode_logs))
print('Episodes contain:', episode_logs[0].keys())

# load the selected_df
dataset = os.path.basename(log_fname).replace('.pkl', '')  # name of the eval file without the pickle suffix

eval_folder = os.path.dirname(log_fname) + '/'
selected_df = log_analysis.get_selected_df(eval_folder, [dataset],
                                        n_episodes_home=n_home,
                                        n_episodes_other=n_other,
                                        balanced=False,
                                        oob_only=False,
                                        verbose=True,
                                        log_fname=log_fname) # separately provide log_fname instead of concat. dataset with eval_folder

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
    u_sim = squash_and_scale_actions(raw_actions, force_physics)

    # load the trajectory data
    epoch_traj_df = traj_df_stacked[traj_df_stacked['ep_idx'] == eps_idx]
    epoch_latent_activity = stacked_neural_activity[epoch_traj_df.index]

    gt_dict = {'x':[], 'y':[], 'v_para': [], 'v_perp': [], 'phi': [], 'phi_dot': [], 'w': [], 'zeta': [],
            'psi_ego_course_dir': [], 'v_allo': []}
    gt_dict['x'] = epoch_traj_df['loc_x'].values
    gt_dict['y'] = epoch_traj_df['loc_y'].values
    gt_dict['phi'] = np.angle(epoch_traj_df['agent_angle_x'] + 1j*epoch_traj_df['agent_angle_y'], deg=False)
    gt_dict['phi_dot'] = epoch_traj_df['angular_velocity'].values # logged env state under force physics (info 'ang_vel')
    gt_dict['w'] = np.round(epoch_traj_df['wind_speed_ground'].values, 3)
    gt_dict['zeta'] = epoch_traj_df['wind_angle_ground_theta'].values # normalized by pi and then shifted to 0-1
    gt_dict['psi_ego_course_dir'] = epoch_traj_df['ego_course_direction_theta'].values # normalized by pi and then shifted to 0-1
    gt_dict['v_allo'] = np.stack(epoch_traj_df['allo_ground_velocity'].values) # true integrator state under force physics
    # scale angles from 0-1 to -pi to pi
    gt_dict['zeta'] = np.pi * (2*gt_dict['zeta'] - 1)
    gt_dict['psi_ego_course_dir'] = np.pi * (2*gt_dict['psi_ego_course_dir'] - 1)

    gt_dict['v_para'] = np.cos(gt_dict['psi_ego_course_dir']) * np.linalg.norm(np.stack(gt_dict['v_allo']), axis=1) # cos(psi) * g = v_para
    gt_dict['v_perp'] = np.sin(gt_dict['psi_ego_course_dir']) * np.linalg.norm(np.stack(gt_dict['v_allo']), axis=1) # sin(psi) * g = v_perp

    gt_dict['phi'] = np.unwrap(gt_dict['phi'])

    # wind-change inputs (aligned with u_sim: transition s_i -> s_{i+1} uses the wind logged at i+1)
    u_sim['u_zeta_dot'] = np.diff(np.unwrap(gt_dict['zeta'])) / dt
    u_sim['u_w_dot'] = np.diff(gt_dict['w']) / dt


    # simulate the episode to get the ground truth states and measurements
    x0 = {'x': gt_dict['x'][0], 'y': gt_dict['y'][0],
          'v_para': gt_dict['v_para'][0], 'v_perp': gt_dict['v_perp'][0],
          'phi': gt_dict['phi'][0], 'phi_dot': gt_dict['phi_dot'][0],
          'w': gt_dict['w'][0], 'zeta': gt_dict['zeta'][0]}
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=x0, mpc=False, u=u_sim, return_full_output=True)

    # Choose sensors to use from O: what the agent observes - heading, ego course direction,
    # and the apparent-wind vector (well-conditioned near zero airspeed, unlike gamma/a)
    o_sensors = ['phi', 'psi', 'appW_para', 'appW_perp']

    # Chose states to use from O (jointly estimated in the Fisher information)
    o_states = [
                    # 'x',       # x position [m]
                    # 'y',       # y position [m]
                    'v_para',    # parallel ground velocity [m/s]
                    'v_perp',    # perpendicular ground velocity [m/s]
                    # 'phi',     # heading [rad] - directly measured
                    'phi_dot',   # angular velocity [rad/s]
                    'w',         # ambient wind speed [m/s]
                    'zeta',      # ambient wind angle [rad]
                    ]


    # Choose time-steps to use from O
    o_time_steps = np.arange(0, window_size, step=1)
    # Construct O in sliding windows
    SEOM = SlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=window_size, eps=1e-6)
    # Compute Fisher information matrix & inverse for each sliding window
    SFO = SlidingFisherObservability(SEOM.O_df_sliding, time=SEOM.t_sim, lam=1e-6, R=0.000001, #sensor_noise_dict=sensor_noise,
                                    states=o_states, sensors=o_sensors, time_steps=o_time_steps, w=None)
    # Pull out minimum error variance, 'time' column is the time vector shifted forward by w/2 and 'time_initial' is the original time
    EV_aligned = SFO.get_minimum_error_variance()
    EV_no_nan = EV_aligned.bfill().ffill()  # same as fillna(method=...), which pandas 3 removed
    # Keep only the wind states in the saved output (the Fisher computation above still jointly
    # estimates all o_states; this only trims the saved columns)
    EV_no_nan = EV_no_nan[['time', 'time_initial', 'w', 'zeta']]

    # Save analysis results
    analysis_results.append([EV_no_nan, t_sim, x_sim, window_size, eps_idx])
    # clear the simulator
    del simulator
    # garbage collect

    gc.collect()
    print('Analysis complete for episode', eps_idx, 'in', time.time() - start_time, 'seconds')

print('Analysis complete')

# Save the analysis results
analysis_results_fname = log_fname.replace('.pkl', f'_w{window_size}_observability_test.pkl')
with open(analysis_results_fname, 'wb') as f_handle:
    pickle.dump(analysis_results, f_handle)
print('Saved analysis results to', analysis_results_fname)
