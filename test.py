import argparse
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import os
from os.path import dirname, join, abspath
from sys import argv

from pathlib import Path
import time

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"

pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "mj_franka_fr3_general")
 
robot = RobotWrapper.BuildFromURDF("fr3.urdf", package_dirs=pinocchio_model_dir)
model = mujoco.MjModel.from_xml_path("fr3.xml")
data = mujoco.MjData(model)

#@title Spline Trajectory Generator
import numpy as np
import scipy.interpolate as spi

def create_cubic_traj(waypoints, timepoints):
    """
    Creates callable functions for cubic polynomial trajectory evaluation.
    
    Args:
        waypoints: Array of waypoints with shape (n_points, n_dims) where
                  n_points is the number of waypoints and n_dims is the
                  dimension of each waypoint (e.g., 2 for 2D points, 3 for 3D points)
        timepoints: Array of corresponding time points for each waypoint with shape (n_points,)
    
    Returns:
        q_func: Function that returns position at time t
        qd_func: Function that returns velocity at time t
        qdd_func: Function that returns acceleration at time t
    """
    # Convert inputs to numpy arrays if they aren't already
    waypoints = np.array(waypoints)
    timepoints = np.array(timepoints)
    
    # Check input dimensions
    if len(waypoints.shape) != 2:
        raise ValueError("Waypoints must be a 2D array with shape (n_points, n_dims)")
    if len(timepoints.shape) != 1:
        raise ValueError("Timepoints must be a 1D array")
    if waypoints.shape[0] != timepoints.shape[0]:
        raise ValueError("Number of waypoints must match number of timepoints")
    
    n_dims = waypoints.shape[1]
    splines = []
    
    # Create cubic spline interpolation for each dimension
    for dim in range(n_dims):
        spline = spi.CubicSpline(timepoints, waypoints[:, dim])
        splines.append(spline)
    
    def q_func(t):
        """Evaluate position at time t"""
        t = np.asarray(t)
        result = np.zeros(t.shape + (n_dims,)) if t.shape else np.zeros(n_dims)
        for dim in range(n_dims):
            result[..., dim] = splines[dim](t)
        return result
    
    def qd_func(t):
        """Evaluate velocity at time t"""
        t = np.asarray(t)
        result = np.zeros(t.shape + (n_dims,)) if t.shape else np.zeros(n_dims)
        for dim in range(n_dims):
            result[..., dim] = splines[dim](t, 1)
        return result
    
    def qdd_func(t):
        """Evaluate acceleration at time t"""
        t = np.asarray(t)
        result = np.zeros(t.shape + (n_dims,)) if t.shape else np.zeros(n_dims)
        for dim in range(n_dims):
            result[..., dim] = splines[dim](t, 2)
        return result
    
    return q_func, qd_func, qdd_func

# Example with 2D waypoints (x, y coordinates)
waypoints = np.random.uniform(low=model.jnt_range[:,0],high=model.jnt_range[:,1],size=(4,7))
waypoints[0,:] = 0.0
timepoints = np.linspace(0,10,4)
print(waypoints)
print(model.jnt_range)
# t_samples = np.linspace(0, 10, 100)

# q, qd, qdd = cubic_poly_traj(waypoints, timepoints, t_samples)

q_func, qd_func, qdd_func = create_cubic_traj(waypoints, timepoints)

# Multiple time points for plotting
t_samples = np.linspace(0, 10, 100)

def controller(model, data):
    # this controller is called by mujoco during the control loop
    kp = 10
    kd = 2 * np.sqrt(kp)
    M = np.zeros((7,7))
    # Get mass matrix and coriolis + gravity from mujoco
    mujoco.mj_fullM(model, M, data.qM)
    C = data.qfrc_bias
        # go to set point
    th_dd = -kp * (data.qpos - q_func(data.time)) - kd * (data.qvel - qd_func(data.time))

    # This controller cancels the dynamics and imposes whatever physics we want
    tau = C + np.matmul(M, th_dd)
    data.ctrl[:] = tau

mujoco.set_mjcb_control(controller)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # viewer.cam.azimuth = 90
    # viewer.cam.elevation = 5
    # viewer.cam.distance =  8
    # viewer.cam.lookat = np.array([0.0 , 0.0 , 0.0])
    while viewer.is_running():
        step_start = time.time()
        first_time = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)
        

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)