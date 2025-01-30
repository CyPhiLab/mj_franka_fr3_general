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
robot_URDF = pinocchio_model_dir + "/assets/urdf/fr3.urdf"
robot = RobotWrapper.BuildFromURDF(robot_URDF, pinocchio_model_dir)
from pinocchio.visualize import MeshcatVisualizer
robot.setVisualizer(MeshcatVisualizer())
robot.initViewer(open=True)
robot.loadViewerModel()
robot.display(robot.q0)
print(robot.q0)
input("Press Enter to close MeshCat and terminate... ")
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
    # C = pin.computeCoriolisMatrix(robot.model, robot.data, data.qpos, data.qvel)
    # G = pin.computeGeneralizedGravity(robot.model, robot.data, data.qpos)
    C, G = get_coriolis_and_gravity(model, data)
        # go to set point
    th_dd = -kp * (data.qpos - q_func(data.time)) - kd * (data.qvel - qd_func(data.time))

    # This controller cancels the dynamics and imposes whatever physics we want
    tau = C @ qd_func(data.time) + G + np.matmul(M, th_dd)
    data.ctrl[:] = tau


def get_coriolis_and_gravity(model, data):
    """
    Calculate the Coriolis matrix and gravity vector for a MuJoCo model
    
    Parameters:
        model: MuJoCo model object
        data: MuJoCo data object
    
    Returns:
        C: Coriolis matrix (nv x nv)
        g: Gravity vector (nv,)
    """
    nv = model.nv  # number of degrees of freedom
    
    # Calculate gravity vector
    g = np.zeros(nv)
    dummy = np.zeros(7,)
    mujoco.mj_factorM(model, data)  # Compute sparse M factorization
    mujoco.mj_rne(model, data, 0, dummy)  # Run RNE with zero acceleration and velocity
    g = data.qfrc_bias.copy()
    
    # Calculate Coriolis matrix
    C = np.zeros((nv, nv))
    q_vel = data.qvel.copy()
    
    # Compute each column of C using finite differences
    eps = 1e-6
    for i in range(nv):
        # Save current state
        vel_orig = q_vel.copy()
        
        # Perturb velocity
        q_vel[i] += eps
        data.qvel = q_vel
        
        # Calculate forces with perturbed velocity
        mujoco.mj_rne(model, data, 0, dummy)
        tau_plus = data.qfrc_bias.copy()
        
        # Restore original velocity
        q_vel = vel_orig
        data.qvel = q_vel
        
        # Compute column of C using finite difference
        C[:, i] = (tau_plus - data.qfrc_bias) / eps
    
    return C, g

def verify_dynamics(model, data, q, v):
    """
    Verify the computed dynamics against MuJoCo's internal computations
    
    Parameters:
        model: MuJoCo model object
        data: MuJoCo data object
        q: Joint positions
        v: Joint velocities
    
    Returns:
        bool: True if verification passes within tolerance
    """
    # Set the state
    data.qpos = q
    data.qvel = v
    mujoco.mj_forward(model, data)
    
    # Get our computed matrices
    C, g = get_coriolis_and_gravity(model, data)
    dummy = np.zeros(7,)
    # Compare against MuJoCo's internal computations
    mujoco.mj_rne(model, data, 0, dummy)
    expected_cv = data.qfrc_bias - g
    computed_cv = C @ v
    
    # Check if results match within tolerance
    tol = 1e-5
    return np.allclose(expected_cv, computed_cv, atol=tol, rtol=tol)

print(verify_dynamics(model, data, waypoints[2,:], np.ones(7,)))

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