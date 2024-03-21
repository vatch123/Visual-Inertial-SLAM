import numpy as np
import scipy as sp
from tqdm import tqdm

from src.camera import get_world_coordinates
from utils import *


def find_motion_and_mean_exp(tau, gen_velocity):

	exp_motion_perturb = axangle2pose(gen_velocity.T * tau[:,None])

	ad_twist = axangle2adtwist(gen_velocity.T)
	mean_perturb = ad_twist * -tau[:, None, None]
	exp_mean_perturb = np.zeros_like(mean_perturb)
	for i in range(mean_perturb.shape[0]):
		exp_mean_perturb[i,:,:] = sp.linalg.expm(mean_perturb[i,:,:])
	
	return exp_motion_perturb, exp_mean_perturb


def find_localization_poses_landmarks(
		ts,
		features,
		linear_velocity,
		angular_velocity,
		K,
		b,
		imu_T_cam
	):

	imu_T_cam = flip_IMU(imu_T_cam)
    
	tau = np.zeros_like(ts)
	tau[0, 1:] = ts[0, 1:] - ts[0, :-1]
	tau = np.squeeze(tau)

	gen_velocity = np.vstack((linear_velocity, angular_velocity))

	exp_motion_perturb, exp_mean_perturb = find_motion_and_mean_exp(tau, gen_velocity)

	mu = np.eye(4,4)
	sigma = np.eye(6,6)
	delta_mu = np.random.multivariate_normal(mean=np.zeros((6,)), cov=sigma)

	noise_cov = np.eye(6,6) * 0.01

	trajectory = [mu]
	landmarks = np.zeros((3, features.shape[1]))

	for t_idx in tqdm(range(1, tau.shape[0])):
		mu = trajectory[-1]
		noise = np.random.multivariate_normal(mean=np.zeros((6,)), cov=noise_cov)
		
		# Motion model
		mu_t = mu @ exp_motion_perturb[t_idx,:,:]
		delta_mu_t = exp_mean_perturb[t_idx,:,:] @ delta_mu + noise

		sigma_t = exp_mean_perturb[t_idx,:,:] @ sigma @ exp_mean_perturb[t_idx,:,:].T + noise_cov

		trajectory.append(mu_t)
		delta_mu = delta_mu_t
		sigma = sigma_t

		pixel_coords = features[:,:,t_idx]
		valid1 = np.logical_and(pixel_coords[0,:] != -1, pixel_coords[1,:] != -1)
		valid2 = np.logical_and(pixel_coords[2,:] != -1, pixel_coords[3,:] != -1)
		valid = np.logical_and(valid1, valid2)
		pixel_coords = pixel_coords[:,valid]

		world_h_coords = get_world_coordinates(K, b, imu_T_cam, mu_t, pixel_coords)
		landmarks[:, valid] = world_h_coords[:3,:]

	poses = np.stack(trajectory)
	poses = poses.transpose((1,2,0))

	return poses, landmarks

