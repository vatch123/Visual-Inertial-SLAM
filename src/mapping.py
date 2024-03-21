import numpy as np
from tqdm import tqdm

from src.camera import get_world_coordinates
from utils import *


def get_valid_landmarks_and_mean(
		features,
    poses,
		K,
		b,
		imu_T_cam
	):

    landmark_mu = np.zeros((3, features.shape[1]))
    valid_landmarks = []
    for t_idx in tqdm(range(features.shape[2])):
        T_t = poses[:,:,t_idx]
        
        pixel_coords = features[:,:,t_idx]
        valid1 = np.logical_and(pixel_coords[0,:] != -1, pixel_coords[1,:] != -1)
        valid2 = np.logical_and(pixel_coords[2,:] != -1, pixel_coords[3,:] != -1)
        valid = np.logical_and(valid1, valid2)

        world_h_coords = np.zeros((4, pixel_coords.shape[1]))
        world_h_coords[:, valid] = get_world_coordinates(K, b, imu_T_cam, T_t, pixel_coords[:,valid])

        valid_landmark = landmark_mu == np.zeros((3,1))
        valid_landmark = np.sum(valid_landmark, axis=0) == 3

        landmark_mu[:,valid_landmark & valid] = world_h_coords[:3, valid_landmark & valid]

        valid_landmarks.append(np.where(valid)[0])
    return valid_landmarks, landmark_mu


def get_predicted_landmarks(
    poses,
		ts,
		features,
		linear_velocity,
		angular_velocity,
		K,
		b,
		imu_T_cam
	):

    imu_T_cam = flip_IMU(imu_T_cam)

    valid_landmarks, landmark_mu = get_valid_landmarks_and_mean(
      features,
      poses,
		  K,
		  b,
		  imu_T_cam
    )

    landmark_mu_flattened = landmark_mu.T.reshape(-1)
    landmark_cov = np.eye(landmark_mu_flattened.shape[0], landmark_mu_flattened.shape[0])

    Ks = np.vstack((K, np.zeros(3)))
    Ks = np.hstack((Ks, np.zeros((4,1))))
    Ks[3,:] = Ks[1,:]
    Ks[2,:] = Ks[0,:]
    Ks[2,3] = -Ks[0, 0] * b

    P = np.hstack((np.eye(3,3), np.zeros((3,1))))

    observation_noise_cov = np.eye(4,4) * 4

    for t_idx in tqdm(range(ts.shape[1])):
        T_t = poses[:,:,t_idx]
        valid = valid_landmarks[t_idx]
        Nt = valid.shape[0]
        
        obs = features[:, valid, t_idx]
        
        landmark_mu_coords_t = np.ones((3 * Nt))
        landmark_cov_t = np.zeros((3*Nt, 3*Nt))
        for i in range(Nt):
            landmark_mu_coords_t[3*i:3*i+3] = landmark_mu_flattened[3*valid[i]: 3*valid[i]+3]
            landmark_cov_t[3*i:3*i+3, 3*i:3*i+3] = landmark_cov[3*valid[i]:3*valid[i]+3, 3*valid[i]:3*valid[i]+3]

        landmark_mu_h_coords_t = landmark_mu_coords_t.reshape((-1,3)).T
        landmark_mu_h_coords_t = np.vstack((landmark_mu_h_coords_t, np.ones(landmark_mu_h_coords_t.shape[1])))
        optical_h_coords = np.linalg.inv(imu_T_cam) @ np.linalg.inv(T_t) @ landmark_mu_h_coords_t
        normalized_optical_coords = optical_h_coords / optical_h_coords[2,:]

        predicted_obs = Ks @ normalized_optical_coords
        innovation = obs - predicted_obs

        jacobian = projectionJacobian(predicted_obs.T)
        H_t = Ks @ jacobian @ np.linalg.inv(imu_T_cam) @ np.linalg.inv(T_t) @ P.T

        H = np.zeros((4*Nt, 3*Nt))
        observation_noise_cov_stacked = np.zeros((4*Nt, 4*Nt))
        for i in range(valid.shape[0]):
            H[4*i:4*i+4,3*i:3*i+3] = H_t[i,:,:]
            observation_noise_cov_stacked[4*i:4*i+4, 4*i:4*i+4] = observation_noise_cov

        Kalman_gain = landmark_cov_t @ H.T @ np.linalg.inv(H @ landmark_cov_t @ H.T + observation_noise_cov_stacked)

        landmark_mu_coords_t = landmark_mu_coords_t + np.squeeze(Kalman_gain @ innovation.T.reshape((-1,1)))
        landmark_cov_t = (np.eye(3*Nt, 3*Nt) - Kalman_gain @ H) @ landmark_cov_t

        for i in range(Nt):
            landmark_mu_flattened[3*valid[i]: 3*valid[i]+3] = landmark_mu_coords_t[3*i:3*i+3]
            landmark_cov[3*valid[i]:3*valid[i]+3, 3*valid[i]:3*valid[i]+3] = landmark_cov_t[3*i:3*i+3, 3*i:3*i+3]

    predicted_landmarks = landmark_mu_flattened.reshape((-1,3)).T
    
    return predicted_landmarks
