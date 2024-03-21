import numpy as np


def get_world_coordinates(K, b, imu_T_cam, T, pixel_coordinates):
    Fsu = K[0,0]
    Fsv = K[1,1]
    Cu = K[0,2]
    Cv = K[1,2]

    uL = pixel_coordinates[0,:]
    vL = pixel_coordinates[1,:]
    uR = pixel_coordinates[2,:]
    vR = pixel_coordinates[3,:]

    z = Fsu * b / (uL - uR)
    x = z * (uL - Cu) / Fsu
    y = z * (vL - Cv) / Fsv
    ones = np.ones_like(z)

    camera_h_coordinates = np.stack((x,y,z,ones))
    imu_h_coodinates = imu_T_cam @ camera_h_coordinates
    world_h_coordinates = T @ imu_h_coodinates

    return world_h_coordinates
