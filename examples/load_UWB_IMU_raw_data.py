# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl

import torch
import numpy as np
import scipy.spatial.transform as sct
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
import pandas as pd

def load_csv(csv_file: str, resample = False):
    '''
    Load gt quaternion and position from csv file
    '''
    def resample_time_series(dataframe, time_col: str,val_col: list, desired_time_stamp: str):
        # Convert the time column to a datetime object
        dataframe[time_col] = pd.to_datetime(dataframe[time_col], unit='ms')
        dataframe.set_index(time_col, inplace=True)
        # Create a regular time index with the desired frequency
        resample_time_indx = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq=desired_time_stamp)
        incoming_df = pd.DataFrame({time_col: resample_time_indx})
        for k in val_col:
            incoming_df[k] = np.nan 
        incoming_df.set_index(incoming_df[time_col], inplace=True)
        
        merge_df = dataframe.merge(incoming_df, left_index=True, right_index=True, how='outer', suffixes=('', '_incoming'))[val_col]
        
        merge_df.interpolate(method="linear",inplace=True)

        return merge_df.loc[resample_time_indx]
    
    data = pd.read_csv(csv_file)
    device_key = "device"
    device_ids = [0,1,2,3,4,5]
    devices_id_val = set(data[device_key])
    keys = ["time[ms]","x_gt","y_gt","z_gt","qx_gt","qy_gt","qz_gt","qw_gt"]
    output_data = []
    for dev_id in device_ids:
        if dev_id not in devices_id_val:
            output_data.append([]) #missing device
            print("detect missing device!")
            continue

        device_data = data.query(f"{device_key}=={dev_id}")[keys]

        if resample:
            imu_df = device_data[keys]
            imu_df = resample_time_series(imu_df, time_col="time[ms]",
                                        val_col=["x_gt","y_gt","z_gt","qx_gt","qy_gt","qz_gt","qw_gt"],
                                        desired_time_stamp="33ms")
        else:
            imu_df = device_data[keys[1:]]
            
        print(imu_df)
        
        output_data.append(np.array(imu_df)[None,...])
    output_data = np.concatenate(output_data,axis=0)
    
    return output_data.transpose(1,0,2) #frame, device, dim

def get_valid_transformation(rotmat, position):
    '''
    Get ori,tran for imu in smpl coordinate frame to build rigid body
    '''
    uwb_rot = torch.tensor([[0, 1, 0], [0, 0, 1.0], [1, 0, 0]]) 
    
    imu_ori = torch.from_numpy(rotmat).float()
    imu_ori = torch.einsum('ij,abjk->abik',uwb_rot, imu_ori)
    
    imu_position = torch.from_numpy(position).float()
    imu_position = torch.einsum('ij,abj->abi',uwb_rot, imu_position)
    
    return imu_ori, imu_position




if __name__ == "__main__":
    csv_file_path = "/home/chqian/DATA/Motion_capture_raw_dataset/Dataset_B/subject_4/freestyle/04_freestyle_devices_0/dataset_imu.csv"
    output_data = load_csv(csv_file_path,resample=True)
    frame_num, device_num, dim = output_data.shape
    pos_device = output_data[:,:,:3]
    device_ori = output_data[:,:,3:]
    rotmat_device = sct.Rotation.from_quat(device_ori.reshape(-1,4)).as_matrix().reshape(frame_num,device_num,3,3)
    
    imu_ori,imu_pos = get_valid_transformation(rotmat_device, pos_device)
    
    rb = RigidBodies(imu_pos.cpu().numpy(), imu_ori.cpu().numpy(),color=(0,0,0,1))


    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(rb)
    v.run()
    
