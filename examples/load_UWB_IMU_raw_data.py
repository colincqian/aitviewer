# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl

import torch
import numpy as np
import scipy.spatial.transform as sct
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
import pandas as pd

uwb_imu_mapping = torch.tensor([1, 2, 4, 5, 3, 0])
def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

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
    #keys = ["time[ms]","x_gt","y_gt","z_gt","qx_gt","qy_gt","qz_gt","qw_gt"]
    keys = ["time","x","y","z","qx","qy","qz","qw"]
    output_data = []
    for dev_id in device_ids:
        if dev_id not in devices_id_val:
            output_data.append([]) #missing device
            print("detect missing device!")
            continue

        device_data = data.query(f"{device_key}=={dev_id}")[keys]

        if resample:
            imu_df = device_data[keys]
            # imu_df = resample_time_series(imu_df, time_col="time[ms]",
            #                             val_col=["x_gt","y_gt","z_gt","qx_gt","qy_gt","qz_gt","qw_gt"],
            #                             desired_time_stamp="16.667ms")
            imu_df = resample_time_series(imu_df, time_col="time",
                                        val_col=["x","y","z","qx","qy","qz","qw"],
                                        desired_time_stamp="16.667ms")
        else:
            imu_df = device_data[keys[1:]]
            
        print(imu_df)
        
        output_data.append(np.array(imu_df)[None,...])
    output_data = np.concatenate(output_data,axis=0)
    
    return output_data.transpose(1,0,2) #frame, device, dim

def get_valid_transformation(imu_ori, imu_acc):
    '''
    Get ori,tran for imu in smpl coordinate frame to build rigid body
    '''
    uwb_rot = torch.tensor([[0, 1, 0], [0, 0, 1.0], [1, 0, 0]])
    offset = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]) 
    uwb_rot_i = offset.matmul(uwb_rot)
    
    imu_ori = torch.einsum('ij,abjk->abik',uwb_rot, imu_ori)
    imu_acc = torch.einsum('ij,abj->abi',uwb_rot, imu_acc)
    
    return imu_ori, imu_acc

def process_imu_ori(imu_ori: torch.Tensor, align_init_ori = False):
    #Our assumption
    init_root_ori = torch.eye(3)
    #get target imu ori at initial
    if align_init_ori:
        #align all ori at time 0 to be identity, assume identical ori for all imu (syn)
        init_imu_target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]).tile(1,1)
    else:
        #only assume root orientation to be identity, asuume root ori is identity, other imu is not identical to root(real)
        init_imu_target = init_root_ori @ imu_ori[[0],[0],...].unsqueeze(0).permute(0,1,3,2) @ imu_ori[[0],...]
        
    init_imu_ori = imu_ori[[0],...]
    #offset at time 0
    cal_mat_offset = init_imu_target.matmul(init_imu_ori.permute(0,1,3,2))
    #Rotation from time 0 to time t
    rot_0_t = imu_ori.matmul(init_imu_ori.permute(0,1,3,2))
    #offset at time t
    cal_mat_offset = rot_0_t @ cal_mat_offset @ rot_0_t.permute(0,1,3,2) 
    
    imu_ori = cal_mat_offset.matmul(imu_ori) 

    return imu_ori

if __name__ == "__main__":
    #csv_file_path = "/home/chqian/DATA/Motion_capture_raw_dataset/Dataset_B/subject_4/freestyle/04_freestyle_devices_0/dataset_imu.csv"
    csv_file_path = "/home/chqian/DATA/UWB_test_data/20231121_markers_tests-20231122T092610Z-001/20231121_markers_tests/aligned_dense_markers.csv"
    output_data = load_csv(csv_file_path,resample=True)
    frame_num, device_num, dim = output_data.shape
    
    pos_device = output_data[:,:,:3]
    device_ori = output_data[:,:,3:]
    rotmat_device = sct.Rotation.from_quat(device_ori.reshape(-1,4)).as_matrix().reshape(frame_num,device_num,3,3)
    
    #Transform to smpl space (z_up to y_up)
    imu_ori = torch.from_numpy(rotmat_device).float()
    imu_pos = torch.from_numpy(pos_device).float()
    imu_ori,imu_pos = get_valid_transformation(imu_ori,imu_pos)
    imu_acc = _syn_acc(imu_pos)
    
    imu_ori = process_imu_ori(imu_ori,align_init_ori=True)

    rb = RigidBodies(imu_pos.cpu().numpy(), imu_ori.cpu().numpy(),color=(0,0,0,1))
    arr = Arrows(origins=imu_pos.cpu().numpy(), tips=imu_pos.cpu().numpy() + 0.01 * imu_acc.cpu().numpy())
    frame_num = imu_ori.size(0)
    torch.save({'acc': [imu_acc[:,uwb_imu_mapping]], 'ori': [imu_ori[:,uwb_imu_mapping]], 'pose': [torch.zeros(frame_num,24,3)], 'tran': [torch.zeros(frame_num,3)]},
               "/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/Dataset_B_not_align/test.pt")
    # # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(rb,arr)
    v.run()
    
