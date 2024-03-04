# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl
import os
import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle,matrix_to_quaternion,quaternion_to_axis_angle
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from examples.load_UWB_IMU import load_uwb_imu_dataset
from aitviewer.headless import HeadlessRenderer

def mean_filter(data, window_size):
    """
    Applies a mean filter to the data using PyTorch.

    :param data: The input data tensor.
    :param window_size: The size of the window for the mean filter.
    :return: The filtered data.
    """
    # Create the mean filter kernel
    kernel = torch.ones(1, 1, window_size) / window_size
    kernel = kernel.to(data.device)  # Move kernel to the same device as data

    # Apply the convolution
    data_unsqueezed = data.permute(1,0).unsqueeze(1)  # Add two dimensions for batch and channel
    filtered = F.conv1d(data_unsqueezed, kernel, padding=window_size//2)
    return filtered.squeeze(1).permute(1,0)

def mean_filter_3d(data, window_size):
    """
    Apply a mean filter to 3D data across frames.

    Parameters:
    data (numpy array): The 3D array of data to filter, with shape (num_frames, num_vertices, 3).
    window_size (int): The size of the window for filtering across frames (must be an odd integer).

    Returns:
    filtered_data (numpy array): The filtered data.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer")

    num_frames, num_vertices, _ = data.shape
    half_window = window_size // 2
    extended_data = np.pad(data, ((half_window, half_window), (0, 0), (0, 0)), mode='reflect')
    
    # Initialize an empty array for the filtered data
    filtered_data = np.zeros_like(data)

    # Apply the mean filter
    for i in range(num_frames):
        start_index = i
        end_index = i + window_size
        filtered_data[i, :, :] = np.mean(extended_data[start_index:end_index, :, :], axis=0)

    return filtered_data


def load_imu(path,seq_name,frame_range=(0,15000),stride=2):
    data = torch.load(path)

    seq_id = None
    for i,file_path in enumerate(data["fnames"]):
        if seq_name == os.path.basename(file_path)+'.pt':
            seq_id = i
            break
    if seq_id is None:
        raise Exception(f"{seq_name} not Found !!")
    
    # Get the data.
    oris = data["ori"][seq_id].numpy().reshape(-1,6,3,3)
    accs = data["acc"][seq_id].numpy().reshape(-1,6,3)
    if "vuwb" in data:
        uwb = data["vuwb"][seq_id].numpy().reshape(-1,6,6)
        uwb = uwb[frame_range[0]:frame_range[1]:1]
    else:
        uwb = None

    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"
    oris = oris[frame_range[0]:frame_range[1]:stride]
    accs = accs[frame_range[0]:frame_range[1]:stride]


    return oris,accs

def visualize_smpl_models(path,rgb=(0.62, 0.62, 0.62),frame_range=(0,15000),stride = 2,filter_output=False,name=''):
    data = torch.load(path)
    uwb_imu_rot = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1, 0]])
    # Get the data.
    poses = matrix_to_axis_angle(data[0]).view(-1,72)
    tran = data[1].view(-1,3)
    
    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[frame_range[0]:frame_range[1]:stride]
    tran = tran[frame_range[0]:frame_range[1]:stride]
    # DIP has no shape information, assume the mean shape.
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].to(C.device),
        poses_root=poses[:, :3].to(C.device),
        betas=betas.to(C.device),
        trans=tran.to(C.device)
    )

    #rbs_v = RigidBodies(joints[:, joint_idxs].cpu().numpy(), v_oris,color=(0,0,0,1))
    
    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3],trans=tran,is_rigged=False)
    smpl_seq.mesh_seq.color = rgb + (1.0,)
    smpl_seq.name = name
    if filter_output:
        smpl_seq.mesh_seq.vertices = mean_filter_3d(smpl_seq.mesh_seq.vertices,window_size=5)
    
    return smpl_seq,joints

def visualize_leaf_joint_position(path,joints,frame_range=(0,15000),stride = 2):
    data = torch.load(path)
    
    # Get the data.
    tran = data[1].view(-1,3)
    root_ori = data[0][:,0]
    leaf_joint_position = data[2].view(-1,5,3)
    
    # Downsample to 30 Hz.
    #leaf_joint_position = leaf_joint_position[:seq_end:stride,[2,3,4,0,1]]
    leaf_joint_position = leaf_joint_position[frame_range[0]:frame_range[1]:stride]
    root_ori = root_ori[frame_range[0]:frame_range[1]:stride]
    tran = tran[frame_range[0]:frame_range[1]:stride]
    f,n,_ = leaf_joint_position.size()
    leaf_joint_position = leaf_joint_position @ root_ori.permute(0,2,1)
  
    root_position = np.tile(joints[:,0].cpu().numpy(),(1,5)).reshape(f,n,3)
    arr_head = Arrows(origins=root_position[:,[2]], tips=root_position[:,[2]] - leaf_joint_position[:,[2]].cpu().numpy(),color=(0,0,0.5,1))#b
    arr_leg = Arrows(origins=root_position[:,[0,1]], tips=root_position[:,[0,1]] - leaf_joint_position[:,[0,1]].cpu().numpy(),color=(0,0.5,0,1))#g
    arr_upper = Arrows(origins=root_position[:,[3,4]], tips=root_position[:,[3,4]] - leaf_joint_position[:,[3,4]].cpu().numpy(),color=(0.5,0,0,1))#r
    #rb = RigidBodies(leaf_joint_position.cpu().numpy(), np.tile(np.eye(3),(f,n,1,1)).reshape(f,n,3,3),color=(0,0,0,1))
    
    return [arr_head,arr_leg,arr_upper]
if __name__ == "__main__":
    take_name = ["processed_01_pantry_take_1.pkl.pt","processed_02_pantry_take_2.pkl.pt","processed_03_pantry_take_3.pkl.pt",
                 "processed_04_sports_take_1.pkl.pt","processed_05_sports_take_2.pkl.pt","processed_06_garden_take_1.pkl.pt",
                 "processed_07_garden_take_2.pkl.pt","processed_08_CAB_take_1.pkl.pt","processed_09_CAB_take_2.pkl.pt"]
    take_name_tip = ["seq_uwbimu_01_pantry_take_1.pkl.pt","seq_uwbimu_02_pantry_take_2.pkl.pt","seq_uwbimu_03_pantry_take_3.pkl.pt",
                 "seq_uwbimu_04_sports_take_1.pkl.pt","seq_uwbimu_05_sports_take_2.pkl.pt","seq_uwbimu_06_garden_take_1.pkl.pt",
                 "seq_uwbimu_07_garden_take_2.pkl.pt","seq_uwbimu_08_CAB_take_1.pkl.pt","seq_uwbimu_09_CAB_take_2.pkl.pt"]
    joint_idxs = [20, 21, 4, 5, 15, 0]
    show_lj_pos = False
    show_imu = False
    seq_id = 4
    stride = 2
    seq_s_e = (0,15000)
    demo_sensor_path = "/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/SIGGRAPH_dataset/demo.pt"
    paths = [
            #"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion/processed_01_pantry_take_1.pkl.pt",
           # "/home/chqian/Project/PIP/PIP/demo_output/PIP/processed_01_pantry_take_1.pkl.pt",
            #"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusionfinetuned/processed_01_pantry_take_1.pkl.pt",
            #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusionhip_ct/{take_name[seq_id]}",
            #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_ft_gt_uwb/{take_name[seq_id]}",
            #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_ft_gt_uwb_0_1/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_gt_uwb_0_1/{take_name[seq_id]}",
            # #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_MSE_loss/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_no_recon/{take_name[seq_id]}",
            # #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_vacc_vrot_vuwb/{take_name[seq_id]}",
            # #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_vacc_vrot_vuwb_ct/{take_name[seq_id]}",
            # #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion/{take_name[seq_id]}",
            # #f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_0_1/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP_[syn_train]/{take_name[seq_id]}",
            # #f"/home/chqian/Project/PIP/PIP/demo_output/PIP/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/TIP_pretrain/{take_name_tip[seq_id]}"
            # #"/home/chqian/Project/PIP/PIP/demo_output/TIP/seq_uwbimu_01_pantry_take_1.pkl.pt",


            f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_ft_gt_uwb/{take_name[seq_id]}",
            f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_gt_uwb_0_1/{take_name[seq_id]}",
            f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_no_recon/{take_name[seq_id]}",
            f"/home/chqian/Project/PIP/PIP/demo_output/PIP_vacc_vrot_vuwb_ct/{take_name[seq_id]}",
                
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_ft_gt_uwb/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_gt_uwb_0_1/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion_no_recon_finetune/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP_vacc_vrot_vuwb_ct/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/PIP[syn]/{take_name[seq_id]}",
            # f"/home/chqian/Project/PIP/PIP/demo_output/TIP/{take_name_tip[seq_id]}" 
             ]
    filter_mask = [True,True,True,True,False,False]
    names = ["ours_ft",
            "ours_0_1",
            "ours_RFT",
            "pip_uwb",
            "pip",
            "tip"]
    colors = [(73/255,90/255,136/255), #
                  (73/255,90/255,136/255), 
                  (73/255,90/255,136/255), #
                  (73/255,90/255,136/255), #
                  (0.0539, 0.526, 0.0539), #PIP
                  (210/255, 155/255, 42/255)] #

    #colors = [(73/255,90/255,136/255),(73/255,90/255,136/255),(0.0539, 0.526, 0.0539),(210/255, 155/255, 42/255),(150/255, 150/255, 150/255)]
    smpl_seqs = [];rb_seq = []
    joints = []
    for p,c,m in zip(paths,colors,filter_mask):
        smpl_s,joint = visualize_smpl_models(p,rgb=c,frame_range=seq_s_e,stride=stride,filter_output=m,name=os.path.basename(os.path.dirname(p)))
        smpl_seqs.append(smpl_s)
        f = 60//stride
        jerk = ((joint[3:] - 3 * joint[2:-1] + 3 * joint[1:-2] - joint[:-3]) * (f ** 3)).norm(dim=2).mean()
        print("jittering",jerk/1000)
        if show_lj_pos:
            rb_seq.extend(visualize_leaf_joint_position(p,joint,frame_range=seq_s_e,stride=stride))
        
        if show_imu:
            oris,accs = load_imu(demo_sensor_path,os.path.basename(p),frame_range=seq_s_e,stride=stride)
            rbs = RigidBodies(joint[:, joint_idxs].cpu().numpy(), oris)
            arr = Arrows(origins=joint[:, joint_idxs].cpu().numpy(),tips = joint[:, joint_idxs].cpu().numpy() - accs * 1/30)
            rb_seq.extend([rbs,arr])
        

    
    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 60.0
    v.scene.add(smpl_s)
    v.reset_camera()
    #v.scene.camera.load_cam_file(f"take_{seq_id+1}_gopro.pkl")
    v.scene.camera.load_cam_file(f"take_{seq_id+1}_phone.pkl")
    v.scene.add(*smpl_seqs)
    if rb_seq:
        v.scene.add(*rb_seq)
    v.run()
