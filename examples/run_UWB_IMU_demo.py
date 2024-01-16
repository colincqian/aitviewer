# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl
import os
import torch
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from examples.load_UWB_IMU import load_uwb_imu_dataset

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

def visualize_smpl_models(path,rgb=(0.62, 0.62, 0.62),frame_range=(0,15000),stride = 2,vis_leaf_joint_position=False):
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
        betas=betas,
        trans=tran.to(C.device)
    )

    #rbs_v = RigidBodies(joints[:, joint_idxs].cpu().numpy(), v_oris,color=(0,0,0,1))
    
    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3],trans=tran)
    smpl_seq.mesh_seq.color = rgb + (1.0,)
    
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
    # This is loading the DIP-IMU data that can be downloaded from the DIP project website here:
    # https://dip.is.tue.mpg.de/download.php
    # Download the "DIP IMU and others" and point the following path to one of the extracted pickle files.
    # with open(r"/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/Dataset_B/test.pt", "rb") as f:
    #     data = pkl.load(f, encoding="latin1")
    
    # paths = [
    #          "/home/chqian/Project/PIP/PIP/data/result/20231124_marker_label_tests/PIP_vacc_vrot_vuwb_no_opt/0.pt",
    #          "/home/chqian/Project/PIP/PIP/data/result/20231124_marker_label_tests/PIP_no_opt/0.pt"
    #           ]
    joint_idxs = [20, 21, 4, 5, 15, 0]
    show_lj_pos = False
    show_imu = False
    seq_id = 2
    stride = 2
    seq_s_e = (5000,20000)
    demo_sensor_path = "/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/SIGGRAPH_dataset/demo.pt"
    paths = [
            #"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion/processed_01_pantry_take_1.pkl.pt",
           # "/home/chqian/Project/PIP/PIP/demo_output/PIP/processed_01_pantry_take_1.pkl.pt",
            #"/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusionfinetuned/processed_01_pantry_take_1.pkl.pt",
            "/home/chqian/Project/PIP/PIP/demo_output/PIP_GNN_fusion/processed_01_pantry_take_1.pkl.pt",
            "/home/chqian/Project/PIP/PIP/demo_output/PIP/processed_01_pantry_take_1.pkl.pt",
            "/home/chqian/Project/PIP/PIP/demo_output/TIP/seq_uwbimu_01_pantry_take_1.pkl.pt",
            
             ]
      
    colors = [(0.627, 0.3, 0.3),(0.3, 0.627, 0.3),(0.3, 0.3, 0.627)]
    smpl_seqs = [];rb_seq = []
    joints = []
    for p,c in zip(paths,colors):
        smpl_s,joint = visualize_smpl_models(p,rgb=c,frame_range=seq_s_e,stride=stride)
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
    v.playback_fps = 30.0

    v.scene.add(*smpl_seqs)
    if rb_seq:
        v.scene.add(*rb_seq)
    v.run()
