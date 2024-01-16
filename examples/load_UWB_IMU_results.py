# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl

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

def visualize_smpl_models(path,rgb=(0.62, 0.62, 0.62),seq_end=-1,stride = 2,vis_leaf_joint_position=False):
    data = torch.load(path)
    uwb_imu_rot = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1, 0]])
    
    # Get the data.
    poses = matrix_to_axis_angle(data[0]).view(-1,72)
    tran = data[1].view(-1,3)
        
    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[:seq_end:stride]
    tran = tran[:seq_end:stride]
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
    smpl_seq.mesh_seq.color = rgb + (0.5,)
    
    return smpl_seq,joints

def visualize_leaf_joint_position(path,joints,seq_end=-1,stride = 2):
    data = torch.load(path)
    
    # Get the data.
    tran = data[1].view(-1,3)
    root_ori = data[0][:,0]
    leaf_joint_position = data[2].view(-1,5,3)
    
    # Downsample to 30 Hz.
    #leaf_joint_position = leaf_joint_position[:seq_end:stride,[2,3,4,0,1]]
    leaf_joint_position = leaf_joint_position[:seq_end:stride]
    root_ori = root_ori[:seq_end:stride]
    tran = tran[:seq_end:stride]
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
    show_lj_pos = True
    seq_id = 2
    paths = [
            #f"/home/chqian/Project/PIP/PIP/data/result/SIGGRAPH_dataset/edge_dp0.4_L0.01_rnn_along/{seq_id}.pt",
            # f"/home/chqian/Project/PIP/PIP/data/result/SIGGRAPH_dataset/PIP/{seq_id}.pt",
            # f"/home/chqian/Project/PIP/PIP/data/result/SIGGRAPH_dataset/PIP_GNN_fusion/{seq_id}.pt",

            f"/home/chqian/Project/PIP/PIP/data/result/SIGGRAPH_dataset/PIP_GNN_fusion/{seq_id}.pt",
            #f"/home/chqian/Project/PIP/PIP/data/result/SIGGRAPH_dataset/PIP_GNN_fusion{seq_id}.pt",
             ]
      
    colors = [(0.627, 0.627, 0.627),(0.3, 0.3, 0.3)]
    smpl_seqs = [];rb_seq = []
    for p,c in zip(paths,colors):
        smpl_s,joint = visualize_smpl_models(p,rgb=c,seq_end=15000,stride=4)
        smpl_seqs.append(smpl_s)
        if show_lj_pos:
            rb_seq.extend(visualize_leaf_joint_position(p,joint,seq_end=15000,stride=4))
        
        
    gt_path = "/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/SIGGRAPH_dataset/test.pt"
    #gt_path = "/home/chqian/Project/PIP/PIP/data/dataset_work/DIP_IMU/test.pt"
    
    smpl_gt,*res = load_uwb_imu_dataset(gt_path,seq_id=seq_id,rgb=(0.1,0.8,0.1),seq_end=15000,stride=4)
    smpl_seqs.append(smpl_gt)
    
    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(*smpl_seqs)
    if rb_seq:
        v.scene.add(*rb_seq)
    v.run()
