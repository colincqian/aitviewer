# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl

import torch
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer
import pickle 

def load_processed_uwb_imu_pkl(file_path,rgb,seq_end=-1,stride=2):
    data = pickle.load(open(file_path,'rb'),encoding='latin1')
    poses = torch.from_numpy(data["pose"]).float().view(-1,72)
    tran = torch.from_numpy(data["trans"]).float().view(-1,3)
    oris = data["imu_ori"].reshape(-1,6,3,3)
    accs = data["imu_acc"].reshape(-1,6,3)

    # Downsample to 30 Hz.
    poses = poses[:seq_end:stride]
    # import ipdb
    # ipdb.set_trace()
    oris = oris[:seq_end:stride]
    tran = tran[:seq_end:stride]
    accs = accs[:seq_end:stride]

    # DIP has no shape information, assume the mean shape.
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)
    poses[:,20*3:22*3] = 0 #hard code hand pose

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].to(C.device),
        poses_root=poses[:, :3].to(C.device),
        betas=betas,
        trans=tran.to(C.device)
    )
    

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3],trans=tran)
    smpl_seq.mesh_seq.color = rgb + (0.5,)
    
    return smpl_seq,joints,oris,accs,None

def load_uwb_imu_dataset(path,seq_id,rgb,seq_end=-1,stride=2):
    data = torch.load(path)

    # Get the data.
    poses = data["pose"][seq_id].view(-1,72)
    tran = data["tran"][seq_id].view(-1,3)
    oris = data["ori"][seq_id].numpy().reshape(-1,6,3,3)
    accs = data["acc"][seq_id].numpy().reshape(-1,6,3)
    if "vuwb" in data:
        uwb = data["vuwb"][seq_id].numpy().reshape(-1,6,6)
        uwb = uwb[:seq_end:2]
    else:
        uwb = None

    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[:seq_end:stride]
    # import ipdb
    # ipdb.set_trace()
    oris = oris[:seq_end:stride]
    tran = tran[:seq_end:stride]
    accs = accs[:seq_end:stride]

    # DIP has no shape information, assume the mean shape.
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)
    poses[:,20*3:22*3] = 0 #hard code hand pose

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].to(C.device),
        poses_root=poses[:, :3].to(C.device),
        betas=betas,
        trans=tran.to(C.device)
    )
    

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3],trans=tran)
    smpl_seq.mesh_seq.color = rgb + (0.5,)
    
    return smpl_seq,joints,oris,accs,uwb
    
if __name__ == "__main__":
    # This is loading the DIP-IMU data that can be downloaded from the DIP project website here:
    # https://dip.is.tue.mpg.de/download.php
    # Download the "DIP IMU and others" and point the following path to one of the extracted pickle files.
    # with open(r"/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/Dataset_B/test.pt", "rb") as f:
    #     data = pkl.load(f, encoding="latin1")
    
    path = "/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/Dataset_B/test.pt"
    #path = "/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/20231124_marker_label_tests/test.pt"
    path = "/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/SIGGRAPH_dataset/test.pt"
    path = "/home/chqian/Project/PIP/PIP/data/dataset_work/TotalCapture/[author_direct_ver]test.pt"
    smpl_seq,joints,oris,accs,uwb = load_uwb_imu_dataset(path,seq_id=7,rgb=(0.6,0.6,0.6),seq_end=7200+3600,stride=4)
    
    # pkl_file_path = "/home/chqian/DATA/SIGGRAPH_dataset/subject_4/04_session2_0/processed_04_session2_0.pkl"
    # smpl_seq,joints,oris,accs,uwb = load_processed_uwb_imu_pkl(pkl_file_path,rgb=(0.6,0.6,0.6),seq_end=7200+3600,stride=4)
    # We manually choose the SMPL joint indices cooresponding to the above sensor placement.
    joint_idxs = [20, 21, 4, 5, 15, 0]
    
    # Select only the 6 input sensors if configured.
    rbs = RigidBodies(joints[:, joint_idxs].cpu().numpy(), oris)

    arr = Arrows(origins=joints[:, joint_idxs].cpu().numpy(),tips = joints[:, joint_idxs].cpu().numpy() - accs * 1/30)

    # jp = joints[:, joint_idxs]
    # dist_m = torch.cdist(jp,jp)
    # import ipdb
    # ipdb.set_trace()
    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(smpl_seq, rbs, arr)
    v.run()
