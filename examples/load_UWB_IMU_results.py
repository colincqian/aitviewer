# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl

import torch
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # This is loading the DIP-IMU data that can be downloaded from the DIP project website here:
    # https://dip.is.tue.mpg.de/download.php
    # Download the "DIP IMU and others" and point the following path to one of the extracted pickle files.
    # with open(r"/home/chqian/Project/PIP/PIP/data/dataset_work/UWB_IMU/Dataset_B/test.pt", "rb") as f:
    #     data = pkl.load(f, encoding="latin1")
    
    #path = "/home/chqian/Project/PIP/PIP/data/result/Dataset_B/PIP/0.pt"
    
    path = "/home/chqian/Project/PIP/PIP/data/result/Dataset_B_not_align/PIP/0.pt"
    
    data = torch.load(path)

    # Whether we want to visualize all 17 sensors or just the 6 sensors used by DIP.
    all_sensors = False
    uwb_imu_rot = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1, 0]])
    
    # Get the data.
    poses = matrix_to_axis_angle(data[0]).view(-1,72)
    tran = data[1].view(-1,3)
    
    
    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[::2]
    tran = tran[::2]

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

    # This is the sensor placement (cf. https://github.com/eth-ait/dip18/issues/16).
    sensor_placement = [
        "head",
        "sternum",
        "pelvis",
        "lshoulder",
        "rshoulder",
        "lupperarm",
        "rupperarm",
        "llowerarm",
        "rlowerarm",
        "lupperleg",
        "rupperleg",
        "llowerleg",
        "rlowerleg",
        "lhand",
        "rhand",
        "lfoot",
        "rfoot",
    ]


    #rbs_v = RigidBodies(joints[:, joint_idxs].cpu().numpy(), v_oris,color=(0,0,0,1))

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3],trans=tran)
    smpl_seq.mesh_seq.color = smpl_seq.mesh_seq.color[:3] + (0.5,)

    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(smpl_seq)
    v.run()
