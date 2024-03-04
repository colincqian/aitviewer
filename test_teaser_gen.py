from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.arrows import Arrows

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
import numpy as np
import torch

from pytorch3d.transforms import quaternion_multiply,matrix_to_quaternion,quaternion_to_axis_angle,quaternion_to_matrix,matrix_to_axis_angle
from aitviewer.utils import local_to_global
from aitviewer.utils import to_numpy as c2c
def rotation_swing_twist_decomposition(q: torch.Tensor,
                                       n_twist: torch.Tensor):
    _n_twist= n_twist[:,[1,2,0]]
    u = torch.sum(q[:,[2,3,1]] * _n_twist)
    n = torch.sum(torch.square(_n_twist),dim=-1)
    m = q[:,0] * n
    l = torch.sqrt(torch.square(m) + torch.square(u) * n)
    
    w,q1,q2,q3 = m/l,_n_twist[:,2]*u/l,_n_twist[:,0]*u/l,_n_twist[:,1]*u/l
    q_twist = torch.cat([w,q1,q2,q3],dim=-1).view(-1,4)
    q_swing = quaternion_multiply(q,torch.cat([w,-q1,-q2,-q3],dim=-1)).view(-1,4)
    
    q_recon = quaternion_multiply(q_swing, q_twist).view(-1,4)
    
    return q_swing,q_twist


if __name__ == '__main__':
    
    
    #file = "../export/CL_2_output.npz"
    #file = "../export/RA_output.npz"
    file = "/home/chqian/Project/ROMP/cht_1_results.npz"
    amass_file = "/home/chqian/Project/PIP/PIP/data/AMASS_raw/AMASS_CMU/01/01_01_poses.npz"
    results = np.load(file,allow_pickle=True)['results'][()]
    amass_body_data = np.load(amass_file)["poses"][[5]]
    #results["body_pose"][:,22*3:23*3] = amass_body_data[:,37*3:38*3]
    amass_body_data[:,3:66] = results["body_pose"][:,:-6]
    smpl_layer = SMPLLayer(model_type="smplh", gender="male")

    #21 lw;22 rw;19 le;20 rl
    i_root_end = 3
    i_body_end = i_root_end + smpl_layer.bm.NUM_BODY_JOINTS * 3
    i_left_hand_end = i_body_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
    i_right_hand_end = i_left_hand_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
    print(i_left_hand_end,i_right_hand_end)
    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    
    _, joints = smpl_layer(
            poses_body=torch.from_numpy(amass_body_data[:,i_root_end:i_body_end]).float(),
            poses_left_hand=torch.from_numpy(amass_body_data[:, i_body_end:i_left_hand_end]).float(),
            poses_right_hand=torch.from_numpy(amass_body_data[:, i_left_hand_end:i_right_hand_end]).float(),
            poses_root=torch.from_numpy(results["global_orient"]),
            betas=torch.from_numpy(results["smpl_betas"]),
            trans=torch.from_numpy(results["cam_trans"]),
    )
    

    poses = torch.from_numpy(amass_body_data[:,i_root_end:i_body_end]).float()
    l_pose = torch.from_numpy(amass_body_data[:, i_body_end:i_left_hand_end]).float()
    r_pose = torch.from_numpy(amass_body_data[:, i_left_hand_end:i_right_hand_end]).float()
    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(
                            poses_body=poses,
                            poses_left_hand=l_pose,
                            poses_right_hand=r_pose,
                            smpl_layer=smpl_layer, 
                            poses_root=torch.zeros_like(torch.from_numpy(results["global_orient"])),
                            trans=torch.from_numpy(results["cam_trans"]))
    
    
    poses_local = torch.cat([torch.from_numpy(results["global_orient"]), torch.from_numpy(amass_body_data[:,i_root_end:i_body_end]).float()], dim=-1)
    global_oris = local_to_global(
        poses_local,
        smpl_seq.skeleton[:, 0],
        output_format="rotmat",
    )
    global_oris = c2c(global_oris.reshape((1, -1, 3, 3)))
    
    wrist_pos = torch.from_numpy(smpl_seq.joints[:,20]).float()
    elbow_pos = torch.from_numpy(smpl_seq.joints[:,18]).float()
    n_twist = torch.nn.functional.normalize((elbow_pos - wrist_pos),dim=1)
    
    wrist_rot = torch.from_numpy(global_oris[:,20]).float()
    elbow_rot = torch.from_numpy(global_oris[:,18]).float()
    shoulder_rot = torch.from_numpy(global_oris[:,16]).float()
    tar_rot = torch.eye(3).unsqueeze(0)
    
    q_ = matrix_to_quaternion(wrist_rot.permute(0,2,1))
    q_swing, q_twist = rotation_swing_twist_decomposition(q_,n_twist)
    res_rot = quaternion_to_matrix(q_swing).view(1,3,3) @ quaternion_to_matrix(q_twist).view(1,3,3) @ wrist_rot
    print(res_rot)
    
    #update twist first
    elbow_rot_g = quaternion_to_matrix(q_twist).view(1,3,3) @ elbow_rot
    elbow_rot_l = shoulder_rot.view(3,3).T @ elbow_rot_g
    poses[:,17*3: 18*3] = matrix_to_axis_angle(elbow_rot_l)
    print(torch.nn.functional.normalize(quaternion_to_axis_angle(q_twist))/n_twist)
    arr_head = Arrows(origins=wrist_pos.numpy(), tips=wrist_pos.numpy() + n_twist.numpy() ,color=(0,0,0.5,1))#b
    #poses[:,17*3] = poses[:,17*3] - 10
    #poses[:,19*3] = poses[:,19*3] - alpha
    
    
    #update swing 
    # wrist_rot_l = elbow_rot_g.view(3,3).T @ quaternion_to_matrix(q_swing).view(3,3) @ elbow_rot_g.view(3,3)
    # axis_angle_local = matrix_to_axis_angle(wrist_rot_l)
    # poses[:,19*3:20*3] = axis_angle_local
    
    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq_new = SMPLSequence(
                            poses_body=poses,
                            poses_left_hand=torch.from_numpy(amass_body_data[:, i_body_end:i_left_hand_end]).float(),
                            poses_right_hand=torch.from_numpy(amass_body_data[:, i_left_hand_end:i_right_hand_end]).float(),
                            smpl_layer=smpl_layer, 
                            poses_root=torch.zeros_like(torch.from_numpy(results["global_orient"])),
                            trans=torch.from_numpy(results["cam_trans"]))
    
    v = Viewer()
    v.scene.add(smpl_seq,smpl_seq_new,arr_head)
    v.run()