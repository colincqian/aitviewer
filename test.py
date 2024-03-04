from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
import numpy as np
import torch

# smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
# poses = np.zeros([1, 69])
# smpl_seq = SMPLSequence(poses, smpl_layer)
# smpl_seq.from_npz_romp("../export/CL_2_output.npz")

if __name__ == '__main__':
    #file = "../export/CL_2_output.npz"
    file = "export/RA_output.npz"
    results = np.load(file,allow_pickle=True)['results'][()]
    smpl_layer = SMPLLayer(model_type="smpl", gender="female")

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
            poses_body=torch.from_numpy(results["body_pose"]),
            poses_root=torch.from_numpy(results["global_orient"]),
            betas=torch.from_numpy(results["smpl_betas"]),
            trans=torch.from_numpy(results["cam_trans"]),
    )
    
    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=torch.from_numpy(results["body_pose"]),
                            smpl_layer=smpl_layer, 
                            poses_root=torch.zeros_like(torch.from_numpy(results["global_orient"])),
                            trans=torch.from_numpy(results["cam_trans"]))
    v = Viewer()
    v.scene.add(smpl_seq)
    v.run()