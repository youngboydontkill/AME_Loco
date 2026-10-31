# Kuavo S54 AMP Motion Data

For GMR, export retargeted pose trajectories as an `.npz` archive. You do not need to calculate velocities or convert world coordinates to body coordinates; `prepare_amp_motion.py` does that. The input archive must contain:

| Array | Shape | Meaning |
| --- | --- | --- |
| `fps` | scalar | Sampling frequency in Hz; use 50 Hz or resample to it |
| `root_pos_w` | `[frames, 3]` | Kuavo `base_link` position in world coordinates, meters |
| `root_quat_w` | `[frames, 4]` | `base_link` orientation in Isaac/GMR `w, x, y, z` quaternion order |
| `joint_pos_rel` | `[frames, 27]` | Retargeted S54 joint angles relative to the S54 default pose, radians |
| `end_effector_pos_w` | `[frames, 4, 3]` | World positions in meters, ordered left foot, right foot, left hand/wrist, right hand/wrist |
| `clip_ids` | `[frames]` | Integer clip ID per frame; each clip must be contiguous and have at least two frames |

The 27 joint columns must follow the loaded asset's `robot.data.joint_names` order. Include both legs, waist yaw, and both 7-DOF arms. The default pose is `leg_[lr]1=0`, `leg_[lr]2=0`, `leg_[lr]3=-0.4`, `leg_[lr]4=0.69`, `leg_[lr]5=-0.33`, `leg_[lr]6=0`, with waist yaw and all arm joints at zero; subtract these per named joint to produce `joint_pos_rel`. For end effectors, use the robot links `leg_l6_link`, `leg_r6_link`, `zarm_l7_link`, and `zarm_r7_link` in the listed order. If GMR only exports joint poses, compute the four endpoint positions with forward kinematics using the Kuavo S54 URDF before exporting. Do not pass human-body keypoints directly as robot endpoint positions after retargeting.

Example packing call:

```python
np.savez_compressed(
    "gmr_kuavo_s54_walk_raw.npz",
    fps=np.float32(50.0),
    root_pos_w=root_positions,
    root_quat_w=root_quaternions_wxyz,
    joint_pos_rel=joint_angles_minus_s54_default,
    end_effector_pos_w=kuavo_foot_and_wrist_positions,
    clip_ids=clip_ids,
)
```

Convert the GMR archive to the 87-dimensional AMP state representation:

```bash
python scripts/rsl_rl/prepare_amp_motion.py gmr_kuavo_s54_walk_raw.npz data/amp/kuavo_s54_walk.npz
```

The resulting file contains `states [frames, 87]` and `clip_ids [frames]`. State features are ordered as root linear velocity (3), root angular velocity (3), projected gravity (3), relative joint angles (27), joint velocities (27), body-frame foot/wrist positions (12), and body-frame foot/wrist velocities (12). AMP trains on consecutive `(state_t, state_t+1)` pairs within each clip. It does not compare absolute world position.

Set `AME_AMP_MOTION_FILE` to the packed file path when launching training. The default is `data/amp/kuavo_s54_walk.npz`.

Launch the new task with:

```bash
AME_AMP_MOTION_FILE=data/amp/kuavo_s54_walk.npz \
  ./isaac_lab23/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task AME-Kuavo-S54-Depth-Enhance-AMP-v0 --headless
```

AMASS is a reasonable source for research gait clips because it consolidates optical motion-capture sequences and provides SMPL body parameters. Its official site requires registration to download. The dataset license permits non-commercial scientific research, education, or artistic use only; it prohibits commercial use and redistribution. Confirm the intended use fits the license before downloading or training. AMASS does not provide Kuavo-ready joint trajectories, so first retarget selected clips through GMR. HumanML3D includes walking labels but is derived from AMASS/HumanAct12 and uses a processed representation; it still needs reconstruction and retargeting, and does not remove upstream license obligations.
