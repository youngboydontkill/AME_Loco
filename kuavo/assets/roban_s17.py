"""Isaac Lab articulation configuration for RobanS17."""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import ISAAC_ASSET_DIR
from .roban_s17_actuators import RobanS17DelayedPDActuatorCfg


ROBAN_S17_ASSET_DIR = f"{ISAAC_ASSET_DIR}/robanS17"
ROBAN_S17_URDF = f"{ROBAN_S17_ASSET_DIR}/urdf/biped_s17.urdf"


ROBAN_S17_MDP_JOINT_ORDER_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "waist_yaw_joint",
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
    ],
    body_names=[
        "waist_yaw_link",
        "leg_l1_link", "leg_l2_link", "leg_l3_link", "leg_l4_link", "leg_l5_link", "leg_l6_link",
        "leg_r1_link", "leg_r2_link", "leg_r3_link", "leg_r4_link", "leg_r5_link", "leg_r6_link",
        "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link",
        "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link",
    ],
    preserve_order=True,
)


@configclass
class RobanS17ArticulationCfg(ArticulationCfg):
    """RobanS17 S2.2 articulation and Leju actuator parameters."""

    spawn = sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        asset_path=ROBAN_S17_URDF,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_[lr][1-6]_joint": 0.0,
            "zarm_[lr][1-4]_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor = 0.95
    actuators = {
        "motor": RobanS17DelayedPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "leg_.*", "zarm_.*"],
            effort_limit_sim={
                "waist_yaw_joint": 80.0,
                "leg_[lr]1_joint": 150.0, "leg_[lr]2_joint": 150.0, "leg_[lr]3_joint": 70.0,
                "leg_[lr]4_joint": 150.0, "leg_[lr]5_joint": 74.0, "leg_[lr]6_joint": 74.0,
                "zarm_[lr]1_joint": 14.1, "zarm_[lr][2-4]_joint": 37.0,
            },
            effort_limit_rated={
                "waist_yaw_joint": 35.0,
                "leg_[lr]1_joint": 51.0, "leg_[lr]2_joint": 51.0, "leg_[lr]3_joint": 34.0,
                "leg_[lr]4_joint": 51.0, "leg_[lr]5_joint": 37.0, "leg_[lr]6_joint": 37.0,
                "zarm_[lr]1_joint": 6.1, "zarm_[lr][2-4]_joint": 11.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 12.0,
                "leg_[lr]1_joint": 14.6, "leg_[lr]2_joint": 14.6, "leg_[lr]3_joint": 12.0,
                "leg_[lr]4_joint": 14.6, "leg_[lr][56]_joint": 17.0,
                "zarm_[lr]1_joint": 10.5, "zarm_[lr][2-4]_joint": 15.0,
            },
            stiffness={
                "waist_yaw_joint": 40.1792,
                "leg_[lr]1_joint": 90.1792, "leg_[lr]2_joint": 150.0984, "leg_[lr]3_joint": 40.1792,
                "leg_[lr]4_joint": 150.0984, "leg_[lr][56]_joint": 34.2506,
                "zarm_[lr][1-4]_joint": 14.2506,
            },
            damping={
                "waist_yaw_joint": 3.5579, "leg_[lr]1_joint": 3.5579, "leg_[lr]2_joint": 8.3088,
                "leg_[lr]3_joint": 3.5579, "leg_[lr]4_joint": 8.3088, "leg_[lr][56]_joint": 2.9072,
                "zarm_[lr][1-4]_joint": 1.9072,
            },
            armature={
                "waist_yaw_joint": 0.03178707, "leg_[lr]1_joint": 0.0493, "leg_[lr]2_joint": 0.0493,
                "leg_[lr]3_joint": 0.03178707, "leg_[lr]4_joint": 0.0493, "leg_[lr][56]_joint": 0.036,
                "zarm_[lr]1_joint": 0.015, "zarm_[lr][2-4]_joint": 0.018,
            },
            friction=0.0,
            min_delay=0,
            max_delay=4,
            friction_static=0.2,
            friction_activation_vel=0.1,
            friction_dynamic=0.0,
        ),
    }
    preserve_joint_order = ROBAN_S17_MDP_JOINT_ORDER_CFG
    end_effector_configs = [
        ("leg_l6_link", None),
        ("leg_r6_link", None),
        ("zarm_l4_link", [0.0, 0.0, -0.2]),
        ("zarm_r4_link", [0.0, 0.0, -0.2]),
    ]


ROBAN_S17_CFG = RobanS17ArticulationCfg()
