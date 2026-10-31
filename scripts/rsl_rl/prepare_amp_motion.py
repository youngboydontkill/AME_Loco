"""Convert GMR-retargeted Kuavo poses into AMP state features."""

import argparse
from pathlib import Path

import numpy as np


def _rotate_inverse(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by inverse wxyz quaternions."""
    xyz = -quaternions[..., 1:]
    w = quaternions[..., :1]
    twice_cross = 2.0 * np.cross(xyz, vectors)
    return vectors + w * twice_cross + np.cross(xyz, twice_cross)


def _differentiate(values: np.ndarray, clip_ids: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    for clip_id in np.unique(clip_ids):
        indices = np.flatnonzero(clip_ids == clip_id)
        if indices.size < 2 or not np.all(np.diff(indices) == 1):
            raise ValueError("Each clip must occupy a contiguous range and contain at least two frames.")
        result[indices] = np.gradient(values[indices], dt, axis=0, edge_order=1)
    return result


def _angular_velocity_body(quaternions: np.ndarray, clip_ids: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros((len(quaternions), 3), dtype=np.float32)
    for clip_id in np.unique(clip_ids):
        indices = np.flatnonzero(clip_ids == clip_id)
        if indices.size < 2 or not np.all(np.diff(indices) == 1):
            raise ValueError("Each clip must occupy a contiguous range and contain at least two frames.")
        for left, right in zip(indices[:-1], indices[1:]):
            q0 = quaternions[left]
            q1 = quaternions[right]
            if np.dot(q0, q1) < 0:
                q1 = -q1
            q0_inv = q0 * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            delta = np.array(
                [
                    q0_inv[0] * q1[0] - np.dot(q0_inv[1:], q1[1:]),
                    *(q0_inv[0] * q1[1:] + q1[0] * q0_inv[1:] + np.cross(q0_inv[1:], q1[1:])),
                ],
                dtype=np.float32,
            )
            delta /= max(np.linalg.norm(delta), 1e-8)
            angle = 2.0 * np.arctan2(np.linalg.norm(delta[1:]), np.clip(delta[0], -1.0, 1.0))
            axis_norm = np.linalg.norm(delta[1:])
            if axis_norm > 1e-8:
                result[left] = delta[1:] / axis_norm * (angle / dt)
        result[indices[-1]] = result[indices[-2]]
    return result


def build_amp_states(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    joint_pos_rel: np.ndarray,
    end_effector_pos_w: np.ndarray,
    clip_ids: np.ndarray,
    fps: float,
) -> np.ndarray:
    frame_count = root_pos_w.shape[0]
    expected = {
        "root_pos_w": (frame_count, 3),
        "root_quat_w": (frame_count, 4),
        "joint_pos_rel": (frame_count, 27),
        "end_effector_pos_w": (frame_count, 4, 3),
        "clip_ids": (frame_count,),
    }
    actual = {
        "root_pos_w": root_pos_w.shape,
        "root_quat_w": root_quat_w.shape,
        "joint_pos_rel": joint_pos_rel.shape,
        "end_effector_pos_w": end_effector_pos_w.shape,
        "clip_ids": clip_ids.shape,
    }
    for key, shape in expected.items():
        if actual[key] != shape:
            raise ValueError(f"{key} must have shape {shape}, got {actual[key]}.")
    if frame_count < 2 or fps <= 0:
        raise ValueError("At least two frames and a positive fps value are required.")
    arrays = (root_pos_w, root_quat_w, joint_pos_rel, end_effector_pos_w)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("GMR motion arrays contain NaN or infinite values.")

    quat_norm = np.linalg.norm(root_quat_w, axis=1, keepdims=True)
    if np.any(quat_norm < 1e-8):
        raise ValueError("root_quat_w contains a zero-length quaternion.")
    root_quat_w = root_quat_w / quat_norm
    dt = 1.0 / float(fps)
    root_lin_vel_w = _differentiate(root_pos_w, clip_ids, dt)
    root_lin_vel_b = _rotate_inverse(root_quat_w, root_lin_vel_w)
    root_ang_vel_b = _angular_velocity_body(root_quat_w, clip_ids, dt)
    gravity_w = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (frame_count, 1))
    projected_gravity_b = _rotate_inverse(root_quat_w, gravity_w)

    end_effector_relative_w = end_effector_pos_w - root_pos_w[:, None, :]
    root_quat_ee = np.broadcast_to(root_quat_w[:, None, :], (frame_count, 4, 4))
    end_effector_pos_b = _rotate_inverse(root_quat_ee, end_effector_relative_w)
    end_effector_vel_b = _differentiate(end_effector_pos_b, clip_ids, dt)
    joint_vel = _differentiate(joint_pos_rel, clip_ids, dt)

    features = (
        root_lin_vel_b,
        root_ang_vel_b,
        projected_gravity_b,
        joint_pos_rel,
        joint_vel,
        end_effector_pos_b.reshape(frame_count, 12),
        end_effector_vel_b.reshape(frame_count, 12),
    )
    states = np.concatenate(features, axis=1).astype(np.float32, copy=False)
    if states.shape[1] != 87:
        raise RuntimeError(f"Internal AMP feature size should be 87, got {states.shape[1]}.")
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="GMR NPZ containing the arrays documented in doc/amp_motion_data.md")
    parser.add_argument("output", help="Output NPZ path for the AMP expert loader")
    args = parser.parse_args()

    with np.load(args.input) as source:
        required = ("root_pos_w", "root_quat_w", "joint_pos_rel", "end_effector_pos_w", "clip_ids", "fps")
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"GMR motion archive is missing required arrays: {', '.join(missing)}")
        states = build_amp_states(
            np.asarray(source["root_pos_w"], dtype=np.float32),
            np.asarray(source["root_quat_w"], dtype=np.float32),
            np.asarray(source["joint_pos_rel"], dtype=np.float32),
            np.asarray(source["end_effector_pos_w"], dtype=np.float32),
            np.asarray(source["clip_ids"], dtype=np.int64),
            float(np.asarray(source["fps"]).item()),
        )
        clip_ids = np.asarray(source["clip_ids"], dtype=np.int64)

    Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, states=states, clip_ids=clip_ids)
    print(f"Wrote {len(states)} AMP states with {states.shape[1]} features to {args.output}")


if __name__ == "__main__":
    main()
