import os
import math
import imageio
import numpy as np

from datetime import datetime

RENDER_ENABLED = True 


# Date/time utilities
DATE = datetime.now().strftime("%Y-%m-%d")
DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from LIBERO.libero.libero import get_libero_path
    
    
    
    if RENDER_ENABLED:
        from LIBERO.libero.libero.envs import OffScreenRenderEnv
    else:
        class OffScreenRenderEnv:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Rendering is disabled and no headless mode is defined.")



    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_image(obs):
    """Extracts third-person image from observations and rotates it."""
    img = obs["agentview_image"]
    return img[::-1, ::-1]  # Flip vertically and horizontally


def get_libero_wrist_image(obs):
    """Extracts wrist camera image and rotates it."""
    img = obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1]


def quat2axisangle(quat):
    """Converts quaternion to axis-angle format."""
    quat=quat.flatten()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves a video of rollout images to ./rollouts/<DATE>/...mp4."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"

    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    return mp4_path
