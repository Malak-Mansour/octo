# ADAPTED FROM OCTO
# '''
# python3 examples/03_eval_finetuned_openvla_data.py \
#   --finetuned_path=examples/libero/libero_object/libero_object_no_noops/finetuned_chkpt \
#   --data_dir=examples/libero/libero_object
# '''


# from absl import app, flags, logging
# import jax
# import numpy as np
# import wandb
# import os

# from octo.data.dataset import make_single_dataset
# from octo.model.octo_model import OctoModel
# from octo.utils.train_utils import process_text


# from functools import wraps

# def supply_rng(fn):
#     """Wraps a function to automatically supply a new RNG each call."""
#     key = jax.random.PRNGKey(0)

#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         nonlocal key
#         key, subkey = jax.random.split(key)
#         return fn(*args, rng=subkey, **kwargs)

#     return wrapper





# FLAGS = flags.FLAGS

# flags.DEFINE_string("finetuned_path", None, "Path to finetuned Octo checkpoint directory.")
# flags.DEFINE_string("data_dir", None, "Path to dataset in RLDS format.")
# flags.DEFINE_integer("num_rollouts", 3, "Number of rollouts to run.")
# flags.DEFINE_integer("action_horizon", 50, "Action prediction horizon.")

# def main(_):
#     wandb.init(name="eval_libero", project="octo")

#     logging.info("Loading finetuned model...")
#     model = OctoModel.load_pretrained(FLAGS.finetuned_path)

#     logging.info("Loading evaluation dataset...")
#     dataset = make_single_dataset(
#         dataset_kwargs=dict(
#             name="libero_object_no_noops",
#             data_dir=FLAGS.data_dir,
#             image_obs_keys={
#                 "rgb": "image",
#                 "wrist": "wrist_image"
#             },
#             proprio_obs_key="state",
#             language_key="language_instruction",
#         ),
#         traj_transform_kwargs=dict(
#             window_size=1,
#             action_horizon=FLAGS.action_horizon,
#         ),
#         frame_transform_kwargs=dict(
#             resize_size={"primary": (256, 256)},
#         ),
#         train=False,
#     )

#     dataset_iter = (
#         dataset
#         .unbatch()
#         .batch(1)
#         .repeat()
#         .iterator()
#     )

#     # Setup policy
#     policy_fn = supply_rng(
#         lambda obs, task, rng: model.sample_actions(
#             obs,
#             task,
#             rng=rng, #malak: added this
#             unnormalization_statistics=model.dataset_statistics["action"]
#         )
#     )


#     text_processor = model.text_processor

#     for rollout_idx in range(FLAGS.num_rollouts):
#         batch = next(dataset_iter)
#         batch = process_text(batch, text_processor)

#         observation = batch["observation"]
#         language_instruction = batch["task"]["language_instruction"]
#         task = model.create_tasks(language_instruction)

#         # Simulate rollout
#         predicted_actions = policy_fn(observation, task)
#         print(f"[{rollout_idx}] Predicted actions shape: {predicted_actions.shape}")


#         instruction_str = language_instruction.tolist()[0] if hasattr(language_instruction, "tolist") else str(language_instruction)
#         wandb.log({
#             f"rollout_{rollout_idx}_actions": wandb.Histogram(np.array(predicted_actions)),
#             f"rollout_{rollout_idx}_lang_instruction": instruction_str,
#         })

#         # wandb.log({
#         #     f"rollout_{rollout_idx}_actions": wandb.Histogram(np.array(predicted_actions)),
#         #     f"rollout_{rollout_idx}_lang_instruction": language_instruction[0],
#         # })

# if __name__ == "__main__":
#     app.run(main)



# ADAPTED FROM OPENVLA, DOESNT WORK
# """
# run_libero_eval_octo.py

# Evaluates a finetuned Octo model on LIBERO benchmark tasks.

# python3 03_eval_finetuned_openvla_data.py --finetuned_path=libero/libero_object/libero_object_no_noops/finetuned_chkpt

# """

# import os
# import json
# import tqdm
# import wandb
# import numpy as np
# from functools import partial
# from dataclasses import dataclass
# from pathlib import Path
# from absl import app, flags, logging
# from collections import deque

# from libero.libero import benchmark
# from octo.model.octo_model import OctoModel
# from octo.utils.train_callbacks import supply_rng
# from octo.utils.jax_utils import initialize_compilation_cache
# from experiments.robot.libero.libero_utils import (
#     get_libero_env,
#     get_libero_image,
#     get_libero_wrist_image,
#     quat2axisangle,
#     save_rollout_video,
# )
# from experiments.robot.robot_utils import (
#     set_seed_everywhere,
#     normalize_gripper_action,
# )

# FLAGS = flags.FLAGS

# TASK_MAX_STEPS = {
#     "libero_spatial": 220,
#     "libero_object": 280,
#     "libero_goal": 300,
#     "libero_10": 520,
#     "libero_90": 400,
# }

# @dataclass
# class Config:
#     pretrained_checkpoint: str
#     task_suite_name: str = "libero_object"
#     num_trials_per_task: int = 50
#     num_steps_wait: int = 10
#     env_img_res: int = 256
#     local_log_dir: str = "./experiments/logs"
#     use_wandb: bool = False
#     seed: int = 42


# def prepare_observation(obs):
#     image = get_libero_image(obs)
#     wrist_image = get_libero_wrist_image(obs)
#     state = np.concatenate([
#         obs["robot0_eef_pos"],
#         quat2axisangle(obs["robot0_eef_quat"]),
#         obs["robot0_gripper_qpos"],
#     ])
#     obs_dict = {
#         "image": image,
#         "wrist_image": wrist_image,
#         "state": state.astype(np.float32),
#         "timestep_pad_mask": np.array([1.0], dtype=np.float32)
#     }
#     return obs_dict, image


# def run_episode(cfg, env, model, task_text, stats, initial_state=None):
#     env.reset()
#     if initial_state is not None:
#         obs = env.set_init_state(initial_state)
#     else:
#         obs = env.get_observation()

#     obs_dict, img = prepare_observation(obs)
#     task_dict = model.create_tasks(texts=[task_text])
#     policy_fn = supply_rng(
#         partial(model.sample_actions, unnormalization_statistics=stats["action"])
#     )

#     replay_images = [img]
#     max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
#     t, success = 0, False

#     while t < max_steps + cfg.num_steps_wait:
#         if t < cfg.num_steps_wait:
#             obs, _, done, _ = env.step(np.zeros(7))  # dummy
#             t += 1
#             continue

#         actions = policy_fn({k: v[None] for k, v in obs_dict.items()}, task_dict)[0]
#         actions = normalize_gripper_action(actions, binarize=True)
#         obs, reward, done, info = env.step(actions.tolist())

#         obs_dict, img = prepare_observation(obs)
#         replay_images.append(img)
#         if done:
#             success = True
#             break
#         t += 1

#     return success, replay_images


# def run_task(cfg, model, stats, task_suite, task_id, log_file):
#     task = task_suite.get_task(task_id)
#     env, task_desc = get_libero_env(task, "octo", resolution=cfg.env_img_res)
#     init_states = task_suite.get_task_init_states(task_id)

#     success_count = 0
#     for i in tqdm.trange(cfg.num_trials_per_task, desc=f"Task {task_id}"):
#         init_state = init_states[i]
#         success, replay_images = run_episode(cfg, env, model, task_desc, stats, init_state)
#         success_count += int(success)

#         save_rollout_video(
#             replay_images,
#             step=(task_id * cfg.num_trials_per_task + i),
#             success=success,
#             task_description=task_desc,
#             log_file=log_file,
#         )

#     return success_count, cfg.num_trials_per_task


# def main(_):
#     cfg = Config(
#         pretrained_checkpoint=FLAGS.finetuned_path,
#     )

#     set_seed_everywhere(cfg.seed)
#     os.makedirs(cfg.local_log_dir, exist_ok=True)

#     log_file_path = os.path.join(cfg.local_log_dir, f"eval_{cfg.task_suite_name}.log")
#     log_file = open(log_file_path, "w")

#     if cfg.use_wandb:
#         wandb.init(project="octo", name=f"eval-{cfg.task_suite_name}")

#     logging.info("Loading model...")
#     initialize_compilation_cache()
#     model = OctoModel.load_pretrained(cfg.pretrained_checkpoint)

#     benchmark_dict = benchmark.get_benchmark_dict()
#     task_suite = benchmark_dict[cfg.task_suite_name]()
#     num_tasks = task_suite.n_tasks

#     total_success, total_trials = 0, 0
#     for task_id in range(num_tasks):
#         s, t = run_task(cfg, model, model.dataset_statistics, task_suite, task_id, log_file)
#         total_success += s
#         total_trials += t
#         logging.info(f"Task {task_id}: {s}/{t} ({100. * s / t:.1f}%)")

#     logging.info(f"Overall: {total_success}/{total_trials} ({100. * total_success / total_trials:.1f}%)")
#     if cfg.use_wandb:
#         wandb.log({"total_success_rate": total_success / total_trials})
#         wandb.finish()
#     log_file.close()


# if __name__ == "__main__":
#     flags.DEFINE_string("finetuned_path", None, "Path to Octo finetuned checkpoint dir")
#     app.run(main)





# OFFICIAL BENCHMARK
# from absl import app, flags, logging
# import os
# import jax
# import numpy as np
# import wandb
# import imageio

# from functools import wraps
# from octo.data.dataset import make_single_dataset
# from octo.model.octo_model import OctoModel
# from octo.utils.train_utils import process_text

# FLAGS = flags.FLAGS

# flags.DEFINE_string("finetuned_path", None, "Path to finetuned Octo checkpoint directory.")
# flags.DEFINE_string("data_dir", None, "Path to dataset in RLDS format.")
# flags.DEFINE_integer("num_episodes", 10, "Number of episodes to evaluate.")
# flags.DEFINE_integer("action_horizon", 50, "Action prediction horizon.")
# flags.DEFINE_string("log_dir", "eval_logs", "Directory to save rollout videos and logs.")

# def supply_rng(fn):
#     key = jax.random.PRNGKey(0)
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         nonlocal key
#         key, subkey = jax.random.split(key)
#         return fn(*args, rng=subkey, **kwargs)
#     return wrapper

# def log_video(images, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     imageio.mimsave(path, images, fps=10)

# def main(_):
#     wandb.init(name="octo_libero_eval", project="octo", mode="offline")
#     logging.info("Loading finetuned model...")
#     model = OctoModel.load_pretrained(FLAGS.finetuned_path)

#     logging.info("Loading evaluation dataset...")
#     dataset = make_single_dataset(
#         dataset_kwargs=dict(
#             name="libero_object_no_noops",
#             data_dir=FLAGS.data_dir,
#             image_obs_keys={"rgb": "image", "wrist": "wrist_image"},
#             proprio_obs_key="state",
#             language_key="language_instruction",
#         ),
#         traj_transform_kwargs=dict(
#             window_size=1,
#             action_horizon=FLAGS.action_horizon,
#         ),
#         frame_transform_kwargs=dict(
#             resize_size={"primary": (256, 256)},
#         ),
#         train=False,
#     )

#     dataset_iter = dataset.unbatch().batch(1).repeat().iterator()
#     policy_fn = supply_rng(lambda obs, task, rng: model.sample_actions(
#         obs,
#         task,
#         unnormalization_statistics=model.dataset_statistics["action"],
#         rng=rng,
#     ))

#     text_processor = model.text_processor

#     total_successes = 0
#     for ep in range(FLAGS.num_episodes):
#         batch = next(dataset_iter)
#         batch = process_text(batch, text_processor)
#         observation = batch["observation"]
#         language_instruction = batch["task"]["language_instruction"]
#         task = model.create_tasks(language_instruction)

#         actions = policy_fn(observation, task)
#         print(f"[{ep}] Predicted actions shape: {actions.shape}")

#         # Dummy success condition (replace with env-based eval)
#         success = np.random.rand() > 0.5
#         total_successes += int(success)

#         # Dummy visualization: convert action trajectory into grayscale images for now
#         images = [np.uint8(np.clip(np.expand_dims(actions[0, t, :3], 0), 0, 1) * 255).repeat(64, axis=0).repeat(64, axis=1)
#                   for t in range(actions.shape[1])]
#         video_path = os.path.join(FLAGS.log_dir, f"rollout_{ep}_{'success' if success else 'fail'}.mp4")
#         log_video(images, video_path)

#         wandb.log({
#             f"episode_{ep}/success": success,
#             f"episode_{ep}/language_instruction": language_instruction[0].decode("utf-8"),
#         })

#     success_rate = total_successes / FLAGS.num_episodes
#     print(f"✅ Success Rate: {success_rate:.2f}")
#     wandb.log({"overall_success_rate": success_rate})

# if __name__ == "__main__":
#     app.run(main)



# ADAPTED FROM BOTH OPENVLA AND OCTO
# python 03_eval_finetuned_openvla_data.py 
# (inside conda env “octo”)

# export MUJOCO_GL=glx        # or egl once driver fixed
# export WANDB_MODE=offline   # keep as you did, or login for online
# export JAX_PLATFORM_NAME=cpu  # unless you have matching cuDNN

# python examples/03_eval_finetuned_openvla_data.py


RENDER_ENABLED = True  # 👈 Set to False to disable EGL/OpenGL rendering on HPC

import os
import sys
import numpy as np
import gym
import tqdm
import wandb
from collections import deque
import sys
# sys.path.append("/home/malak.mansour/Downloads/ICL/LIBERO")  # adjust to exact LIBERO root
# from libero.libero.benchmark import benchmark
# sys.path.append("/home/malak.mansour/Downloads/ICL/LIBERO/libero")
from LIBERO.libero.libero.benchmark import get_benchmark_dict

# sys.path.append("envs/act")  # optional, for aloha_sim compatibility

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import NormalizeProprio, HistoryWrapper, RHCWrapper
from octo.utils.robosuite_gym_adapter import RobosuiteGymAdapter
from octo.utils.train_callbacks import supply_rng
from libero_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)


import jax.numpy as jnp

TASK_SUITE_NAME = "libero_object"
MAX_STEPS = 280  # from run_libero_eval.py
TRIALS_PER_TASK = 5  # you can increase later

CHECKPOINT_PATH = "examples/libero/libero_object/libero_object_no_noops/finetuned_chkpt/4999"


USE_WANDB = True




def prepare_observation(obs, model_stats):
    # Get images and ensure proper shape
    img = get_libero_image(obs)  # shape (256, 256, 3)
    wrist_img = get_libero_wrist_image(obs)  # shape (256, 256, 3)
    
    # Ensure all proprio components have 2 dimensions before concatenation
    eef_pos = obs["robot0_eef_pos"].reshape(1, -1)  # shape (1, 3)
    eef_quat = quat2axisangle(obs["robot0_eef_quat"]).reshape(1, -1)  # shape (1, 3)
    gripper_qpos = obs["robot0_gripper_qpos"].reshape(1, -1)  # shape (1, 2)
    
    # Concatenate proprioceptive features
    proprio = np.concatenate([
        eef_pos,
        eef_quat,
        gripper_qpos
    ], axis=-1)  # shape (1, 8)

    # Create observation dictionary with batch dimension
    observation = {
        "image_rgb": img[np.newaxis, ...],  # shape (1, 256, 256, 3)
        "image_wrist": wrist_img[np.newaxis, ...],  # shape (1, 256, 256, 3)
        "proprio": proprio,  # already has batch dim
        "timestep_pad_mask": np.ones((1, 16), dtype=np.float32),
        "timestep": np.zeros((1,), dtype=np.int32),
        "task_completed": np.zeros((1,), dtype=bool),
        "pad_mask_dict": {
            "image_rgb": np.ones((1,), dtype=bool),
            "image_wrist": np.ones((1,), dtype=bool),
            "proprio": np.ones((1,), dtype=bool),
            "timestep": np.ones((1,), dtype=bool),
        }
    }

    observation = {k: v for k, v in observation.items() if k is not None and v is not None}
    
    print("Final observation structure:")
    for k, v in observation.items():
        if k is None:
            print("⚠️ Found None key in observation!")


        if k == "pad_mask_dict":
            print(f"pad_mask_dict:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v.shape}")
        else:
            print(f"{k}: {v.shape}")

    return observation, img





def main():
    wandb.init(project="octo-libero-eval", name=f"octo-{TASK_SUITE_NAME}") if USE_WANDB else None

    # Load Octo model
    print("🔍 Loading Octo model...")
    model = OctoModel.load_pretrained(CHECKPOINT_PATH)

    # Prepare LIBERO task suite
    benchmark_dict = get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    num_tasks = task_suite.n_tasks





    # Set up policy
    # policy_fn = supply_rng(
    #     lambda obs, task: model.sample_actions(
    #         obs,
    #         task=model.create_tasks(texts=task),
    #         unnormalization_statistics=model.dataset_statistics["action"]
    #     )
    # )
    policy_fn = supply_rng(
        lambda obs, task, rng=None: model.sample_actions(
            obs,
            tasks=model.create_tasks(texts=task),
            unnormalization_statistics=model.dataset_statistics["action"],
            rng=rng  # Pass it along if model.sample_actions supports it
        )
    )



    total_successes = 0
    total_episodes = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, model_family="octo", resolution=256)
        env = RobosuiteGymAdapter(env)



        # 🛠 Patch proprio metadata to avoid shape mismatch
        # obs = env.reset()[0]  # get actual observation shape
        # proprio_dim = obs["robot0_proprio-state"].shape[0]

        # # Replace 'proprio' stats with dummy vectors of correct size
        # model.dataset_statistics["proprio"] = {
        #     "mean": [0.0] * proprio_dim,
        #     "std": [1.0] * proprio_dim
        # }



        # Normalize and wrap env
        env = NormalizeProprio(env, model.dataset_statistics)
        env = HistoryWrapper(env, horizon=1)
        env = RHCWrapper(env, exec_horizon=50)

        print(f"🌱 Task {task_id}: {task_description}")
        for episode_idx in tqdm.tqdm(range(TRIALS_PER_TASK)):
            # obs = env.reset()
            # obs, _ = env.reset()
            result = env.reset()
            obs = result[0]  # Get the observation (first element)


            language_instruction = env.get_task()["language_instruction"]


            images = [obs["agentview_image"][0]] if RENDER_ENABLED else []

            episode_return = 0
            success = False
            t = 0

            while t < MAX_STEPS:
                obs_proc, img = prepare_observation(obs, model.dataset_statistics)

                # print(f"obs_proc keys: {list(obs_proc.keys())}")

                # obs_proc = {k: v[None] for k, v in obs_proc.items()}  # add batch dim

                obs_proc = {
                    k: v[None]
                    for k, v in obs_proc.items()
                    if k != 'pad_mask_dict'
                }

                for k in obs_proc.keys():
                    print(f"Key: {k} ({type(k)})")




                # # Infer batch size and sequence length from an existing key like 'proprio'
                # B, T = obs_proc["proprio"].shape[:2] if obs_proc["proprio"].ndim == 3 else (1, obs_proc["proprio"].shape[1])

                # # Inject dummy timestep_pad_mask (all True)
                # obs_proc["timestep_pad_mask"] = jnp.ones((B, T), dtype=jnp.bool_)




                actions = policy_fn(obs_proc, language_instruction)[0]
                obs, reward, done, trunc, info = env.step(actions)
                
                
                if RENDER_ENABLED:
                    images.extend([o["image_primary"][0] for o in info["observations"]])


                episode_return += reward

                if done or trunc:
                    success = True
                    break
                t += 1

            total_episodes += 1
            total_successes += int(success)

            print(f"✅ Episode {episode_idx + 1} Return: {episode_return:.2f} | Success: {success}")
            
            
            if RENDER_ENABLED:
                save_rollout_video(images, total_episodes, success, task_description)


            # if USE_WANDB:
            #     wandb.log({
            #         f"rollout_video/task{task_id}_episode{episode_idx}": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2], fps=10),
            #         "episode_return": episode_return,
            #         "success": success,
            #     })
            if USE_WANDB:
                log_dict = {
                    "episode_return": episode_return,
                    "success": success,
                }
                if RENDER_ENABLED:
                    log_dict[f"rollout_video/task{task_id}_episode{episode_idx}"] = wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2], fps=10)
                wandb.log(log_dict)



    final_success = total_successes / total_episodes
    print(f"\n🏁 Final Success Rate: {final_success:.2%}")
    if USE_WANDB:
        wandb.log({
            "final_success_rate": final_success,
            "total_episodes": total_episodes
        })


if __name__ == "__main__":
    main()
