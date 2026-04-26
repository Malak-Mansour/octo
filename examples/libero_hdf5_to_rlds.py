# # # # # # #!/usr/bin/env python3
# # # # # # """
# # # # # # Convert LIBERO object HDF5 dataset to RLDS format.

# # # # # # # Convert all files in a directory
# # # # # # python examples/libero_hdf5_to_rlds.py --input_dir=examples/libero/libero_object --output_dir=examples/libero/libero_object/rlds

# # # # # # # Convert a single file
# # # # # # python examples/libero_hdf5_to_rlds.py --input_dir=single_file.hdf5 --output_dir=examples/libero/libero_object/rlds
# # # # # # """

# # # # # # import os
# # # # # # import re
# # # # # # import argparse
# # # # # # import h5py
# # # # # # import tensorflow as tf
# # # # # # import tensorflow_datasets as tfds
# # # # # # import numpy as np
# # # # # # from tqdm import tqdm
# # # # # # from typing import List, Optional

# # # # # # def parse_task_description(filename: str) -> str:
# # # # # #     """
# # # # # #     Extract and format task description from LIBERO filename.
# # # # # #     Example:
# # # # # #     'libero_10_Pick_up_the_black_bowl_and_place_it_on_the_plate_demo_0.hdf5'
# # # # # #     -> 'Pick up the black bowl and place it on the plate'
# # # # # #     """
# # # # # #     basename = os.path.basename(filename)
    
# # # # # #     # Remove LIBERO-specific prefixes/suffixes
# # # # # #     clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
    
# # # # # #     # Convert underscores to spaces and clean up
# # # # # #     description = clean_name.replace('_', ' ').strip()
# # # # # #     description = re.sub(r'\s+', ' ', description)  # Collapse multiple spaces
    
# # # # # #     return description

# # # # # # def convert_hdf5_to_rlds(hdf5_path: str, output_path: str) -> None:
# # # # # #     """
# # # # # #     Convert a single HDF5 file to RLDS format.
# # # # # #     Args:
# # # # # #         hdf5_path: Path to input HDF5 file
# # # # # #         output_path: Directory to save RLDS dataset
# # # # # #     """
# # # # # #     # Create RLDS dataset builder
# # # # # #     builder = tfds.core.DatasetBuilder(
# # # # # #         data_dir=output_path,
# # # # # #         dataset_name='libero_object',
# # # # # #     )
    
# # # # # #     with h5py.File(hdf5_path, 'r') as hf:
# # # # # #         # Define RLDS feature structure
# # # # # #         builder.info.features = tfds.features.FeaturesDict({
# # # # # #             'steps': tfds.features.Dataset({
# # # # # #                 'observation': tfds.features.FeaturesDict({
# # # # # #                     'image_primary': tfds.features.Image(shape=(128, 128, 3)),  # agentview
# # # # # #                     'image_secondary': tfds.features.Image(shape=(128, 128, 3)),  # eye-in-hand
# # # # # #                     'ee_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # # # # #                     'ee_ori': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # # # # #                     'joint_states': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # # # # #                     'gripper_states': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
# # # # # #                 }),
# # # # # #                 'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # # # # #                 'reward': tfds.features.Scalar(dtype=tf.float32),
# # # # # #                 'is_terminal': tfds.features.Scalar(dtype=tf.bool),
# # # # # #                 'is_first': tfds.features.Scalar(dtype=tf.bool),
# # # # # #                 'is_last': tfds.features.Scalar(dtype=tf.bool),
# # # # # #             }),
# # # # # #             'episode_metadata': tfds.features.FeaturesDict({
# # # # # #                 'language_instruction': tfds.features.Text(),
# # # # # #                 'task_id': tfds.features.Text(),
# # # # # #                 'original_filename': tfds.features.Text(),
# # # # # #             }),
# # # # # #         })
        
# # # # # #         # Process each demonstration in the HDF5 file
# # # # # #         for demo_id in hf.keys():
# # # # # #             demo = hf[demo_id]
# # # # # #             task_desc = parse_task_description(hdf5_path)
            
# # # # # #             # Convert each step in the demonstration
# # # # # #             steps = []
# # # # # #             for i in range(len(demo['actions'])):
# # # # # #                 steps.append({
# # # # # #                     'observation': {
# # # # # #                         'image_primary': demo['obs/agentview_rgb'][i],
# # # # # #                         'image_secondary': demo['obs/eye_in_hand_rgb'][i],
# # # # # #                         'ee_pos': demo['obs/ee_pos'][i],
# # # # # #                         'ee_ori': demo['obs/ee_ori'][i],
# # # # # #                         'joint_states': demo['obs/joint_states'][i],
# # # # # #                         'gripper_states': demo['obs/gripper_states'][i],
# # # # # #                     },
# # # # # #                     'action': demo['actions'][i],
# # # # # #                     'reward': float(demo['rewards'][i]),
# # # # # #                     'is_terminal': bool(demo['dones'][i]),
# # # # # #                     'is_first': (i == 0),
# # # # # #                     'is_last': (i == len(demo['actions']) - 1),
# # # # # #                 })
            
# # # # # #             # Add episode to dataset
# # # # # #             builder.as_dataset({
# # # # # #                 'steps': steps,
# # # # # #                 'episode_metadata': {
# # # # # #                     'language_instruction': task_desc,
# # # # # #                     'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
# # # # # #                     'original_filename': os.path.basename(hdf5_path),
# # # # # #                 },
# # # # # #             })
    
# # # # # #     # Finalize the dataset
# # # # # #     builder.download_and_prepare()

# # # # # # def process_directory(input_dir: str, output_dir: str) -> None:
# # # # # #     """
# # # # # #     Process all HDF5 files in a directory.
# # # # # #     Args:
# # # # # #         input_dir: Directory containing HDF5 files
# # # # # #         output_dir: Base directory for RLDS output
# # # # # #     """
# # # # # #     if not os.path.exists(output_dir):
# # # # # #         os.makedirs(output_dir)
    
# # # # # #     hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    
# # # # # #     for filename in tqdm(hdf5_files, desc="Converting HDF5 files"):
# # # # # #         hdf5_path = os.path.join(input_dir, filename)
# # # # # #         task_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        
# # # # # #         if os.path.exists(task_output_dir):
# # # # # #             print(f"Skipping {filename} - output already exists")
# # # # # #             continue
            
# # # # # #         try:
# # # # # #             convert_hdf5_to_rlds(hdf5_path, task_output_dir)
# # # # # #         except Exception as e:
# # # # # #             print(f"Failed to convert {filename}: {str(e)}")

# # # # # # def main():
# # # # # #     parser = argparse.ArgumentParser(
# # # # # #         description='Convert LIBERO object HDF5 dataset to RLDS format'
# # # # # #     )
# # # # # #     parser.add_argument(
# # # # # #         '--input_dir',
# # # # # #         type=str,
# # # # # #         default='examples/libero/libero_object',
# # # # # #         help='Directory containing HDF5 files'
# # # # # #     )
# # # # # #     parser.add_argument(
# # # # # #         '--output_dir',
# # # # # #         type=str,
# # # # # #         default='output_rlds',
# # # # # #         help='Directory to save RLDS datasets'
# # # # # #     )
# # # # # #     args = parser.parse_args()
    
# # # # # #     print(f"Converting HDF5 files from {args.input_dir} to RLDS in {args.output_dir}")
# # # # # #     process_directory(args.input_dir, args.output_dir)
# # # # # #     print("Conversion complete!")

# # # # # # if __name__ == '__main__':
# # # # # #     main()

# # # # # #!/usr/bin/env python3
# # # # # """
# # # # # Convert LIBERO object HDF5 dataset to RLDS format.
# # # # # Fixed version with proper DatasetBuilder implementation.
# # # # # """
# # # # # import os
# # # # # import re
# # # # # import argparse
# # # # # import h5py
# # # # # import tensorflow as tf
# # # # # import tensorflow_datasets as tfds
# # # # # import numpy as np
# # # # # from tqdm import tqdm
# # # # # from typing import Dict, Any

# # # # # class LiberoRldsBuilder(tfds.core.GeneratorBasedBuilder):
# # # # #     """Proper implementation of RLDS builder for LIBERO datasets."""
# # # # #     VERSION = tfds.core.Version('1.0.0')
    
# # # # #     def _info(self) -> tfds.core.DatasetInfo:
# # # # #         return tfds.core.DatasetInfo(
# # # # #             builder=self,
# # # # #             description="LIBERO object manipulation dataset in RLDS format",
# # # # #             features=tfds.features.FeaturesDict({
# # # # #                 'steps': tfds.features.Dataset({
# # # # #                     'observation': tfds.features.FeaturesDict({
# # # # #                         'image_primary': tfds.features.Image(shape=(128, 128, 3)),
# # # # #                         'image_secondary': tfds.features.Image(shape=(128, 128, 3)),
# # # # #                         'ee_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # # # #                         'ee_ori': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # # # #                         'joint_states': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # # # #                         'gripper_states': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
# # # # #                     }),
# # # # #                     'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # # # #                     'reward': tfds.features.Scalar(dtype=tf.float32),
# # # # #                     'is_terminal': tfds.features.Scalar(dtype=tf.bool),
# # # # #                     'is_first': tfds.features.Scalar(dtype=tf.bool),
# # # # #                     'is_last': tfds.features.Scalar(dtype=tf.bool),
# # # # #                 }),
# # # # #                 'episode_metadata': tfds.features.FeaturesDict({
# # # # #                     'language_instruction': tfds.features.Text(),
# # # # #                     'task_id': tfds.features.Text(),
# # # # #                     'original_filename': tfds.features.Text(),
# # # # #                 }),
# # # # #             }),
# # # # #             supervised_keys=None,
# # # # #         )

# # # # #     def _split_generators(self, dl_manager: tfds.download.DownloadManager):
# # # # #         return {'train': self._generate_examples()}

# # # # #     def _generate_examples(self):
# # # # #         hdf5_path = self._builder_config.hdf5_path
# # # # #         with h5py.File(hdf5_path, 'r') as hf:
# # # # #             for demo_id in hf.keys():
# # # # #                 demo = hf[demo_id]
# # # # #                 task_desc = parse_task_description(hdf5_path)
                
# # # # #                 steps = []
# # # # #                 for i in range(len(demo['actions'])):
# # # # #                     steps.append({
# # # # #                         'observation': {
# # # # #                             'image_primary': demo['obs/agentview_rgb'][i],
# # # # #                             'image_secondary': demo['obs/eye_in_hand_rgb'][i],
# # # # #                             'ee_pos': demo['obs/ee_pos'][i],
# # # # #                             'ee_ori': demo['obs/ee_ori'][i],
# # # # #                             'joint_states': demo['obs/joint_states'][i],
# # # # #                             'gripper_states': demo['obs/gripper_states'][i],
# # # # #                         },
# # # # #                         'action': demo['actions'][i],
# # # # #                         'reward': float(demo['rewards'][i]),
# # # # #                         'is_terminal': bool(demo['dones'][i]),
# # # # #                         'is_first': (i == 0),
# # # # #                         'is_last': (i == len(demo['actions']) - 1),
# # # # #                     })
                
# # # # #                 yield demo_id, {
# # # # #                     'steps': steps,
# # # # #                     'episode_metadata': {
# # # # #                         'language_instruction': task_desc,
# # # # #                         'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
# # # # #                         'original_filename': os.path.basename(hdf5_path),
# # # # #                     },
# # # # #                 }

# # # # # def parse_task_description(filename: str) -> str:
# # # # #     """Extract task description from LIBERO filename."""
# # # # #     basename = os.path.basename(filename)
# # # # #     clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
# # # # #     description = clean_name.replace('_', ' ').strip()
# # # # #     return re.sub(r'\s+', ' ', description)

# # # # # def convert_hdf5_to_rlds(hdf5_path: str, output_path: str) -> None:
# # # # #     """Convert single HDF5 file to RLDS using proper builder."""
# # # # #     builder_config = tfds.core.BuilderConfig(
# # # # #         name=os.path.splitext(os.path.basename(hdf5_path))[0],
# # # # #         version="1.0.0",
# # # # #         hdf5_path=hdf5_path,
# # # # #     )
    
# # # # #     builder = LiberoRldsBuilder(
# # # # #         data_dir=output_path,
# # # # #         config=builder_config,
# # # # #     )
    
# # # # #     builder.download_and_prepare()
# # # # #     builder.as_dataset()

# # # # # def process_directory(input_dir: str, output_dir: str) -> None:
# # # # #     """Process all HDF5 files in directory."""
# # # # #     os.makedirs(output_dir, exist_ok=True)
# # # # #     hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    
# # # # #     for filename in tqdm(hdf5_files, desc="Converting HDF5 files"):
# # # # #         hdf5_path = os.path.join(input_dir, filename)
# # # # #         task_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        
# # # # #         if os.path.exists(task_output_dir):
# # # # #             print(f"Skipping {filename} - output already exists")
# # # # #             continue
            
# # # # #         try:
# # # # #             convert_hdf5_to_rlds(hdf5_path, task_output_dir)
# # # # #         except Exception as e:
# # # # #             print(f"Failed to convert {filename}: {str(e)}")

# # # # # def main():
# # # # #     parser = argparse.ArgumentParser(
# # # # #         description='Convert LIBERO object HDF5 dataset to RLDS format'
# # # # #     )
# # # # #     parser.add_argument(
# # # # #         '--input_dir',
# # # # #         type=str,
# # # # #         default='examples/libero/libero_object',
# # # # #         help='Directory containing HDF5 files'
# # # # #     )
# # # # #     parser.add_argument(
# # # # #         '--output_dir',
# # # # #         type=str,
# # # # #         default='output_rlds',
# # # # #         help='Directory to save RLDS datasets'
# # # # #     )
# # # # #     args = parser.parse_args()
    
# # # # #     print(f"Converting HDF5 files from {args.input_dir} to RLDS in {args.output_dir}")
# # # # #     process_directory(args.input_dir, args.output_dir)
# # # # #     print("Conversion complete!")

# # # # # if __name__ == '__main__':
# # # # #     # Suppress TensorFlow logging and CUDA warnings
# # # # #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # # # #     tf.get_logger().setLevel('ERROR')
# # # # #     main()

# # # # #!/usr/bin/env python3
# # # # """
# # # # Convert LIBERO object HDF5 dataset to RLDS format.
# # # # Fixed version with proper BuilderConfig implementation.
# # # # """
# # # # import os
# # # # import re
# # # # import argparse
# # # # import h5py
# # # # import tensorflow as tf
# # # # import tensorflow_datasets as tfds
# # # # import numpy as np
# # # # from tqdm import tqdm
# # # # from typing import Dict, Any

# # # # class LiberoBuilderConfig(tfds.core.BuilderConfig):
# # # #     """Custom builder config that accepts hdf5_path parameter."""
# # # #     def __init__(self, *, hdf5_path: str, **kwargs):
# # # #         super().__init__(**kwargs)
# # # #         self.hdf5_path = hdf5_path

# # # # class LiberoRldsBuilder(tfds.core.GeneratorBasedBuilder):
# # # #     """Proper implementation of RLDS builder for LIBERO datasets."""
# # # #     VERSION = tfds.core.Version('1.0.0')
# # # #     BUILDER_CONFIGS = [LiberoBuilderConfig(name="default", description="LIBERO dataset")]
    
# # # #     def _info(self) -> tfds.core.DatasetInfo:
# # # #         return tfds.core.DatasetInfo(
# # # #             builder=self,
# # # #             description="LIBERO object manipulation dataset in RLDS format",
# # # #             features=tfds.features.FeaturesDict({
# # # #                 'steps': tfds.features.Dataset({
# # # #                     'observation': tfds.features.FeaturesDict({
# # # #                         'image_primary': tfds.features.Image(shape=(128, 128, 3)),
# # # #                         'image_secondary': tfds.features.Image(shape=(128, 128, 3)),
# # # #                         'ee_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # # #                         'ee_ori': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # # #                         'joint_states': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # # #                         'gripper_states': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
# # # #                     }),
# # # #                     'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # # #                     'reward': tfds.features.Scalar(dtype=tf.float32),
# # # #                     'is_terminal': tfds.features.Scalar(dtype=tf.bool),
# # # #                     'is_first': tfds.features.Scalar(dtype=tf.bool),
# # # #                     'is_last': tfds.features.Scalar(dtype=tf.bool),
# # # #                 }),
# # # #                 'episode_metadata': tfds.features.FeaturesDict({
# # # #                     'language_instruction': tfds.features.Text(),
# # # #                     'task_id': tfds.features.Text(),
# # # #                     'original_filename': tfds.features.Text(),
# # # #                 }),
# # # #             }),
# # # #             supervised_keys=None,
# # # #         )

# # # #     def _split_generators(self, dl_manager: tfds.download.DownloadManager):
# # # #         return {'train': self._generate_examples()}

# # # #     def _generate_examples(self):
# # # #         hdf5_path = self.builder_config.hdf5_path
# # # #         with h5py.File(hdf5_path, 'r') as hf:
# # # #             for demo_id in hf.keys():
# # # #                 demo = hf[demo_id]
# # # #                 task_desc = parse_task_description(hdf5_path)
                
# # # #                 steps = []
# # # #                 for i in range(len(demo['actions'])):
# # # #                     steps.append({
# # # #                         'observation': {
# # # #                             'image_primary': demo['obs/agentview_rgb'][i],
# # # #                             'image_secondary': demo['obs/eye_in_hand_rgb'][i],
# # # #                             'ee_pos': demo['obs/ee_pos'][i],
# # # #                             'ee_ori': demo['obs/ee_ori'][i],
# # # #                             'joint_states': demo['obs/joint_states'][i],
# # # #                             'gripper_states': demo['obs/gripper_states'][i],
# # # #                         },
# # # #                         'action': demo['actions'][i],
# # # #                         'reward': float(demo['rewards'][i]),
# # # #                         'is_terminal': bool(demo['dones'][i]),
# # # #                         'is_first': (i == 0),
# # # #                         'is_last': (i == len(demo['actions']) - 1),
# # # #                     })
                
# # # #                 yield demo_id, {
# # # #                     'steps': steps,
# # # #                     'episode_metadata': {
# # # #                         'language_instruction': task_desc,
# # # #                         'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
# # # #                         'original_filename': os.path.basename(hdf5_path),
# # # #                     },
# # # #                 }

# # # # def parse_task_description(filename: str) -> str:
# # # #     """Extract task description from LIBERO filename."""
# # # #     basename = os.path.basename(filename)
# # # #     clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
# # # #     description = clean_name.replace('_', ' ').strip()
# # # #     return re.sub(r'\s+', ' ', description)

# # # # def convert_hdf5_to_rlds(hdf5_path: str, output_path: str) -> None:
# # # #     """Convert single HDF5 file to RLDS using proper builder."""
# # # #     builder = LiberoRldsBuilder(
# # # #         data_dir=output_path,
# # # #         config=LiberoBuilderConfig(
# # # #             name=os.path.splitext(os.path.basename(hdf5_path))[0],
# # # #             description=f"LIBERO dataset from {os.path.basename(hdf5_path)}",
# # # #             hdf5_path=hdf5_path,
# # # #         )
# # # #     )
    
# # # #     builder.download_and_prepare()

# # # # def process_directory(input_dir: str, output_dir: str) -> None:
# # # #     """Process all HDF5 files in directory."""
# # # #     os.makedirs(output_dir, exist_ok=True)
# # # #     hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    
# # # #     for filename in tqdm(hdf5_files, desc="Converting HDF5 files"):
# # # #         hdf5_path = os.path.join(input_dir, filename)
# # # #         task_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        
# # # #         if os.path.exists(task_output_dir):
# # # #             print(f"Skipping {filename} - output already exists")
# # # #             continue
            
# # # #         try:
# # # #             convert_hdf5_to_rlds(hdf5_path, task_output_dir)
# # # #             print(f"Successfully converted {filename}")
# # # #         except Exception as e:
# # # #             print(f"Failed to convert {filename}: {str(e)}")

# # # # def main():
# # # #     parser = argparse.ArgumentParser(
# # # #         description='Convert LIBERO object HDF5 dataset to RLDS format'
# # # #     )
# # # #     parser.add_argument(
# # # #         '--input_dir',
# # # #         type=str,
# # # #         default='examples/libero/libero_object',
# # # #         help='Directory containing HDF5 files'
# # # #     )
# # # #     parser.add_argument(
# # # #         '--output_dir',
# # # #         type=str,
# # # #         default='output_rlds',
# # # #         help='Directory to save RLDS datasets'
# # # #     )
# # # #     args = parser.parse_args()
    
# # # #     print(f"Converting HDF5 files from {args.input_dir} to RLDS in {args.output_dir}")
# # # #     process_directory(args.input_dir, args.output_dir)
# # # #     print("Conversion complete!")

# # # # if __name__ == '__main__':
# # # #     # Suppress TensorFlow logging and CUDA warnings
# # # #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # # #     tf.get_logger().setLevel('ERROR')
# # # #     main()

# # # #!/usr/bin/env python3
# # # """
# # # Convert LIBERO object HDF5 dataset to RLDS format.
# # # Final working version with proper BuilderConfig implementation.
# # # """
# # # import os
# # # import re
# # # import argparse
# # # import h5py
# # # import tensorflow as tf
# # # import tensorflow_datasets as tfds
# # # import numpy as np
# # # from tqdm import tqdm
# # # from typing import Dict, Any

# # # class LiberoBuilderConfig(tfds.core.BuilderConfig):
# # #     """Custom builder config that accepts hdf5_path parameter."""
# # #     def __init__(self, *, hdf5_path: str = None, **kwargs):
# # #         super().__init__(**kwargs)
# # #         self.hdf5_path = hdf5_path

# # # class LiberoRldsBuilder(tfds.core.GeneratorBasedBuilder):
# # #     """Proper implementation of RLDS builder for LIBERO datasets."""
# # #     VERSION = tfds.core.Version('1.0.0')
# # #     BUILDER_CONFIGS = [
# # #         LiberoBuilderConfig(
# # #             name="default",
# # #             description="Default LIBERO dataset config",
# # #             hdf5_path=None  # This will be overridden when used
# # #         )
# # #     ]
    
# # #     def _info(self) -> tfds.core.DatasetInfo:
# # #         return tfds.core.DatasetInfo(
# # #             builder=self,
# # #             description="LIBERO object manipulation dataset in RLDS format",
# # #             features=tfds.features.FeaturesDict({
# # #                 'steps': tfds.features.Dataset({
# # #                     'observation': tfds.features.FeaturesDict({
# # #                         'image_primary': tfds.features.Image(shape=(128, 128, 3)),
# # #                         'image_secondary': tfds.features.Image(shape=(128, 128, 3)),
# # #                         'ee_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # #                         'ee_ori': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
# # #                         'joint_states': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # #                         'gripper_states': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
# # #                     }),
# # #                     'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
# # #                     'reward': tfds.features.Scalar(dtype=tf.float32),
# # #                     'is_terminal': tfds.features.Scalar(dtype=tf.bool),
# # #                     'is_first': tfds.features.Scalar(dtype=tf.bool),
# # #                     'is_last': tfds.features.Scalar(dtype=tf.bool),
# # #                 }),
# # #                 'episode_metadata': tfds.features.FeaturesDict({
# # #                     'language_instruction': tfds.features.Text(),
# # #                     'task_id': tfds.features.Text(),
# # #                     'original_filename': tfds.features.Text(),
# # #                 }),
# # #             }),
# # #             supervised_keys=None,
# # #         )

# # #     def _split_generators(self, dl_manager: tfds.download.DownloadManager):
# # #         return {'train': self._generate_examples()}

# # #     def _generate_examples(self):
# # #         if not self.builder_config.hdf5_path:
# # #             raise ValueError("hdf5_path must be specified in the builder config")
            
# # #         hdf5_path = self.builder_config.hdf5_path
# # #         with h5py.File(hdf5_path, 'r') as hf:
# # #             for demo_id in hf.keys():
# # #                 demo = hf[demo_id]
# # #                 task_desc = parse_task_description(hdf5_path)
                
# # #                 steps = []
# # #                 for i in range(len(demo['actions'])):
# # #                     steps.append({
# # #                         'observation': {
# # #                             'image_primary': demo['obs/agentview_rgb'][i],
# # #                             'image_secondary': demo['obs/eye_in_hand_rgb'][i],
# # #                             'ee_pos': demo['obs/ee_pos'][i],
# # #                             'ee_ori': demo['obs/ee_ori'][i],
# # #                             'joint_states': demo['obs/joint_states'][i],
# # #                             'gripper_states': demo['obs/gripper_states'][i],
# # #                         },
# # #                         'action': demo['actions'][i],
# # #                         'reward': float(demo['rewards'][i]),
# # #                         'is_terminal': bool(demo['dones'][i]),
# # #                         'is_first': (i == 0),
# # #                         'is_last': (i == len(demo['actions']) - 1),
# # #                     })
                
# # #                 yield demo_id, {
# # #                     'steps': steps,
# # #                     'episode_metadata': {
# # #                         'language_instruction': task_desc,
# # #                         'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
# # #                         'original_filename': os.path.basename(hdf5_path),
# # #                     },
# # #                 }

# # # def parse_task_description(filename: str) -> str:
# # #     """Extract task description from LIBERO filename."""
# # #     basename = os.path.basename(filename)
# # #     clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
# # #     description = clean_name.replace('_', ' ').strip()
# # #     return re.sub(r'\s+', ' ', description)

# # # def convert_hdf5_to_rlds(hdf5_path: str, output_path: str) -> None:
# # #     """Convert single HDF5 file to RLDS using proper builder."""
# # #     builder = LiberoRldsBuilder(
# # #         data_dir=output_path,
# # #         config=LiberoBuilderConfig(
# # #             name=os.path.splitext(os.path.basename(hdf5_path))[0],
# # #             description=f"LIBERO dataset from {os.path.basename(hdf5_path)}",
# # #             hdf5_path=hdf5_path,
# # #         )
# # #     )
    
# # #     builder.download_and_prepare()

# # # def process_directory(input_dir: str, output_dir: str) -> None:
# # #     """Process all HDF5 files in directory."""
# # #     os.makedirs(output_dir, exist_ok=True)
# # #     hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    
# # #     for filename in tqdm(hdf5_files, desc="Converting HDF5 files"):
# # #         hdf5_path = os.path.join(input_dir, filename)
# # #         task_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        
# # #         if os.path.exists(task_output_dir):
# # #             print(f"Skipping {filename} - output already exists")
# # #             continue
            
# # #         try:
# # #             convert_hdf5_to_rlds(hdf5_path, task_output_dir)
# # #             print(f"Successfully converted {filename}")
# # #         except Exception as e:
# # #             print(f"Failed to convert {filename}: {str(e)}")

# # # def main():
# # #     parser = argparse.ArgumentParser(
# # #         description='Convert LIBERO object HDF5 dataset to RLDS format'
# # #     )
# # #     parser.add_argument(
# # #         '--input_dir',
# # #         type=str,
# # #         default='examples/libero/libero_object',
# # #         help='Directory containing HDF5 files'
# # #     )
# # #     parser.add_argument(
# # #         '--output_dir',
# # #         type=str,
# # #         default='output_rlds',
# # #         help='Directory to save RLDS datasets'
# # #     )
# # #     args = parser.parse_args()
    
# # #     print(f"Converting HDF5 files from {args.input_dir} to RLDS in {args.output_dir}")
# # #     process_directory(args.input_dir, args.output_dir)
# # #     print("Conversion complete!")

# # # if __name__ == '__main__':
# # #     # Suppress TensorFlow logging and CUDA warnings
# # #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # #     tf.get_logger().setLevel('ERROR')
# # #     main()

# # #!/usr/bin/env python3
# # """
# # Convert LIBERO object HDF5 dataset to standard RLDS TFRecord format.
# # """
# # import os
# # import re
# # import argparse
# # import h5py
# # import tensorflow as tf
# # import numpy as np
# # from tqdm import tqdm
# # from typing import Dict, Any

# # def parse_task_description(filename: str) -> str:
# #     """Extract task description from LIBERO filename."""
# #     basename = os.path.basename(filename)
# #     clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
# #     description = clean_name.replace('_', ' ').strip()
# #     return re.sub(r'\s+', ' ', description)

# # def create_rlds_example(demo, hdf5_path: str) -> Dict[str, Any]:
# #     """Create RLDS example from HDF5 demo."""
# #     task_desc = parse_task_description(hdf5_path)
    
# #     steps = []
# #     for i in range(len(demo['actions'])):
# #         steps.append({
# #             'observation': {
# #                 'image_primary': demo['obs/agentview_rgb'][i].tobytes(),
# #                 'image_secondary': demo['obs/eye_in_hand_rgb'][i].tobytes(),
# #                 'ee_pos': demo['obs/ee_pos'][i].astype(np.float32),
# #                 'ee_ori': demo['obs/ee_ori'][i].astype(np.float32),
# #                 'joint_states': demo['obs/joint_states'][i].astype(np.float32),
# #                 'gripper_states': demo['obs/gripper_states'][i].astype(np.float32),
# #             },
# #             'action': demo['actions'][i].astype(np.float32),
# #             'reward': float(demo['rewards'][i]),
# #             'is_terminal': bool(demo['dones'][i]),
# #             'is_first': (i == 0),
# #             'is_last': (i == len(demo['actions']) - 1),
# #         })
    
# #     return {
# #         'steps': steps,
# #         'episode_metadata': {
# #             'language_instruction': task_desc,
# #             'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
# #             'original_filename': os.path.basename(hdf5_path),
# #         }
# #     }

# # def write_tfrecords(hdf5_path: str, output_path: str, num_shards: int = 8) -> None:
# #     """Write HDF5 data to TFRecord files."""
# #     os.makedirs(output_path, exist_ok=True)
    
# #     # Create dataset info
# #     dataset_info = {
# #         "description": "LIBERO object manipulation dataset",
# #         "dataset_size": 0,
# #         "num_episodes": 0,
# #         "num_steps": 0,
# #         "action_spec": {
# #             "shape": [7],
# #             "dtype": "float32"
# #         },
# #         "observation_spec": {
# #             "image_primary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #             "image_secondary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #             "ee_pos": {"shape": [3], "dtype": "float32"},
# #             "ee_ori": {"shape": [3], "dtype": "float32"},
# #             "joint_states": {"shape": [7], "dtype": "float32"},
# #             "gripper_states": {"shape": [2], "dtype": "float32"},
# #         }
# #     }

# #     with h5py.File(hdf5_path, 'r') as hf:
# #         # Create TFRecord writers
# #         writers = [
# #             tf.io.TFRecordWriter(
# #                 os.path.join(output_path, f"{os.path.splitext(os.path.basename(hdf5_path))[0]}-train.tfrecord-{i:05d}-of-{num_shards:05d}")
# #             ) for i in range(num_shards)
# #         ]
        
# #         # Process each demonstration
# #         demo_count = 0
# #         step_count = 0
# #         for demo_id in hf.keys():
# #             demo = hf[demo_id]
# #             example = create_rlds_example(demo, hdf5_path)
            
# #             # Write to TFRecord
# #             tf_example = tf.train.Example(
# #                 features=tf.train.Features(
# #                     feature={
# #                         'steps': tf.train.Feature(
# #                             bytes_list=tf.train.BytesList(
# #                                 value=[tf.io.serialize_tensor(tf.nest.map_structure(tf.convert_to_tensor, example['steps'])).numpy()]
# #                             )
# #                         ),
# #                         'episode_metadata': tf.train.Feature(
# #                             bytes_list=tf.train.BytesList(
# #                                 value=[tf.io.serialize_tensor(tf.nest.map_structure(tf.convert_to_tensor, example['episode_metadata'])).numpy()]
# #                             )
# #                         )
# #                     }
# #                 )
# #             )
            
# #             writers[demo_count % num_shards].write(tf_example.SerializeToString())
# #             demo_count += 1
# #             step_count += len(demo['actions'])
        
# #         # Update dataset info
# #         dataset_info["dataset_size"] = os.path.getsize(hdf5_path)
# #         dataset_info["num_episodes"] = demo_count
# #         dataset_info["num_steps"] = step_count
        
# #         # Close writers
# #         for writer in writers:
# #             writer.close()
    
# #     # Write dataset info and statistics
# #     with open(os.path.join(output_path, "dataset_info.json"), 'w') as f:
# #         json.dump(dataset_info, f)
    
# #     # For now, we'll create an empty statistics file - you should compute real statistics
# #     with open(os.path.join(output_path, "dataset_statistics.json"), 'w') as f:
# #         json.dump({}, f)
    
# #     # Write features specification
# #     features_spec = {
# #         "steps": {
# #             "observation": {
# #                 "image_primary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #                 "image_secondary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #                 "ee_pos": {"shape": [3], "dtype": "float32"},
# #                 "ee_ori": {"shape": [3], "dtype": "float32"},
# #                 "joint_states": {"shape": [7], "dtype": "float32"},
# #                 "gripper_states": {"shape": [2], "dtype": "float32"},
# #             },
# #             "action": {"shape": [7], "dtype": "float32"},
# #             "reward": {"dtype": "float32"},
# #             "is_terminal": {"dtype": "bool"},
# #             "is_first": {"dtype": "bool"},
# #             "is_last": {"dtype": "bool"},
# #         },
# #         "episode_metadata": {
# #             "language_instruction": {"dtype": "string"},
# #             "task_id": {"dtype": "string"},
# #             "original_filename": {"dtype": "string"},
# #         }
# #     }
    
# #     with open(os.path.join(output_path, "features.json"), 'w') as f:
# #         json.dump(features_spec, f)

# # def process_directory(input_dir: str, output_dir: str) -> None:
# #     """Process all HDF5 files in directory."""
# #     os.makedirs(output_dir, exist_ok=True)
# #     hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    
# #     for filename in tqdm(hdf5_files, desc="Converting HDF5 files"):
# #         hdf5_path = os.path.join(input_dir, filename)
# #         task_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        
# #         if os.path.exists(task_output_dir):
# #             print(f"Skipping {filename} - output already exists")
# #             continue
            
# #         try:
# #             write_tfrecords(hdf5_path, task_output_dir)
# #             print(f"Successfully converted {filename}")
# #         except Exception as e:
# #             print(f"Failed to convert {filename}: {str(e)}")

# # def main():
# #     parser = argparse.ArgumentParser(
# #         description='Convert LIBERO object HDF5 dataset to RLDS TFRecord format'
# #     )
# #     parser.add_argument(
# #         '--input_dir',
# #         type=str,
# #         default='examples/libero/libero_object',
# #         help='Directory containing HDF5 files'
# #     )
# #     parser.add_argument(
# #         '--output_dir',
# #         type=str,
# #         default='output_rlds',
# #         help='Directory to save RLDS datasets'
# #     )
# #     args = parser.parse_args()
    
# #     print(f"Converting HDF5 files from {args.input_dir} to RLDS in {args.output_dir}")
# #     process_directory(args.input_dir, args.output_dir)
# #     print("Conversion complete!")

# # if __name__ == '__main__':
# #     import json  # Added for JSON serialization
# #     # Suppress TensorFlow logging and CUDA warnings
# #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# #     tf.get_logger().setLevel('ERROR')
# #     main()

# # """
# # Convert LIBERO object HDF5 dataset to standard RLDS TFRecord format.


# # python examples/libero_hdf5_to_rlds.py --input_dir=examples/libero/libero_object --output_dir=examples/libero/libero_object/rlds

# # the files in the output directory are empty

# # """
# # import os
# # import re
# # import argparse
# # import h5py
# # import tensorflow as tf
# # import numpy as np
# # from tqdm import tqdm
# # from typing import Dict, Any
# # import json

# # def parse_task_description(filename: str) -> str:
# #     """Extract task description from LIBERO filename."""
# #     basename = os.path.basename(filename)
# #     clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
# #     description = clean_name.replace('_', ' ').strip()
# #     return re.sub(r'\s+', ' ', description)

# # def create_rlds_example(demo_group, hdf5_path: str) -> Dict[str, Any]:
# #     """Create RLDS example from HDF5 demo group."""
# #     task_desc = parse_task_description(hdf5_path)
    
# #     # Get the length from one of the arrays
# #     length = len(demo_group['actions'])
    
# #     steps = []
# #     for i in range(length):
# #         steps.append({
# #             'observation': {
# #                 'image_primary': demo_group['obs/agentview_rgb'][i].tobytes(),
# #                 'image_secondary': demo_group['obs/eye_in_hand_rgb'][i].tobytes(),
# #                 'ee_pos': demo_group['obs/ee_pos'][i].astype(np.float32),
# #                 'ee_ori': demo_group['obs/ee_ori'][i].astype(np.float32),
# #                 'joint_states': demo_group['obs/joint_states'][i].astype(np.float32),
# #                 'gripper_states': demo_group['obs/gripper_states'][i].astype(np.float32),
# #             },
# #             'action': demo_group['actions'][i].astype(np.float32),
# #             'reward': float(demo_group['rewards'][i]),
# #             'is_terminal': bool(demo_group['dones'][i]),
# #             'is_first': (i == 0),
# #             'is_last': (i == length - 1),
# #         })
    
# #     return {
# #         'steps': steps,
# #         'episode_metadata': {
# #             'language_instruction': task_desc,
# #             'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
# #             'original_filename': os.path.basename(hdf5_path),
# #         }
# #     }

# # def write_tfrecords(hdf5_path: str, output_path: str, num_shards: int = 8) -> None:
# #     """Write HDF5 data to TFRecord files."""
# #     os.makedirs(output_path, exist_ok=True)
    
# #     # Create dataset info
# #     dataset_info = {
# #         "description": "LIBERO object manipulation dataset",
# #         "dataset_size": 0,
# #         "num_episodes": 0,
# #         "num_steps": 0,
# #         "action_spec": {
# #             "shape": [7],
# #             "dtype": "float32"
# #         },
# #         "observation_spec": {
# #             "image_primary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #             "image_secondary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #             "ee_pos": {"shape": [3], "dtype": "float32"},
# #             "ee_ori": {"shape": [3], "dtype": "float32"},
# #             "joint_states": {"shape": [7], "dtype": "float32"},
# #             "gripper_states": {"shape": [2], "dtype": "float32"},
# #         }
# #     }

# #     with h5py.File(hdf5_path, 'r') as hf:
# #         # Create TFRecord writers
# #         writers = [
# #             tf.io.TFRecordWriter(
# #                 os.path.join(output_path, f"{os.path.splitext(os.path.basename(hdf5_path))[0]}-train.tfrecord-{i:05d}-of-{num_shards:05d}")
# #             ) for i in range(num_shards)
# #         ]
        
# #         # Process each demonstration group (demo_0, demo_1, etc.)
# #         demo_count = 0
# #         step_count = 0
        
# #         # Get all demo groups (they start with 'demo_')
# #         demo_groups = [key for key in hf.keys() if key.startswith('demo_')]
        
# #         for demo_key in demo_groups:
# #             demo_group = hf[demo_key]
# #             example = create_rlds_example(demo_group, hdf5_path)
            
# #             # Write to TFRecord
# #             tf_example = tf.train.Example(
# #                 features=tf.train.Features(
# #                     feature={
# #                         'steps': tf.train.Feature(
# #                             bytes_list=tf.train.BytesList(
# #                                 value=[tf.io.serialize_tensor(tf.nest.map_structure(tf.convert_to_tensor, example['steps'])).numpy()]
# #                             )
# #                         ),
# #                         'episode_metadata': tf.train.Feature(
# #                             bytes_list=tf.train.BytesList(
# #                                 value=[tf.io.serialize_tensor(tf.nest.map_structure(tf.convert_to_tensor, example['episode_metadata'])).numpy()]
# #                             )
# #                         )
# #                     }
# #                 )
# #             )
            
# #             writers[demo_count % num_shards].write(tf_example.SerializeToString())
# #             demo_count += 1
# #             step_count += len(demo_group['actions'])
        
# #         # Update dataset info
# #         dataset_info["dataset_size"] = os.path.getsize(hdf5_path)
# #         dataset_info["num_episodes"] = demo_count
# #         dataset_info["num_steps"] = step_count
        
# #         # Close writers
# #         for writer in writers:
# #             writer.close()
    
# #     # Write dataset info and statistics
# #     with open(os.path.join(output_path, "dataset_info.json"), 'w') as f:
# #         json.dump(dataset_info, f, indent=2)
    
# #     # For now, we'll create an empty statistics file - you should compute real statistics
# #     with open(os.path.join(output_path, "dataset_statistics.json"), 'w') as f:
# #         json.dump({}, f, indent=2)
    
# #     # Write features specification
# #     features_spec = {
# #         "steps": {
# #             "observation": {
# #                 "image_primary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #                 "image_secondary": {"shape": [128, 128, 3], "dtype": "uint8"},
# #                 "ee_pos": {"shape": [3], "dtype": "float32"},
# #                 "ee_ori": {"shape": [3], "dtype": "float32"},
# #                 "joint_states": {"shape": [7], "dtype": "float32"},
# #                 "gripper_states": {"shape": [2], "dtype": "float32"},
# #             },
# #             "action": {"shape": [7], "dtype": "float32"},
# #             "reward": {"dtype": "float32"},
# #             "is_terminal": {"dtype": "bool"},
# #             "is_first": {"dtype": "bool"},
# #             "is_last": {"dtype": "bool"},
# #         },
# #         "episode_metadata": {
# #             "language_instruction": {"dtype": "string"},
# #             "task_id": {"dtype": "string"},
# #             "original_filename": {"dtype": "string"},
# #         }
# #     }
    
# #     with open(os.path.join(output_path, "features.json"), 'w') as f:
# #         json.dump(features_spec, f, indent=2)

# # def process_directory(input_dir: str, output_dir: str) -> None:
# #     """Process all HDF5 files in directory."""
# #     os.makedirs(output_dir, exist_ok=True)
# #     hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    
# #     for filename in tqdm(hdf5_files, desc="Converting HDF5 files"):
# #         hdf5_path = os.path.join(input_dir, filename)
# #         task_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        
# #         if os.path.exists(task_output_dir):
# #             print(f"Skipping {filename} - output already exists")
# #             continue
            
# #         try:
# #             write_tfrecords(hdf5_path, task_output_dir)
# #             print(f"Successfully converted {filename}")
# #         except Exception as e:
# #             print(f"Failed to convert {filename}: {str(e)}")

# # def main():
# #     parser = argparse.ArgumentParser(
# #         description='Convert LIBERO object HDF5 dataset to RLDS TFRecord format'
# #     )
# #     parser.add_argument(
# #         '--input_dir',
# #         type=str,
# #         required=True,
# #         help='Directory containing HDF5 files'
# #     )
# #     parser.add_argument(
# #         '--output_dir',
# #         type=str,
# #         required=True,
# #         help='Directory to save RLDS datasets'
# #     )
# #     args = parser.parse_args()
    
# #     print(f"Converting HDF5 files from {args.input_dir} to RLDS in {args.output_dir}")
# #     process_directory(args.input_dir, args.output_dir)
# #     print("Conversion complete!")

# # if __name__ == '__main__':
# #     # Suppress TensorFlow logging and CUDA warnings
# #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# #     tf.get_logger().setLevel('ERROR')
# #     main()

# import os
# import re
# import h5py
# import tensorflow as tf
# import numpy as np

# def parse_task_description(filename: str) -> str:
#     basename = os.path.basename(filename)
#     clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
#     description = clean_name.replace('_', ' ').strip()
#     return re.sub(r'\s+', ' ', description)

# def create_rlds_example(demo_group, hdf5_path: str):
#     task_desc = parse_task_description(hdf5_path)
#     length = len(demo_group['actions'])

#     steps = []
#     for i in range(length):
#         step = {
#             'observation': {
#                 'image_primary': demo_group['obs/agentview_rgb'][i],
#                 'image_secondary': demo_group['obs/eye_in_hand_rgb'][i],
#                 'ee_pos': demo_group['obs/ee_pos'][i],
#                 'ee_ori': demo_group['obs/ee_ori'][i],
#                 'joint_states': demo_group['obs/joint_states'][i],
#                 'gripper_states': demo_group['obs/gripper_states'][i],
#             },
#             'action': demo_group['actions'][i],
#             'reward': float(demo_group['rewards'][i]),
#             'is_terminal': bool(demo_group['dones'][i]),
#             'is_first': (i == 0),
#             'is_last': (i == length - 1),
#         }
#         steps.append(step)

#     return {
#         'steps': steps,
#         'episode_metadata': {
#             'language_instruction': task_desc,
#             'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
#             'original_filename': os.path.basename(hdf5_path),
#         }
#     }

# def write_single_sequence_example(example, hdf5_path, output_path, demo_key):
#     steps = example['steps']
#     metadata = example['episode_metadata']

#     def float_list_feature(values):
#         return tf.train.Feature(float_list=tf.train.FloatList(value=values))

#     def bytes_feature(value):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#     def string_feature(value):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

#     def bool_feature(value):
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

#     sequence_features = {
#         "action": tf.train.FeatureList(feature=[float_list_feature(step["action"]) for step in steps]),
#         "reward": tf.train.FeatureList(feature=[float_list_feature([step["reward"]]) for step in steps]),
#         "is_terminal": tf.train.FeatureList(feature=[bool_feature(step["is_terminal"]) for step in steps]),
#         "is_first": tf.train.FeatureList(feature=[bool_feature(step["is_first"]) for step in steps]),
#         "is_last": tf.train.FeatureList(feature=[bool_feature(step["is_last"]) for step in steps]),
#         "ee_pos": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["ee_pos"]) for step in steps]),
#         "ee_ori": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["ee_ori"]) for step in steps]),
#         "joint_states": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["joint_states"]) for step in steps]),
#         "gripper_states": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["gripper_states"]) for step in steps]),
#         "image_primary": tf.train.FeatureList(feature=[bytes_feature(step["observation"]["image_primary"].tobytes()) for step in steps]),
#         "image_secondary": tf.train.FeatureList(feature=[bytes_feature(step["observation"]["image_secondary"].tobytes()) for step in steps]),
#     }

#     context_features = tf.train.Features(feature={
#         "language_instruction": string_feature(metadata["language_instruction"]),
#         "task_id": string_feature(metadata["task_id"]),
#         "original_filename": string_feature(metadata["original_filename"]),
#     })

#     seq_example = tf.train.SequenceExample(
#         context=context_features,
#         feature_lists=tf.train.FeatureLists(feature_list=sequence_features)
#     )

#     output_file = os.path.join(
#         output_path,
#         f"{os.path.splitext(os.path.basename(hdf5_path))[0]}_{demo_key}.tfrecord"
#     )
#     with tf.io.TFRecordWriter(output_file) as writer:
#         writer.write(seq_example.SerializeToString())

#     print(f"✅ Wrote {output_file}")

# def write_tfrecords_sequence_example(hdf5_path: str, output_path: str):
#     os.makedirs(output_path, exist_ok=True)
#     with h5py.File(hdf5_path, 'r') as hf:
#         # First, try to find demo groups under a "data" group
#         if "data" in hf:
#             data_group = hf["data"]
#             demo_keys = [k for k in data_group.keys() if "demo" in k]
#             # if demo_keys:
#             #     demo_group = data_group[demo_keys[0]]
#             if demo_keys:
#                 for demo_key in demo_keys:
#                     demo_group = data_group[demo_key]
#                     example = create_rlds_example(demo_group, hdf5_path)
#                     write_single_sequence_example(example, hdf5_path, output_path, demo_key)

#             else:
#                 print("No demo groups found under 'data'.")
#                 return
#         else:
#             # Fall back to searching at the top level
#             demo_keys = [k for k in hf.keys() if "demo" in k]
#             if demo_keys:
#                 demo_group = hf[demo_keys[0]]
#             else:
#                 print("No demo groups found.")
#                 return

#         example = create_rlds_example(demo_group, hdf5_path)
#         steps = example['steps']
#         metadata = example['episode_metadata']

#     # Print example values to confirm
#     print(f"✅ Total steps: {len(steps)}")
#     print("🧩 Sample step keys:", steps[0].keys())
#     print("🧠 Sample action:", steps[0]['action'])
#     print("🎯 Sample reward:", steps[0]['reward'])

#     def float_list_feature(values):
#         return tf.train.Feature(float_list=tf.train.FloatList(value=values))

#     def bytes_feature(value):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#     def string_feature(value):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

#     def bool_feature(value):
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

#     sequence_features = {
#         "action": tf.train.FeatureList(feature=[
#             float_list_feature(step["action"]) for step in steps
#         ]),
#         "reward": tf.train.FeatureList(feature=[
#             float_list_feature([step["reward"]]) for step in steps
#         ]),
#         "is_terminal": tf.train.FeatureList(feature=[
#             bool_feature(step["is_terminal"]) for step in steps
#         ]),
#         "is_first": tf.train.FeatureList(feature=[
#             bool_feature(step["is_first"]) for step in steps
#         ]),
#         "is_last": tf.train.FeatureList(feature=[
#             bool_feature(step["is_last"]) for step in steps
#         ]),
#         "ee_pos": tf.train.FeatureList(feature=[
#             float_list_feature(step["observation"]["ee_pos"]) for step in steps
#         ]),
#         "ee_ori": tf.train.FeatureList(feature=[
#             float_list_feature(step["observation"]["ee_ori"]) for step in steps
#         ]),
#         "joint_states": tf.train.FeatureList(feature=[
#             float_list_feature(step["observation"]["joint_states"]) for step in steps
#         ]),
#         "gripper_states": tf.train.FeatureList(feature=[
#             float_list_feature(step["observation"]["gripper_states"]) for step in steps
#         ]),
#         "image_primary": tf.train.FeatureList(feature=[
#             bytes_feature(step["observation"]["image_primary"].tobytes()) for step in steps
#         ]),
#         "image_secondary": tf.train.FeatureList(feature=[
#             bytes_feature(step["observation"]["image_secondary"].tobytes()) for step in steps
#         ]),
#     }

#     context_features = tf.train.Features(feature={
#         "language_instruction": string_feature(metadata["language_instruction"]),
#         "task_id": string_feature(metadata["task_id"]),
#         "original_filename": string_feature(metadata["original_filename"]),
#     })

#     seq_example = tf.train.SequenceExample(
#         context=context_features,
#         feature_lists=tf.train.FeatureLists(feature_list=sequence_features)
#     )

#     output_file = os.path.join(
#         output_path, f"{os.path.splitext(os.path.basename(hdf5_path))[0]}.tfrecord"
#     )
#     with tf.io.TFRecordWriter(output_file) as writer:
#         writer.write(seq_example.SerializeToString())

#     print(f"✅ Wrote TFRecord to {output_file}")

# # Only run this part outside of Jupyter/IPython
# # if __name__ == "__main__":
# #     hdf5_path = "libero/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
# #     output_dir = "libero/libero_object/rlds"
# #     write_tfrecords_sequence_example(hdf5_path, output_dir)

# if __name__ == "__main__":
#     import glob
#     output_dir = "examples/libero/libero_object/rlds"

#     input_dir = "examples/libero/libero_object"
#     # Find all .hdf5 files in the input directory (non-recursive)
#     hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    
#     if not hdf5_files:
#         print("No HDF5 files found in", input_dir)
#     else:
#         for hdf5_path in hdf5_files:
#             print(f"Processing file: {hdf5_path}")
#             write_tfrecords_sequence_example(hdf5_path, output_dir)

import os
import re
import h5py
import tensorflow as tf
import numpy as np

def parse_task_description(filename: str) -> str:
    basename = os.path.basename(filename)
    clean_name = re.sub(r'^(libero_\d+_|_demo_\d+\.hdf5$)', '', basename)
    description = clean_name.replace('_', ' ').strip()
    return re.sub(r'\s+', ' ', description)

def create_rlds_example(demo_group, hdf5_path: str):
    task_desc = parse_task_description(hdf5_path)
    length = len(demo_group['actions'])

    steps = []
    for i in range(length):
        step = {
            'observation': {
                'image_primary': demo_group['obs/agentview_rgb'][i],
                'image_secondary': demo_group['obs/eye_in_hand_rgb'][i],
                'ee_pos': demo_group['obs/ee_pos'][i],
                'ee_ori': demo_group['obs/ee_ori'][i],
                'joint_states': demo_group['obs/joint_states'][i],
                'gripper_states': demo_group['obs/gripper_states'][i],
                'ee_states': demo_group['obs/ee_states'][i],
                'robot_states': demo_group['robot_states'][i],
                'states': demo_group['states'][i],
            },
            'action': demo_group['actions'][i],
            'reward': float(demo_group['rewards'][i]),
            'is_terminal': bool(demo_group['dones'][i]),
            'is_first': (i == 0),
            'is_last': (i == length - 1),
        }
        steps.append(step)

    return {
        'steps': steps,
        'episode_metadata': {
            'language_instruction': task_desc,
            'task_id': os.path.splitext(os.path.basename(hdf5_path))[0],
            'original_filename': os.path.basename(hdf5_path),
        }
    }

def write_single_sequence_example(example, hdf5_path, output_path, demo_key):
    steps = example['steps']
    metadata = example['episode_metadata']

    def float_list_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def string_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    def bool_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    sequence_features = {
        "action": tf.train.FeatureList(feature=[float_list_feature(step["action"]) for step in steps]),
        "reward": tf.train.FeatureList(feature=[float_list_feature([step["reward"]]) for step in steps]),
        "is_terminal": tf.train.FeatureList(feature=[bool_feature(step["is_terminal"]) for step in steps]),
        "is_first": tf.train.FeatureList(feature=[bool_feature(step["is_first"]) for step in steps]),
        "is_last": tf.train.FeatureList(feature=[bool_feature(step["is_last"]) for step in steps]),
        "ee_pos": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["ee_pos"]) for step in steps]),
        "ee_ori": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["ee_ori"]) for step in steps]),
        "joint_states": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["joint_states"]) for step in steps]),
        "gripper_states": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["gripper_states"]) for step in steps]),
        "ee_states": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["ee_states"]) for step in steps]),
        "robot_states": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["robot_states"]) for step in steps]),
        "states": tf.train.FeatureList(feature=[float_list_feature(step["observation"]["states"]) for step in steps]),
        "image_primary": tf.train.FeatureList(feature=[bytes_feature(step["observation"]["image_primary"].tobytes()) for step in steps]),
        "image_secondary": tf.train.FeatureList(feature=[bytes_feature(step["observation"]["image_secondary"].tobytes()) for step in steps]),
    }

    context_features = tf.train.Features(feature={
        "language_instruction": string_feature(metadata["language_instruction"]),
        "task_id": string_feature(metadata["task_id"]),
        "original_filename": string_feature(metadata["original_filename"]),
    })

    seq_example = tf.train.SequenceExample(
        context=context_features,
        feature_lists=tf.train.FeatureLists(feature_list=sequence_features)
    )

    output_file = os.path.join(
        output_path,
        f"{os.path.splitext(os.path.basename(hdf5_path))[0]}_{demo_key}.tfrecord"
    )
    with tf.io.TFRecordWriter(output_file) as writer:
        writer.write(seq_example.SerializeToString())

    print(f"✅ Wrote {output_file}")

def write_tfrecords_sequence_example(hdf5_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    with h5py.File(hdf5_path, 'r') as hf:
        if "data" in hf:
            data_group = hf["data"]
            demo_keys = [k for k in data_group.keys() if "demo" in k]
            if demo_keys:
                for demo_key in demo_keys:
                    demo_group = data_group[demo_key]
                    example = create_rlds_example(demo_group, hdf5_path)
                    write_single_sequence_example(example, hdf5_path, output_path, demo_key)
            else:
                print("No demo groups found under 'data'.")
        else:
            demo_keys = [k for k in hf.keys() if "demo" in k]
            if demo_keys:
                for demo_key in demo_keys:
                    demo_group = hf[demo_key]
                    example = create_rlds_example(demo_group, hdf5_path)
                    write_single_sequence_example(example, hdf5_path, output_path, demo_key)
            else:
                print("No demo groups found.")

if __name__ == "__main__":
    import glob
    output_dir = "libero/libero_object/rlds"
    input_dir = "libero/libero_object"

    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))

    if not hdf5_files:
        print("No HDF5 files found in", input_dir)
    else:
        for hdf5_path in hdf5_files:
            print(f"📂 Processing file: {hdf5_path}")
            write_tfrecords_sequence_example(hdf5_path, output_dir)
