# Load model directly
from transformers import AutoModel
from octo.model import OctoModel

model = AutoModel.from_pretrained("rail-berkeley/octo-small")


model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
task = model.create_tasks(texts=["pick up the spoon"])
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))