from datasets import Dataset
import os

round0_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs/2025_05_21_11_16/demos/round-000/dagger-demo-0-f1d6267bcf3a456c8ca33a673df53f07.npz"
round0_dataset = Dataset.from_file(f"{round0_path}/data-00000-of-00001.arrow")

print("Keys:", round0_dataset.column_names)
print("Example:", round0_dataset[0])




"""
Structure:
    demos/
    ├── round-000/
    │   ├── dagger-demo-0-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.npz
    │   ├── dagger-demo-1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.npz
    │   └── ...
    ├── round-001/
    │   └── ...

- npz folders: This is not a normal .npz file from NumPy — it's actually a directory named 
    with .npz extension (a convention). Inside it, you find a Hugging Face Arrow dataset structure.
- arrow file: This is the main data file, in Apache Arrow format. It contains:
        Flattened transitions: observations, actions, rewards, dones, infos
        Its columnar and optimized for fast access and large scale
- dataset_info.json: Metadata describing: Dataset features, Number of examples, Fingerprint (unique ID)
    Helps datasets library understand the schema without reading the data.
- state.json: Stores internal state info for resuming operations.
    Includes generation seeds, fingerprint hashes, or pre/post-processing steps used
    Mostly internal for the Hugging Face dataset format

Why this format?
    Easy serialization
    Schema consistency
    Compatibility with data pipelines
    Fast loading (you can load 100k+ transitions in seconds)

Note: The saved demos contain expert transitions, not actions from the trained policy.
"""