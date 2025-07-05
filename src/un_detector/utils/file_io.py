import json
import torch


# Function to save dictionaries as JSON files
def save_json(data, json_path):
    with open(json_path, "w") as file:
        json.dump(data, file)


# Function to convert tensors to lists for JSON serialization
def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert Tensor to list
    elif isinstance(obj, dict):
        return {
            key: tensor_to_list(value) for key, value in obj.items()
        }  # Recursively convert dicts
    elif isinstance(obj, list):
        return [tensor_to_list(item) for item in obj]  # Recursively convert lists
    else:
        return obj