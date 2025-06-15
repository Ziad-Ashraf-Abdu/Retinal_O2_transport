import pickle

# 1. Load the pickle
with open("model_checkpoints/best_model.pkl", "rb") as f:
    data = pickle.load(f)

# 2. Define the keys you care about
wanted = [
    "model_state_dict",
    "error",
    "run_number",
    "param_names",
    "param_ranges"
]

# 3. Print their values
for key in wanted:
    if key not in data:
        print(f"{key} → NOT FOUND\n")
        continue

    val = data[key]
    print(f"{key} → ", end="")

    # If it's the state dict, give a shapes-summary
    if key == "model_state_dict" and isinstance(val, dict):
        print(f"[{len(val)} tensors]")
        for name, tensor in val.items():
            # if it's a PyTorch tensor, show .shape; else, repr
            shape = getattr(tensor, "shape", None)
            if shape is not None:
                print(f"  • {name}: {shape}")
            else:
                print(f"  • {name}: {type(tensor)}")
    else:
        # Otherwise just pretty-print the value
        from pprint import pprint
        pprint(val)

    print()  # blank line between entries
