import os
import json
import numpy as np
from scipy.stats import entropy

# Define the directories containing the .npz files and the truesummary.json file
npz_dir = "/data/mmActs/predictions/NPC/baseline/epoch40/mmimagesTr"
selection_path = os.path.join(npz_dir, "selection.json")
obj_path = "/data/mmActs/predictions/NPC/baseline/mmimagesTr/summary.json"
repre_path = "/data/mmActs/predictions/NPC/baseline/mmimagesTr/density.json"


# Load truesummary.json data
with open(repre_path, "r") as f:
    repre_data = json.load(f)
    
with open(obj_path, "r") as f:
    obj_data = json.load(f)

# List all .npz files in the directory
npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]

# Function to compute entropy-based uncertainty
def compute_uncertainty(prob_array):
    """
    Compute uncertainty using entropy.
    prob_array: numpy array of shape (C, H, W, D) where C is the number of classes.
    """
    prob_array = np.clip(prob_array, 1e-10, 1.0)  # Avoid log(0)
    uncertainty_map = entropy(prob_array, axis=0)  # Compute entropy along the class axis (C)
    return uncertainty_map

# Store results
results = []

# Loop through each .npz file
for npz_file in npz_files:
    file_path = os.path.join(npz_dir, npz_file)
    
    with np.load(file_path) as data:
        prob_key = list(data.files)[0]  # Use the first key (modify if needed)
        prob_array = data[prob_key]  # Shape (C, H, W, D)
        
        # print(f"\nShape: {prob_array.shape}")
        
        if prob_array.ndim != 4:
            print(f"Skipping {npz_file}: Unexpected shape {prob_array.shape}")
            continue
        
        # Extract spatial dimensions (H, W, D)
        H, W, D = prob_array.shape[1:]
        
        # Compute uncertainty map
        uncertainty_map = compute_uncertainty(prob_array)

        # Compute metrics
        total_uncertainty = float(np.sum(uncertainty_map))  # Convert to native Python float
        uncertainty_range = float(uncertainty_map.max() - uncertainty_map.min())  # Convert to float
        total_pixels = H * W * D  # Compute total pixels as H * W * D
        uncertainty_score = total_uncertainty / total_pixels * 100

        # Find corresponding entry in truesummary (adjusting for full vs partial file paths)
        prediction_file = f"/{npz_file.replace('.npz', '.nii.gz')}"
        matching_repre_entry = next((entry for entry in  repre_data["metric_per_case"]
                                    if os.path.basename(entry["prediction_file"]) == os.path.basename(prediction_file)), None)
        matching_obj_entry = next((entry for entry in  obj_data["metric_per_case"]
                                    if os.path.basename(entry["prediction_file"]) == os.path.basename(prediction_file)), None)

        # Initialize represenativeness score to None
        repre_score = None

        # If matching entry is found, add its metrics
        if matching_repre_entry and matching_obj_entry:
            n_repre = matching_repre_entry["metrics"]["(1,)"]["n_density"]
            n_con = matching_obj_entry["metrics"]["(1,)"]["n_con"]
            obj_concent = n_con # Compute objective abundance
            n_inform = uncertainty_score * obj_concent
            # Compute selection criterion
            selection_criterion =  n_inform * n_repre

            result_entry = {
                "metrics": {
                    "(1,)": {
                        "Uncertainty sum": round(total_uncertainty, 4),
                        "Uncertainty score": round(uncertainty_score, 4),
                        "Concentration": round(obj_concent, 4),
                        "Informativeness": round(n_inform, 4),
                        "Representativeness": round(n_repre, 4),
                        "Selection criterion": round(selection_criterion, 4)
                    }
                },
                "prediction_file": prediction_file
            }

            # Store results in the required format
            results.append(result_entry)

# Sort results by Uncertainty Score (descending)
results.sort(key=lambda x: x["metrics"]["(1,)"]["Selection criterion"], reverse=True)

# Save to JSON
summary_data = {"metric_per_case": results}
with open(selection_path, "w") as f:
    json.dump(summary_data, f, indent=4)

print(f"Summary saved to {selection_path}")

print("\nTop 5 Candidates:")
for i, res in enumerate(results, 1):
    print(f"{i}. {res['prediction_file']} - Selection criterion Score: {res['metrics']['(1,)']['Selection criterion']}")


# Print Top 5 results for each metric
top_5_uncertainty_score = sorted(results, key=lambda x: x["metrics"]["(1,)"]["Uncertainty score"], reverse=True)[:5]
top_5_concentration = sorted(results, key=lambda x: x["metrics"]["(1,)"]["Concentration"], reverse=True)[:5]
top_5_n_inform = sorted(results, key=lambda x: x["metrics"]["(1,)"]["Informativeness"], reverse=True)[:5]
top_5_n_repre = sorted(results, key=lambda x: x["metrics"]["(1,)"]["Representativeness"], reverse=True)[:5]

print("\nTop 5 Uncertainty Score Candidates:")
for i, res in enumerate(top_5_uncertainty_score, 1):
    print(f"{i}. {res['prediction_file']} - Uncertainty Score: {res['metrics']['(1,)']['Uncertainty score']}")

print("\nTop 5 Uncertainty Sum Candidates:")
for i, res in enumerate(top_5_concentration, 1):
    print(f"{i}. {res['prediction_file']} - Concentration: {res['metrics']['(1,)']['Concentration']}")

print("\nTop 5 n_inform Candidates:")
for i, res in enumerate(top_5_n_inform, 1):
    print(f"{i}. {res['prediction_file']} - n_inform: {res['metrics']['(1,)']['Informativeness']}")

print("\nTop 5 n_repre Candidates:")
for i, res in enumerate(top_5_n_repre, 1):
    print(f"{i}. {res['prediction_file']} - n_repre: {res['metrics']['(1,)']['Representativeness']}")



