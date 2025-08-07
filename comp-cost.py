import os
import json
import csv
from tabulate import tabulate

# Config
COST_PER_KWH = 0.15  # $ per kilowatt-hour

# Realistic epoch durations (minutes) and power draw per GPU
GPU_PROFILES = {
    "nvidia rtx 4050": {
        "power_watts": 80,
        "epoch_times": {
            "mobilenet": 0.12,
            "resnet": 0.23,
            "efficientnet": 0.14
        }
    },
    "nvidia 3080 ti": {
        "power_watts": 250,
        "epoch_times": {
            "mobilenet": 0.09,
            "resnet": 0.53,
            "efficientnet": 0.37
        }
    }
}

def parse_args(args_list):
    args_dict = {}
    it = iter(args_list)
    for arg in it:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            try:
                value = next(it)
                if value.startswith("--"):
                    args_dict[key] = True
                    it = [value] + list(it)
                    continue
                args_dict[key] = value
            except StopIteration:
                args_dict[key] = True
    return args_dict

def identify_model_type(model_name):
    model_name = model_name.lower()
    if "mobilenet" in model_name:
        return "mobilenet"
    elif "resnet" in model_name:
        return "resnet"
    elif "efficientnet" in model_name:
        return "efficientnet"
    else:
        return "unknown"

def estimate_cost(gpu_name, model_type, epochs):
    profile = GPU_PROFILES.get(gpu_name.lower())
    if not profile or model_type not in profile["epoch_times"]:
        return None  # Unknown GPU or model

    epoch_time_min = profile["epoch_times"][model_type]
    total_time_hr = (epoch_time_min * epochs) / 60
    energy_kwh = (profile["power_watts"] * total_time_hr) / 1000
    cost = energy_kwh * COST_PER_KWH

    return {
        "Est. Hours": round(total_time_hr, 2),
        "Est. Energy (kWh)": round(energy_kwh, 2),
        "Est. Cost ($)": round(cost, 2),
        "Power (W)": profile["power_watts"]
    }

def scan_and_summarize(root_dir):
    summary = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "wandb-metadata.json":
                full_path = os.path.join(dirpath, file)
                with open(full_path, "r") as f:
                    try:
                        meta = json.load(f)
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")
                        continue

                    args = parse_args(meta.get("args", []))
                    name = args.get("name") or os.path.basename(dirpath)
                    epochs = int(args.get("e", 0))
                    batch_size = int(args.get("b", 0))
                    img_size = args.get("ims", "unknown")
                    model_name = meta.get("args", [])[1] if len(meta.get("args", [])) > 1 else "unknown"
                    gpu_name = meta.get("gpu", "unknown")

                    model_type = identify_model_type(model_name)
                    cost_data = estimate_cost(gpu_name, model_type, epochs)

                    if not cost_data:
                        print(f"⚠️ Skipping unknown model/GPU: {model_name} / {gpu_name}")
                        continue

                    summary.append({
                        "Run Name": name,
                        "Model": model_name,
                        "Epochs": epochs,
                        "GPU": gpu_name,
                        "Power (W)": cost_data["Power (W)"],
                        "Est. Hours": cost_data["Est. Hours"],
                        "Est. Energy (kWh)": cost_data["Est. Energy (kWh)"],
                        "Est. Cost ($)": cost_data["Est. Cost ($)"]
                    })

    return summary

def print_summary(summary):
    if not summary:
        print("No valid runs found.")
        return

    print("\n🧾 Training Cost Summary:")
    print(tabulate(summary, headers="keys", tablefmt="fancy_grid"))

    # Totals per GPU
    from collections import defaultdict
    totals = defaultdict(lambda: {"hours": 0, "energy": 0, "cost": 0})

    for row in summary:
        gpu = row["GPU"].lower()
        totals[gpu]["hours"] += row["Est. Hours"]
        totals[gpu]["energy"] += row["Est. Energy (kWh)"]
        totals[gpu]["cost"] += row["Est. Cost ($)"]

    print("\n💻 Totals by GPU:")
    for gpu, data in totals.items():
        print(f"\n🔹 GPU: {gpu}")
        print(f"   Total Time:   {data['hours']:.2f} hrs")
        print(f"   Total Energy: {data['energy']:.2f} kWh")
        print(f"   Total Cost:   ${data['cost']:.2f}")

def write_csv(summary, path="comp_cost_summary.csv"):
    if not summary:
        return
    keys = summary[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n📁 CSV saved to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute training cost per GPU")
    parser.add_argument("root_dir", help="Directory containing wandb runs")
    parser.add_argument("--csv", action="store_true", help="Write CSV output")
    args = parser.parse_args()

    results = scan_and_summarize(args.root_dir)
    print_summary(results)

    if args.csv:
        write_csv(results)
