import os
import json
import csv
from datetime import datetime

# CONFIGURABLE VALUES
POWER_WATTS = 70  # Typical GPU power draw during training (Watts)
COST_PER_KWH = 0.15  # Electricity cost in $ per kWh

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
        elif arg.startswith("-"):
            key = arg.lstrip("-")
            try:
                value = next(it)
                args_dict[key] = value
            except StopIteration:
                args_dict[key] = True
    return args_dict

def get_model_profile(model_name):
    model_name = model_name.lower()
    if "resnet" in model_name:
        return {"epoch_time_hr": 0.45}
    elif "mobilenet" in model_name:
        return {"epoch_time_hr": 0.30}
    elif "efficientnet" in model_name:
        return {"epoch_time_hr": 0.40}
    else:
        return {"epoch_time_hr": 0.40}  # fallback

def estimate_hours(model, epochs):
    profile = get_model_profile(model)
    return epochs * profile["epoch_time_hr"]

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
                    name = args.get("name") or args.get("--name") or os.path.basename(dirpath)
                    epochs = int(args.get("e", 0))
                    batch_size = int(args.get("b", 0))
                    img_size = args.get("ims", "unknown")
                    model = meta.get("args", [])[1] if len(meta.get("args", [])) > 1 else "unknown"
                    gpu_name = meta.get("gpu", "unknown")

                    if "efficientnet" in model.lower():
                        continue  # Skip ongoing training runs

                    est_hours = estimate_hours(model, epochs)
                    energy_kwh = (POWER_WATTS * est_hours) / 1000
                    cost = energy_kwh * COST_PER_KWH

                    summary.append({
                        "Name": name,
                        "Model": model,
                        "Epochs": epochs,
                        "Batch Size": batch_size,
                        "Image Size": img_size,
                        "GPU": gpu_name,
                        "Est. Hours": round(est_hours, 2),
                        "Est. Energy (kWh)": round(energy_kwh, 2),
                        "Est. Cost ($)": round(cost, 2)
                    })

    return summary

def write_csv(summary, output_path="training_cost_summary.csv"):
    if not summary:
        print("No data to write.")
        return
    keys = summary[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n✅ CSV written to: {output_path}")

def print_summary_table(summary):
    from tabulate import tabulate

    print("\n🧾 Summary:")
    print(tabulate(summary, headers="keys", tablefmt="fancy_grid"))

    total_hours = sum(row["Est. Hours"] for row in summary)
    total_energy = sum(row["Est. Energy (kWh)"] for row in summary)
    total_cost = sum(row["Est. Cost ($)"] for row in summary)

    print("\n💡 Totals:")
    print(f"Total GPU Time:    {total_hours:.2f} hours")
    print(f"Total Energy Use:  {total_energy:.2f} kWh")
    print(f"Estimated Cost:    ${total_cost:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate compute cost from wandb-metadata.json files.")
    parser.add_argument("root_dir", help="Root folder to scan for wandb-metadata.json files")
    parser.add_argument("--csv", help="Write CSV output", action="store_true")
    args = parser.parse_args()

    results = scan_and_summarize(args.root_dir)
    print_summary_table(results)

    if args.csv:
        write_csv(results)
