import os
import json

WANDB_DIR = r"C:\Users\chryk\Downloads\InnoPP\resized\MPOULO\backup-fullsize\fasterrcnn-pytorch-training-pipeline-1\wandb"

def extract_project_dir(run_path):
    metadata_path = os.path.join(run_path, "files", "wandb-metadata.json")

    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                args = metadata.get("args", [])
                if "--project-dir" in args:
                    idx = args.index("--project-dir")
                    if idx + 1 < len(args):
                        return args[idx + 1]
                    else:
                        return "(missing value after --project-dir)"
                else:
                    return "(--project-dir not found)"
        except Exception as e:
            return f"(error reading metadata: {e})"
    else:
        return "(wandb-metadata.json not found)"

def main():
    for folder in os.listdir(WANDB_DIR):
        if folder.startswith("offline-run-"):
            run_path = os.path.join(WANDB_DIR, folder)
            if os.path.isdir(run_path):
                project_dir = extract_project_dir(run_path)
                print(f"{folder} -> {project_dir}")

if __name__ == "__main__":
    main()
