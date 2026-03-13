import wandb
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
ENTITY = "peterparsons02-eth-z-rich"  # e.g., "jane-doe"
PROJECT = "my_experiment"     # e.g., "my-awesome-model"

# You can find the Run ID in the W&B URL: 
# wandb.ai/entity/project/runs/THIS_IS_THE_ID
RUN_A_ID = "cepffnfh"      # The run that timed out
RUN_B_ID = "cxd4excb"     # The resumed run

# ==========================================
# 2. Initialization
# ==========================================
api = wandb.Api()

print("Fetching runs from W&B...")
run_a = api.run(f"{ENTITY}/{PROJECT}/{RUN_A_ID}")
run_b = api.run(f"{ENTITY}/{PROJECT}/{RUN_B_ID}")

# Start a brand new W&B run to hold the combined data
wandb.init(
    entity=ENTITY,
    project=PROJECT,
    name=f"Merged: {run_a.name} & {run_b.name}",
    job_type="merge_runs",
    config=run_a.config, # Inherit the hyperparameters from the first run
    tags=["merged"]
)

# ==========================================
# 3. Data Extraction & Logging
# ==========================================
last_step_logged = -1

def sync_run_history(run):
    global last_step_logged
    
    # scan_history() downloads ALL data points, avoiding W&B's default sampling
    history = run.scan_history()
    
    for row in tqdm(history, desc=f"Syncing {run.id}"):
        current_step = row.get("_step")
        
        # W&B requires steps to be strictly increasing. 
        # This skips data if Run B repeated a few steps from Run A's last checkpoint.
        if current_step is not None and current_step <= last_step_logged:
            continue
            
        # Remove W&B internal keys (they start with an underscore)
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        
        if clean_row:
            if current_step is not None:
                wandb.log(clean_row, step=current_step)
                last_step_logged = current_step
            else:
                wandb.log(clean_row)

print("\n--- Processing Run A ---")
sync_run_history(run_a)

print("\n--- Processing Run B ---")
sync_run_history(run_b)

# Finish the new run to ensure it syncs to the cloud
wandb.finish()
print("\nMerge complete! Check your W&B dashboard.")