from huggingface_hub import HfApi

api = HfApi()

for repo_id in ["eugenehp/brainiac", "ilex-hub/brainiac.1", "Anthony-Addison/BrainIAC_model_weights"]:
    print(f"\n=== REPOSITORY: {repo_id} ===")
    try:
        files = api.list_repo_files(repo_id=repo_id)
        for f in files:
            print("  ", f)
    except Exception as e:
        print("  Error:", e)
