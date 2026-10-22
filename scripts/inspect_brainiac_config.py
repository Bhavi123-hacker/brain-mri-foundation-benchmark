from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(repo_id="eugenehp/brainiac", filename="config.json")
config_data = json.loads(open(config_path, encoding="utf-8").read())
print("BrainIAC config.json from eugenehp/brainiac:")
print(json.dumps(config_data, indent=2))
