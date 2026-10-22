from huggingface_hub import HfApi

api = HfApi()
models = api.list_models(search="brainiac")
print("Found models matching 'brainiac':")
for m in models:
    print(f"ID: {m.id}, Downloads: {m.downloads}, Likes: {m.likes}")
