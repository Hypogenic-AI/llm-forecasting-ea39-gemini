from huggingface_hub import HfApi
api = HfApi()
datasets = api.list_datasets(filter="ETTm1")
for d in datasets:
    print(f"ID: {d.id}")
