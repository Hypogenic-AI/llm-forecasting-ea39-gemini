from huggingface_hub import HfApi
api = HfApi()

org = "forecastingresearch"
print(f"--- Datasets for organization {org} ---")
try:
    datasets = api.list_datasets(author=org)
    for d in datasets:
        print(f"ID: {d.id}")
except Exception as e:
    print(f"Error: {e}")
