import os
from huggingface_hub import snapshot_download

BASE_DIR = "/data/private/address_bot/models"

MODELS = {
    "tunib/electra-ko-base": "tunib-electra-ko-base",
    "snunlp/KR-ELECTRA-discriminator": "snunlp-kr-electra-discriminator",
}

# PyTorch/Transformers 사용에 필요한 파일 위주
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.bin",
    "*.safetensors",
    "*.py",
    "README.md",
]

IGNORE_PATTERNS = [
    "*.h5",          # TensorFlow weight
    "*.msgpack",    # Flax weight
    ".git/*",
]

def download_model(repo_id: str, local_name: str):
    local_dir = os.path.join(BASE_DIR, local_name)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\nDownloading: {repo_id}")
    print(f"Local dir  : {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        token=os.environ.get("HF_TOKEN"),  # 공개 모델이면 없어도 됨
    )

    print(f"Done: {repo_id}")

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    for repo_id, local_name in MODELS.items():
        download_model(repo_id, local_name)

    print("\nAll models downloaded.")

if __name__ == "__main__":
    main()