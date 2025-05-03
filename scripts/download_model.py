from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cerspense/zeroscope_v2_576w",
    local_dir="./zeroscope-model",
)