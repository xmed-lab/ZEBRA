from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="McGregorW/NSD_fsLR",
    repo_type="dataset",
    local_dir="./NSD_fsLR",
    local_dir_use_symlinks=False,
    resume_download=True,
)
