param(
    [string]$Environment = "rtml-videomae2",
    [switch]$CreateEnvironment
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$cache = Join-Path $root "artifacts\video\videomae2"
$repo = Join-Path $cache "VideoMAEv2"
$checkpoint = Join-Path $cache "vit_s_k710_dl_from_giant.pth"
$commit = "29eab1e8a588d1b3ec0cdec7b03a86cca491b74b"
$url = "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth"

if ($CreateEnvironment) {
    # Clone the already verified CUDA PyTorch environment. This avoids silently
    # replacing the machine-specific CUDA wheel with a CPU-only pip wheel.
    conda create -n $Environment --clone rtml-p002-p016 -y
    # VideoMAE2's model module is loaded with a narrow compatibility shim; no
    # torchvision/timm binary is installed here because pip can replace CUDA
    # PyTorch with a CPU wheel on Windows.
    conda run -n $Environment python -m pip install --no-deps einops==0.8.1
}
New-Item -ItemType Directory -Force -Path $cache | Out-Null
if (-not (Test-Path (Join-Path $repo ".git"))) {
    git clone https://github.com/OpenGVLab/VideoMAEv2.git $repo
}
git -C $repo fetch --depth 1 origin $commit
git -C $repo checkout --detach $commit
if (-not (Test-Path $checkpoint)) {
    Invoke-WebRequest -Uri $url -OutFile $checkpoint
}
conda run -n $Environment python -c "import torch, cv2; assert torch.cuda.is_available(), 'CUDA PyTorch is required'; print(torch.__version__, torch.cuda.get_device_name(0))"
Write-Output "VideoMAE2 source and checkpoint are pinned under $cache"
