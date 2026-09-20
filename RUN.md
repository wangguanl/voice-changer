# 运行命令

- 项目：w-okada/voice-changer（VCClient 实时变声，当前分支 local-custom）
- 生成时间：2026-09-08
- 运行方式：直接运行（uv + Python 3.10）
- 硬件评估：**满足**
  - 项目：实时语音转换（RVC 等），消费级 NVIDIA GPU 即可；官方亦提供 CPU / 预编译包路径，无「最低 24GB」硬门槛
  - 本机：RTX 4080 16GB；评估时约已用 3.5GB、剩余约 12.5GB（另有 seed-vc / so-vits-svc 等 Python 进程）
  - 结论：剩余显存足够跑 RVC 实时推理；若同时再开其他大模型需自行留意显存

## 环境准备

```powershell
# PowerShell 7
pwsh -NoProfile

# ffmpeg（本机已有，按会话加入 PATH）
$env:Path = "E:\Programs\ffmpeg-master-latest-win64-gpl\bin;$env:Path"

# 国内镜像（HF 官方超时；PyPI 官方偏慢）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

cd E:\AI\local-voice\voice-changer\server
uv venv --python 3.10
.\.venv\Scripts\Activate.ps1

# 其余依赖（先装这些；gin 需用 pip）
uv pip install uvicorn==0.21.1 pyOpenSSL==23.1.1 numpy==1.23.5 resampy==0.4.2 python-socketio==5.8.0 fastapi==0.95.1 python-multipart==0.0.6 onnxruntime-gpu==1.13.1 scipy==1.10.1 matplotlib==3.7.1 websockets==11.0.2 faiss-cpu==1.7.3 torchcrepe==0.0.18 librosa==0.9.1 gin_config==0.5.0 einops==0.6.0 local_attention==1.8.5 sounddevice==0.4.6 dataclasses_json==0.5.7 onnxsim==0.4.28 torchfcpe==0.0.3 requests tqdm setuptools==69.5.1 chardet
uv pip install pip
.\.venv\Scripts\python.exe -m pip install gin==0.1.6 -i https://pypi.tuna.tsinghua.edu.cn/simple

# CUDA 版 torch：必须用 pip 从官方 cu118 源装（uv 易落到 +cpu；约 2.6GB）
.\.venv\Scripts\python.exe -m pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA torch 未装上'"
```

权重与示例模型：首次启动由 `MMVCServerSIO.py` 自动下载到 `server/pretrain/` 与 `server/model_dir/`（需 `HF_ENDPOINT`；本机已对 `Downloader.py` 做镜像改写）。

## 启动

- 推荐：`pwsh -NoProfile -File .\start.ps1`
- 说明：单服务，无菜单；默认 HTTPS（自签名，与官方 Docker 一致，避免拉起不存在的 native client）
- 等价手动命令：

```powershell
cd E:\AI\local-voice\voice-changer\server
$env:Path = "E:\Programs\ffmpeg-master-latest-win64-gpl\bin;$env:Path"
$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\Activate.ps1
uv run python MMVCServerSIO.py -p <端口> --https true --host 127.0.0.1 `
  --content_vec_500 pretrain/checkpoint_best_legacy_500.pt `
  --content_vec_500_onnx pretrain/content_vec_500.onnx `
  --content_vec_500_onnx_on true `
  --hubert_base pretrain/hubert_base.pt `
  --hubert_base_jp pretrain/rinna_hubert_base_jp.pt `
  --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt `
  --nsf_hifigan pretrain/nsf_hifigan/model `
  --crepe_onnx_full pretrain/crepe_onnx_full.onnx `
  --crepe_onnx_tiny pretrain/crepe_onnx_tiny.onnx `
  --rmvpe pretrain/rmvpe.pt `
  --model_dir model_dir
```

## 验证

1. 控制台出现 `Please open the following URL` / `https://localhost:<端口>/`
2. 浏览器打开该地址（自签名证书需点继续访问）
3. 页面能加载 GUI；`nvidia-smi` 在加载模型后显存上升且无 CUDA OOM

## 备注

- 端口：建议起点 18888，占用则 `start.ps1` 顺延
- 镜像：HF → `https://hf-mirror.com`；PyPI/uv → 清华；Docker Hub 官方超时（本方案未走 Docker）
- 未选 Docker：官方 `start_docker.sh` 面向 Linux/WSL；本机 Docker Hub 直连超时；项目为单一 Python 服务，直接跑更合适
- 前端：依赖已构建的 `client/demo/dist`
- ffmpeg：`E:\Programs\ffmpeg-master-latest-win64-gpl\bin`
- 官方 README 写 Windows 源码开发以 WSL2+Anaconda 为主；本机按 local-custom + uv 直接跑
- 本目录 `Downloader.py` 支持 `HF_ENDPOINT` 改写权重下载 URL（勿当作上游默认行为）
