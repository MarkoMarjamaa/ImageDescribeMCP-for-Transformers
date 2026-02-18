# ImageDescribeMCP-for-Transformers
Simple Streamable HTTP MCP Server that runs transformers LLM with images, http:// or file:// from internal list. 

I created this because needed some interface for R-4B model, and it seemed there wont be quants for that 
In Open VLM Leaderboard R-4B was doing great for such a small model. 
https://huggingface.co/spaces/opencompass/open_vlm_leaderboard

MCP Server can be used from LLM as a tool when LLM don't have to know the url/username/password for cameras. This script contains array of cameras in use. 

# Downloading the R-4B
```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="YannQi/R-4B",local_dir="R-4B")
```

# Installing packages
I'm using Ryzen 395+ so torch is for that
With Amd's version of "cuda" uses torch_dtype=torch.float32, Nvidia torch_dtype=torch.float16
```
pip install requests
pip install pillow

pip install ./triton-3.2.0+rocm7.1.0.git20943800-cp312-cp312-linux_x86_64.whl
pip install ./torch-2.6.0+rocm7.1.0.lw.git78f6ff78-cp312-cp312-linux_x86_64.whl
pip install ./torchaudio-2.6.0+rocm7.1.0.gitd8831425-cp312-cp312-linux_x86_64.whl
pip install ./torchvision-0.21.0+rocm7.1.0.git4040d51f-cp312-cp312-linux_x86_64.whl

pip install transformers
pip install requests_testadapter

pip install fastapi
pip install uvicorn
pip install mcp
```

## Service definition
This service runs in virtual-env. 
```
[Unit]
Description=ImageDescribeMCP(R-4B)
After=network.target
[Service]
User=marko

WorkingDirectory=/home/mysecretusername/ImageDescribe
ExecStart=/home/mysecretusername/ImageDescribe/bin/python /home/mysecretusername/ImageDescribe/ImageDescribeMCP.py
Restart=always
[Install]
WantedBy=multi-user.target
```

