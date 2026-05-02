#!/usr/bin/env bash

OUTPUT_FILE="env_report.md"

echo "# 实验环境信息报告" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## 一、操作系统信息" >> $OUTPUT_FILE
if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -a >> $OUTPUT_FILE 2>&1
else
    cat /etc/os-release >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

echo "内核版本:" >> $OUTPUT_FILE
uname -a >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## 二、CPU信息" >> $OUTPUT_FILE
lscpu | grep -E "Model name|Socket|Thread|Core|CPU\(s\)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## 三、GPU / 显存 / 驱动" >> $OUTPUT_FILE
if command -v nvidia-smi >/dev/null 2>&1; then
    echo '```' >> $OUTPUT_FILE
    nvidia-smi >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE

    echo "### GPU简要信息" >> $OUTPUT_FILE
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv >> $OUTPUT_FILE
else
    echo "未检测到 NVIDIA GPU 或驱动未安装" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

echo "## 四、CUDA信息" >> $OUTPUT_FILE
if command -v nvcc >/dev/null 2>&1; then
    echo '```' >> $OUTPUT_FILE
    nvcc --version >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
else
    echo "未检测到 nvcc（可能只安装了 runtime）" >> $OUTPUT_FILE
fi

if [ -f "/usr/local/cuda/version.txt" ]; then
    echo "" >> $OUTPUT_FILE
    echo "CUDA目录版本:" >> $OUTPUT_FILE
    cat /usr/local/cuda/version.txt >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

echo "## 五、Python信息" >> $OUTPUT_FILE
if command -v python3 >/dev/null 2>&1; then
    python3 --version >> $OUTPUT_FILE
    echo "路径: $(which python3)" >> $OUTPUT_FILE
else
    echo "未检测到 python3" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

echo "## 六、Python库信息" >> $OUTPUT_FILE

python3 - <<EOF >> $OUTPUT_FILE
def safe_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None

print("### PyTorch")
torch = safe_import("torch")
if torch:
    print("版本:", torch.__version__)
    print("CUDA版本:", torch.version.cuda)
    print("CUDA是否可用:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU数量:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  显存: {props.total_memory / 1024**3:.2f} GB")
else:
    print("未安装 PyTorch")

print()

print("### Transformers")
transformers = safe_import("transformers")
if transformers:
    print("版本:", transformers.__version__)
else:
    print("未安装 Transformers")
EOF

echo "" >> $OUTPUT_FILE
echo "✅ 环境信息已写入 $OUTPUT_FILE"

