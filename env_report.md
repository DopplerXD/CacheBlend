# 实验环境信息报告

## 一、操作系统信息
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.3 LTS
Release:	22.04
Codename:	jammy

内核版本:
Linux 43b8f2eba4bd 5.15.0-113-generic #123-Ubuntu SMP Mon Jun 10 08:16:17 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux

## 二、CPU信息
CPU(s):                             16
On-line CPU(s) list:                0-15
Model name:                         Intel(R) Xeon(R) Platinum 8160 CPU @ 2.10GHz
Thread(s) per core:                 1
Core(s) per socket:                 8
Socket(s):                          2
NUMA node0 CPU(s):                  0-15

## 三、GPU / 显存 / 驱动
```
Sat May  2 07:01:23 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:00:06.0 Off |                  N/A |
| 30%   32C    P0             60W /  350W |       0MiB /  24576MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### GPU简要信息
Field "cuda_version" is not a valid field to query.


## 四、CUDA信息
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```

## 五、Python信息
Python 3.10.19
路径: /root/yi-query-aware/venv/bin/python3

## 六、Python库信息
### PyTorch
版本: 2.4.0+cu121
CUDA版本: 12.1
CUDA是否可用: True
GPU数量: 1
GPU 0: NVIDIA GeForce RTX 3090
  显存: 23.56 GB

### Transformers
版本: 4.45.0

