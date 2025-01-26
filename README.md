### 1. 软件环境
```
Ubuntu: 22.04
CUDA: 11.6/12.1
CUDNN: 8
```

[//]: # (### 2. 配置虚拟环境并激活)

[//]: # (```)

[//]: # (conda env create -f environment.yml)

[//]: # (conda activate avatar_migu)

[//]: # (```)

[//]: # (### 3. 安装nvdiffrast)

[//]: # (```)

[//]: # (cd nvdiffrast)

[//]: # (pip install .)

[//]: # (cd ..)

[//]: # (```)

### 4. 模型推理
```
python main.py --inputDir ./input --outputDir ./output
```
* 推理结果在output文件夹
* 注意：输入仅限后缀名为jpg或者png的图片，并将其放入input文件夹内

### 5. 测试硬件环境
* 硬件：Intel(R) Core(TM) i9-12900K, 显卡RTX 3090Ti
