# PReNet_paddle 飞桨训推一体认证（TIPC）

## 1. 简介
飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）旨在建立模型从学术研究到产业落地的桥梁，方便模型更广泛的使用。您可以从[飞桨训推一体全流程（TIPC）开发文档](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/tipc/README.mdd)了解更多关于飞桨训推一体认证（TIPC）

## 2.本项目TIPC介绍

### 2.1 准备推理数据

训练使用14张图片，位于./RainTrainH_min中，测试使用完整的Rain100H用于后续推理过程验证

### 2.2 准备推理模型

模型动转静方法可以将训练得到的动态图模型转化为用于推理的静态图模型，本小节代码位于tools/export_model.py


### 2.3 准备推理所需代码

基于预测引擎的推理过程包含4个步骤：初始化预测引擎、预处理、推理、后处理,本小节代码位于tools/infer.py

### 2.4 配置文件和测试文档

在repo根目录下面新建`test_tipc`文件夹，目录结构如下所示。

```
test_tipc
    |--configs                              # 配置目录
    |    |--PReNet                      # 您的模型名称
    |           |--train_infer_python.txt   # 基础训练推理测试配置文件
    |----README.md                          # TIPC说明文档
    |----test_train_inference_python.sh     # TIPC基础训练推理测试解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
 ```
 
 修改train_infer_python.txt里面的参数完成所有配置
 
 ## 3.使用
 
 在PReNet_paddle目录下
 运行
 ```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/PReNet/train_infer_python.txt lite_train_lite_infer
```

## 4.运行结果
```
 Run successfully with command - python3.7 train.py --output-dir=./log/PReNet/lite_train_lite_infer/norm_train_gpus_0 --epochs=1   --batch-size=32! 
 Run successfully with command - python3.7 test.py --data_path ./Rain100H  --pretrained=./log/PReNet/lite_train_lite_infer/norm_train_gpus_0/net_latest.pdparams! 
 Run successfully with command - python3.7 tools/export_model.py  --pretrained=./log/PReNet/lite_train_lite_infer/norm_train_gpus_0/net_latest.pdparams --save-inference-dir=./log/PReNet/lite_train_lite_infer/norm_train_gpus_0!  
(1, 3, 224, 224)
image_name: ./data/rain-001.png,, prob_shape: (3, 224, 224)
 Run successfully with command - python3.7 tools/infer.py --use-gpu=True --model-dir=./log/PReNet/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=False > ./log/PReNet/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !
 ```
