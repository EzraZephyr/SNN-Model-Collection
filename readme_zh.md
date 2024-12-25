# SNN 模型合集

[English](readme_zh)  /  中文

↑ Click the link above to switch languages

## 项目简介

⚠️⚠️⚠️ **这是一个玩具项目！**

以下是项目的主要内容概述

- **LIF 模型**: 实现了一个采用 **Sigmoid替代梯度** 的 **MNIST手写数字分类** 分类准确率达到 **98%**; 和一个基于 **STDP** 固定时间步脉冲模型 在预测固定倍数时间脉冲时 准确率保持在 **99%** 以上
- **AdEx 模型**: 在实现单个神经元模拟的基础上测试了总能量消耗和双层网络构建
- **其他模型**: 包括 HH、IF、Izhikevich, ML 和 SRM 模型 这些模型主要实现了单个神经元的行为模拟

## 目录

- [多语言注释](#多语言注释)
- [文件结构](#文件结构)
- [许可证](#许可证)
- [贡献](#贡献)


## 多语言注释

本项目的注释提供了英文和中文两种版本

## 文件结构

项目的文件结构如下

```c++
SNN_Model_Collection/
│
├── AdEx/ 
│   └── en/zh
│       ├── basic.ipynb
│       ├── consumption.ipynb
│       └── double_layer.ipynb
│
├── HH/ 
│   └── en/zh
│       └── basic.ipynb
│
├── IF/ 
│   └── IF.py
│
├── Izhikevich/ 
│   └── en/zh
│       └── basic.ipynb
│
├── LIF/ 
│   ├── LIF_MNIST/
│   │   ├── data/
│   │   │   └── MNIST/
│   │   ├── model/
│   │   │   └── model.pt
│   │   ├── utils(en/zh)/
│   │   │   ├── __init__.py
│   │   │   ├── snn.py
│   │   │   └── train.py
│   │   └── main.py
│   └── en/zh
│       ├── basic.ipynb
│       ├── double_layer.ipynb
│       ├── LIF_STDP.ipynb
│       └── signal_to_spike.ipynb
│
├── ML/ 
│   └── en/zh
│       └── basic.ipynb
│
├── SRM/ 
│   └── en/zh
│       └── basic.ipynb
│
├── LICENSE
├── main
├── readme.md
└── requirements.txt 
```

## 许可证

本项目使用 MIT 许可证 有关详细信息 请参阅 [LICENSE](LICENSE) 文件

## 贡献

欢迎所有形式的贡献！无论是报告错误还是提出建议 非常感谢！！
