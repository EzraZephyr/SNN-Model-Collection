# SNN Model Collection

English  /  [中文](readme_zh.md)

↑ 点击切换语言

## Project Overview

⚠️⚠️⚠️ **This is a toy project!**

Below is a brief overview of the main contents of the project:

- **LIF Model**: Implemented a **MNIST classification** model using **Sigmoid as a surrogate gradient**, achieving a classification accuracy of **98%**. Additionally, developed a spike-based model using **STDP** with fixed time steps, maintaining a prediction accuracy above **99%** for fixed-multiple-time spikes.  
- **AdEx Model**: On the basis of implementing single neuron simulation, tested total energy consumption and constructed a two-layer network.
- **Other Models**: Including HH, IF, ML, Izhikevich and SRM models, which primarily simulate the behavior of single neurons.


## Table of Contents

- [Multilingual Comments](#multilingual-comments)
- [File Structure](#file-structure)
- [License](#license)
- [Contributions](#contributions)

## Multilingual Comments

This project provides comments in both English and Chinese.

## File Structure

The structure of the project files is as follows:

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

## License

This project is licensed under the MIT License. For more details, please refer to the [LICENSE](LICENSE) file.

## Contributions

All forms of contributions are welcome! Whether it's reporting bugs or making suggestions, we greatly appreciate it!!
