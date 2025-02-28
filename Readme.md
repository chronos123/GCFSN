## Pytorch implement for Coupled multiphysics solver for irregular regions based on graph neural network

paper link: https://www.sciencedirect.com/science/article/pii/S266620272400168X

### Enviromnet
python=3.8.8

```py
pip install -r requirements.txt
```

### Example data

The data file dataGNN4.mat is available at https://pan.baidu.com/s/1tUXbUczqjI8Ih8NrunqZpA?pwd=nf7t 提取码: nf7t.

This data corresponds to experiment (4) in Table 2 of the manuscript, solving the Temperature-electric coupled filed.

You should download the data and put them under the dataset directory.

Others comming soon.

### Testing 

```sh
python test.py --exp-name 4 --data-path dataset/dataGNN4.mat --ckpt ckpt/4.pkl
```

### Training

```sh
python train.py --data-path dataset/dataGNN4.mat
```

### New Graph

Modify the create_graph.py

### Citation
```
@article{SUN2024100726,
title = {Coupled multiphysics solver for irregular regions based on graph neural network},
journal = {International Journal of Thermofluids},
volume = {23},
pages = {100726},
year = {2024},
issn = {2666-2027},
doi = {https://doi.org/10.1016/j.ijft.2024.100726},
author = {Xiancheng Sun and Borui Du and Yinpeng Wang and Qiang Ren},
}
```

