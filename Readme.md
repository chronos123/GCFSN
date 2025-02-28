## Pytorch implement for Coupled multiphysics solver for irregular regions based on graph neural network

### Enviromnet
python=3.8.8

```py
pip install -r requirements.txt
```

### Example data

comming soon

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

