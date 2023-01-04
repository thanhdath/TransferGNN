Code for paper "Nature vs. Nurture: Feature vs. Structure for Graph Neural Networks"[1].

## Install Libs

```
conda create -p .env/ python=3.7
conda activate .env/

CUDA=cu101
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

pip install -r requirements.txt
```

[1] Thang, Duong Chi, Hoang Thanh Dat, Nguyen Thanh Tam, Jun Jo, Nguyen Quoc Viet Hung, and Karl Aberer. "Nature vs. Nurture: Feature vs. Structure for Graph Neural Networks." Pattern Recognition Letters 159 (2022): 46-53.
