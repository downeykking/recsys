# RecSys in Pytorch
### Introduction

用pytorch复现经典的ctr模型和推荐系统模型

### Models

1. 实现了[MF(Matrix Factorization)](https://ieeexplore.ieee.org/document/5197422)
2. 实现了[FM(Factorization machines)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)，三个版本：一个与原文公式相同使用v作为嵌入参数，一个只使用sparse进行二阶交叉，一个sparse加入dense层进行二阶交叉。
3. 实现了[Wide and Deep](https://arxiv.org/abs/1606.07792)，wide层concat了sparse和dense层之后全连接激活并dropout输出，deep层输入sparse embed为k维(bs, k)并concat为(bs, field x k)后再concat dense层(bs, dense_dim)变为(bs, field x k + dense_dim)再接dnn层，之后wide+dnn层共同输出logit。
4. 实现了[DeepFM](https://arxiv.org/abs/1703.04247)，fm一阶为线性部分，线性部分sparese和dense直接全连接激活后dropout输出；fm层输入sparse embed为k维并concat为(bs, field, k)后二阶交叉生成(bs, 1)的输出，fm的embed层生成的(bs, field, k)转化为(bs, field x k)后再concat dense层(bs, dense_dim)变为(bs, field x k + dense_dim)作为dnn层的输入，之后fm层+dnn层共同输出logit。
5. 实现了[AFM](https://www.ijcai.org/proceedings/2017/0435.pdf)，线性部分sparese和dense直接全连接激活后dropout输出，attention层输入sparse embed为k维的一个list of (bs, 1, k)之后再做sparse层二阶交叉的attention，之后线性层+attention层共同输出logit。
6. 实现了[NFM](https://arxiv.org/abs/1708.05027#:~:text=NFM%20seamlessly%20combines%20the%20linearity,of%20NFM%20without%20hidden%20layers.)，线性部分sparese和dense直接全连接激活后dropout输出，nfm层输入sparse embed为k维并concat为(bs, field, k)后通过bi-interaction生成(bs, k)的输入，再concat dense层的输入变为(bs, k+dense_dim)后接入一个dnn层输出。之后线性层+dnn层共同输出logit。

### Run

```
eg:
~$: git clone https://github.com/downeykking/recsys.git
~$: cd recsys/fm
~$: run main.py
```

### References

[https://github.com/huangjunheng/recommendation_model](https://github.com/huangjunheng/recommendation_model)

[https://github.com/shawroad/DeepCTR-pytorch](https://github.com/shawroad/DeepCTR-pytorch)

[https://github.com/shenweichen/DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)