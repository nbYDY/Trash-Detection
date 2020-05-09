from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']  # 输入图像大小
        self.num_priors = len(cfg['aspect_ratios'])  # 使用4个还是6个不同比例的box， aspect_ratios是比例
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']  # 划分的网格数，有效特征层的大小
        self.max_size = cfg['max_size']
        self.min_size = cfg['min_size']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']  # box的比例
        self.clip = cfg['clip']
        self.version = cfg['name']  # 数据集格式
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):  # f为格子数,遍历k=6个特征层,每层都生成box
            for i, j in product(range(f), repeat=2):  # 笛卡尔积，和双重循环作用相同，遍历所有坐标
                f_k = self.image_size / self.steps[k]  # 将图像划分为同等steps大小的f_k个区域
                # 计算出网格中心点cx,cy坐标（在原图中的 ）并归一化，原始公式应该为cx = (j+0.5) * step /min_dim
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                """
                计算小正方形的边长，大正方形的边长和长方形的边长
                即两种边长比例为1和每层对应的其他比例的box
                """
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = self.max_size[k] / self.image_size
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # 将box转换为标准的n行4列形式
        output = torch.tensor(mean).view(-1, 4)
        # 限制box与原图的比例
        if self.clip:
            output.clamp_(max=1, min=0)

        return output

