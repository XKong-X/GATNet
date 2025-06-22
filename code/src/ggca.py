import torch
import torch.nn as nn

# 1. 用全连接层替换卷积层
class BasicFC(nn.Module):
    def __init__(self, in_features, out_features, relu=True, bn=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features, eps=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat((max_pool, mean_pool), dim=1)

class SpatialGate(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.compress = ChannelPool()
        self.fc = BasicFC(2, 1, relu=False, bn=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.fc(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.spatial_gate = SpatialGate(in_features)

    def forward(self, x):
        x_out = self.spatial_gate(x)
        return x_out

class GGCA(nn.Module):
    def __init__(self, in_features, gate_threshold=0.3, alpha=0.2):
        super().__init__()
        self.bn  = nn.BatchNorm1d(in_features)
        self.fc  = nn.Linear(in_features, 1)
        self.gate_threshold = gate_threshold
        self.alpha = alpha

    def forward(self, x):
        # x: (B, in_features)
        bn_x = self.bn(x)
        g = torch.sigmoid(self.fc(bn_x))            # (B,1) 0~1
        if self.training:
            gate_mask = (g >= self.gate_threshold).float()
            out = x * gate_mask + self.alpha * x * (1 - gate_mask)
        else:
            out = x * g
        return out                                  # (B, in_features)

if __name__ == "__main__":
    # 假设表格数据有10个特征，batch_size为4
    x = torch.randn(4, 10)
    model = SRM(in_features=10, gate_threshold=0.5, alpha=0.1)
    out = model(x)
    print(out.shape)  # 应输出: torch.Size([4, 10])