import torch
from torch import nn
import torch.nn.functional as F
import einops


#### 여기부터 SeqGeM ####

# B는 batch_size
# D는 Dimension
# SL은 Sequence Length

def seq_gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    B, D, SL = x.shape
    return F.avg_pool1d(x.clamp(min=eps).pow(p), SL).pow(1./p)
    # clamp(min) 함수는 텐서 내부에서 min보다 작은 값을 min으로 교체하는 함수
    # 그래서 위에서 텐서 x 내부의 값이 eps(=1e-6)보다 작으면 eps로 변경함.
    # F.avg_pool1d(x, SL)을 통해서 수식에서 1/L 과 Sum 부분을 처리

class SeqGeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        B, SL, D = x.shape
        x = einops.rearrange(x, "b sl d -> b d sl") # 여기는 그냥 순서 바꿔주는 것 같고
        x = seq_gem(x, p=self.p, eps=self.eps)      # 그대로 seq_gem 함수에 넣어줌
        assert x.shape == torch.Size([B, D, 1]), f"{x.shape}"   # seq_gem 함수를 거치면, SL=1이 되어야 함
        return x[:, :, 0]
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class JistModel(nn.Module):
    def __init__(self, args, agg_type="concat"):
        super().__init__()
        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                                       backbone=args.backbone, fc_output_dim=args.fc_output_dim)
        for name, param in self.model.named_parameters():
            if name.startswith("backbone.7"):  # Train only last residual block
                break
            param.requires_grad = False
        assert name.startswith("backbone.7"), "are you using a resnet? this only work with resnets"
        
        self.features_dim = self.model.aggregation[3].in_features
        self.fc_output_dim = self.model.aggregation[3].out_features
        self.seq_length = args.seq_length
        if agg_type == "concat":
            self.aggregation_dim = self.fc_output_dim * args.seq_length
        if agg_type == "mean":
            self.aggregation_dim = self.fc_output_dim
        if agg_type == "max":
            self.aggregation_dim = self.fc_output_dim
        if agg_type == "conv1d":
            self.conv1d = torch.nn.Conv1d(self.fc_output_dim, self.fc_output_dim, self.seq_length)
            self.aggregation_dim = self.fc_output_dim
        if agg_type in ["simplefc", "meanfc"]:
            self.aggregation_dim = self.fc_output_dim
            self.final_fc = torch.nn.Linear(self.fc_output_dim * args.seq_length, self.fc_output_dim, bias=False)
        if agg_type == "meanfc":
            # Initialize as a mean pooling over the frames
            weights = torch.zeros_like(self.final_fc.weight)
            for i in range(self.fc_output_dim):
                for j in range(args.seq_length):
                    weights[i, j * self.fc_output_dim + i] = 1 / args.seq_length
            self.final_fc.weight = torch.nn.Parameter(weights)
        if agg_type == "seqgem":
            self.aggregation_dim = self.fc_output_dim
            self.seq_gem = SeqGeM()
        
        self.agg_type = agg_type
        
    def forward(self, x):
        return self.model(x)
    
    def aggregate(self, frames_features):
        if self.agg_type == "concat":
            concat_features = einops.rearrange(frames_features, "(b sl) d -> b (sl d)", sl=self.seq_length)
            return concat_features
        if self.agg_type == "mean":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return aggregated_features.mean(1)
        if self.agg_type == "max":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return aggregated_features.max(1)[0]
        if self.agg_type == "conv1d":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            aggregated_features = einops.rearrange(aggregated_features, "b sl d -> b d sl", sl=self.seq_length)
            features = self.conv1d(aggregated_features)
            if len(features.shape) > 2 and features.shape[2] == 1:
                features = features[:, :, 0]
            return features
        if self.agg_type in ["simplefc", "meanfc"]:
            concat_features = einops.rearrange(frames_features, "(b sl) d -> b (sl d)", sl=self.seq_length)
            return self.final_fc(concat_features)
        if self.agg_type == "seqgem":
            aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
            return self.seq_gem(aggregated_features)
