import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
import math

def constant_(tensor, value):
    nn.init.constant_(tensor, value)

def xavier_uniform_(tensor):
    nn.init.xavier_uniform_(tensor)

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_heads=26, n_points=4, active_heads=8):
        """
        单层级可变形注意力模块
        :param d_model      隐藏维度
        :param n_heads     注意力头总数
        :param n_points    每个注意力头的采样点数
        :param active_heads 激活头数 (k)
        """
        super().__init__()
        if d_model % active_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by active_heads ({active_heads})')

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.active_heads = active_heads

        # ----------------- 动态参数生成 -----------------
        # 修改1：移除层级维度乘积 (n_levels=1)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 3)  # 原为 n_heads*n_levels*n_points*3
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)     # 原为 n_heads*n_levels*n_points
        
        # ----------------- 共享参数 -----------------
        self.value_proj = nn.Linear(d_model, d_model // active_heads * n_heads)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):

        constant_(self.sampling_offsets.weight.data, 0.)

        # 面中心 (6)
        directions = [
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ]
        
        # 边中心 (12)
        directions += [
            [1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0],
            [1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1],
            [0,1,1], [0,1,-1], [0,-1,1], [0,-1,-1]
        ]
        
        # 顶点 (8)
        directions += [
            [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
            [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
        ]
        
        # 转换为张量并归一化
        grid_init = torch.tensor(directions, dtype=torch.float32)
        grid_init = grid_init.view(self.n_heads, 1, 3).repeat(1, self.n_points, 1)

        for i in range(self.n_points):
            grid_init[:, i, :] *= 0.005 * (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, all_coords, scale_ranges, reference_points, input_flatten):
        """
        参数变化：
        - 移除 split_sizes（单层级无需拆分）
        - reference_points 形状变为 [B, N, 3]
        """
        B, Len_q, _ = query.shape
        N = input_flatten.size(1)

        # 修改4：单层级注意力权重形状 [B, Len_q, n_heads, P]
        attention_weights = self.attention_weights(query).view(B, Len_q, self.n_heads, self.n_points)
        
        # 使用单层级版选择函数
        topk_indices, _ = select_topk_heads(attention_weights, k=self.active_heads)

        # 值投影（保持与原逻辑一致）
        value = self.value_proj(input_flatten).view(B, N, self.n_heads, self.d_model // self.active_heads)
        active_value = torch.gather(
            value, 
            dim=2, 
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, value.size(-1))
        )  # [B, N, active_heads, d_head]

        # 修改5：偏移量形状调整为 [B, Len_q, n_heads, P, 3]
        sampling_offsets = self.sampling_offsets(query).view(B, Len_q, self.n_heads, self.n_points, 3)
        
        # 修改6：索引扩展移除层级维度
        expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.n_points, 3)
        sampling_offsets = torch.gather(
            sampling_offsets,
            dim=2,
            index=expanded_indices
        )  # [B, Len_q, active_heads, P, 3]

        # 修改7：注意力权重调整
        expanded_attn_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.n_points)
        attention_weights = torch.gather(
            attention_weights, 
            dim=2,
            index=expanded_attn_indices
        )  # [B, Len_q, active_heads, P]
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 修改8：采样位置计算（单层级）
        sampling_locations = reference_points[:, :, None, None, :] + sampling_offsets  # [B, Len_q, active_heads, P, 3]

        # 调用单层级核心函数
        output = ms_deform_attn_core_3d_knn(
            active_value, 
            all_coords,
            sampling_locations,
            scale_ranges,
            attention_weights
        )
        return self.output_proj(output)
    

def ms_deform_attn_core_3d_knn(
    value, 
    all_coords,
    sampling_locations,  # 形状变为 (B, Lq, n_heads, num_points, 3)
    scale_ranges,        # 形状变为 (B, 2, 3)
    attention_weights,   # 形状变为 (B, Lq, n_heads, num_points)
    k=3
):
    """
    单层无维度版 3D 可变形注意力
    
    参数：
    value: 单层特征 (B, N, n_heads, d_head)
    all_coords: 绝对坐标 (B, N, 3)
    sampling_locations: 采样坐标 (B, Lq, n_heads, num_points, 3)
    scale_ranges: 坐标范围 (B, 2, 3)
    attention_weights: 注意力权重 (B, Lq, n_heads, num_points)
    k: 最近邻数量

    返回：
    聚合特征 (B, Lq, n_heads * d_head)
    """
    B, N, n_heads, d_head = value.shape
    Lq, num_points = sampling_locations.shape[1], sampling_locations.shape[3]

    # 提取归一化范围 (B, 3)
    scale_min = scale_ranges[:, 0, :]
    scale_max = scale_ranges[:, 1, :]

    # 归一化采样坐标 (B, Lq, n_heads, num_points, 3)
    norm_locs = (sampling_locations - scale_min.view(B, 1, 1, 1, 3)) / \
                (scale_max - scale_min + 1e-7).view(B, 1, 1, 1, 3)

    # 调整形状以并行处理头与批次 (B*n_heads, Lq, num_points, 3)
    norm_locs = norm_locs.transpose(1, 2).flatten(0, 1)
    level_value = value.transpose(1, 2).flatten(0, 1)  # (B*n_heads, N, d_head)

    # 归一化特征坐标 (B, N, 3)
    norm_coords = (all_coords - scale_min.unsqueeze(1)) / (scale_max - scale_min + 1e-7).unsqueeze(1)

    # 插值特征 (B*n_heads, d_head, Lq, num_points)
    interpolated = interpolate_features_optimized(norm_coords, norm_locs, level_value, k=k)

    # 注意力权重调整 (B*n_heads, 1, Lq, num_points)
    attn_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Lq, num_points)

    # 聚合与形状恢复
    output = (interpolated * attn_weights).sum(-1)  # (B*n_heads, d_head, Lq)
    output = output.view(B, n_heads * d_head, Lq)  # (B, n_heads*d_head, Lq)
    
    return output.transpose(1, 2).contiguous()  # (B, Lq, n_heads*d_head)


def select_topk_heads(
    attn_weights,  # 新形状 [B, Len_q, n_heads, P]
    k=8, 
    reduce_mode='sum'
):
    """
    单层级版：为每个样本的每个查询点选择注意力得分最高的k个头
    
    参数：
        attn_weights: [B, Len_q, n_heads, P] (移除了层级维度)
        k: 选择头数
        reduce_mode: 聚合方式 ('sum'/'mean'/'max')
        
    返回：
        topk_indices: [B, Len_q, k] 选择的头索引
        topk_scores: [B, Len_q, k] 对应的注意力分数
    """
    # 步骤1：压缩采样点维度
    if reduce_mode == 'sum':
        head_scores = attn_weights.sum(dim=-1)  # [B, Len_q, n_heads]
    elif reduce_mode == 'mean':
        head_scores = attn_weights.mean(dim=-1)
    elif reduce_mode == 'max':
        head_scores = attn_weights.amax(dim=-1)
    else:
        raise ValueError(f"Unsupported reduce mode: {reduce_mode}")

    # 步骤2：选择topk头（与原逻辑一致）
    topk_scores, topk_indices = torch.topk(
        head_scores, 
        k=k,
        dim=-1,          # 在n_heads维度选择
        largest=True,    
        sorted=True      
    )

    return topk_indices, topk_scores


def interpolate_features_optimized(src_coords, q_coords, src_features, k=3):
    """
    Args:
        src_coords: (B, M, 3)         源点坐标
        q_coords: (B * n_head, Lq, num_points, 3) 查询坐标
        src_features: (B * n_head, N_l, d_head) 源特征
        k: 最近邻数量
    Returns:
        (B, N, n_heads, num_points, d_head)
    """
    B, M, _ = src_coords.shape
    EB, N, num_points, _ = q_coords.shape
    d_head = src_features.shape[-1]
    n_heads = EB // B

    # ===== 1. 坐标处理 =====
    # 展平查询坐标 
    src_coords = src_coords.repeat(n_heads, 1, 1)
    q_coords_flat = q_coords.reshape(EB, N * num_points, -1)  # (B * n_head, Lq * num_points, 3)
    
    # ===== 2. 计算距离矩阵 =====
    dist = torch.cdist(q_coords_flat, src_coords)  # (B * n_head, Lq * num_points, M)
    
    # ===== 3. 寻找k近邻 =====
    topk_dist, topk_idx = torch.topk(dist, k=k, dim=-1, largest=False)  # (B * n_head, Lq * num_points, k)
    
    # ===== 4. 权重计算 =====
    eps = 1e-7
    weights = 1.0 / (topk_dist + eps)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # (B * n_head, Lq * num_points, k)

    # ===== 5. 特征收集 =====
    # 调整特征维度 (B * n_head, N_l, k, d_head)
    src_features = src_features.unsqueeze(2).expand(-1, -1, k, -1)
    
    # 扩展索引维度 (B * n_head, Lq * num_points, k, d_head)
    topk_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, d_head)
    
    # 收集特征 (关键修复点)
    gathered_features = torch.gather(
        src_features,
        dim=1,  # 在M维度收集
        index=topk_idx
    )  # -> (B * n_head, Lq * num_points, k, d_head)

    # ===== 6. 加权求和 =====
    gathered_features = gathered_features.view(B * n_heads, N, num_points, k, d_head)
    weights = weights.view(B * n_heads, N, num_points, k)
    interpolated = (gathered_features * weights.unsqueeze(-1)).sum(dim=-2)
    
    # ===== 7. 调整输出维度 =====
    return interpolated.permute(0, 3, 1, 2)

    