# model.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import math
try:
    from . import config
except ImportError:
    import config

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # 扩展维度以匹配多头
            scores.masked_fill_(mask == 0, -1e10)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 输出投影和残差连接
        output = self.W_o(context)
        return self.layer_norm(output + query), attention_weights.mean(dim=1)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, hidden_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, hidden_dim, ff_dim=None, dropout=0.1):
        super(FeedForward, self).__init__()
        if ff_dim is None:
            ff_dim = hidden_dim * 4
            
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x + residual)

class ActorCritic(nn.Module):
    """
    改进的基于Transformer的Actor-Critic网络
    集成多头注意力、位置编码、残差连接和层归一化
    """
    def __init__(self, state_dim, action_dim, hidden_dim=config.HIDDEN_DIM, num_heads=4, num_layers=2, dropout=0.1):
        super(ActorCritic, self).__init__()
        self.device = config.DEVICE
        self.hidden_dim = hidden_dim
        
        # --- 改进的Actor Network ---
        # 状态编码器（更深层的网络）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        ).to(self.device)
        
        # 对偶价格特征提取器（处理新增的7维特征）
        self.dual_price_extractor = nn.Sequential(
            nn.Linear(7, hidden_dim // 4),  # 7维伪对偶价格特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        ).to(self.device)
        
        # 基础特征提取器（处理原有的13维特征）
        self.base_feature_extractor = nn.Sequential(
            nn.Linear(13, hidden_dim // 2),  # 13维基础特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        ).to(self.device)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 4 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim).to(self.device)
        
        # 多层多头注意力
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ]).to(self.device)
        
        # 前馈网络层
        self.ff_layers = nn.ModuleList([
            FeedForward(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ]).to(self.device)
        
        # 最终的动作评分层（增强版）
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        ).to(self.device)
        
        # 对偶价格感知的价值评估头
        self.dual_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        ).to(self.device)
        
        # --- 改进的Critic Network ---
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        ).to(self.device)
        
        # 状态-动作交互层（用于更好的价值估计）
        self.state_action_critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # 自适应权重网络
        self.weight_adapter = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim // 4),  # state + score_stats
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),  # 输出alpha和beta的logits
            nn.Softmax(dim=-1)  # 确保权重和为1
        ).to(self.device)
        
    def forward(self, state, candidate_actions, action_mask):
        """
        改进的前向传播，使用多头注意力和Transformer架构
        
        参数:
        - state (Tensor): 形状为 (batch_size, state_dim) 的状态张量
        - candidate_actions (Tensor): 形状为 (batch_size, num_candidates, action_dim) 的候选动作特征张量
        - action_mask (Tensor): 形状为 (batch_size, num_candidates) 的布尔张量，用于屏蔽无效动作
        
        返回:
        - dist (torch.distributions.Categorical): 动作的概率分布
        - value (Tensor): 形状为 (batch_size, 1) 的状态价值
        """
        
        # 确保所有张量都在正确的设备上
        state = state.to(self.device)
        candidate_actions = candidate_actions.to(self.device)
        action_mask = action_mask.to(self.device)
        
        batch_size, num_candidates, _ = candidate_actions.shape
        
        # --- Actor Logic ---
        # 1. 编码状态
        state_encoded = self.state_encoder(state)  # (batch_size, hidden_dim)
        
        # 2. 动作特征分离和编码（支持双层优化架构）
        # 分离基础特征（前13维）和对偶价格特征（后7维）
        base_features = candidate_actions[:, :, :13]  # (batch_size, num_candidates, 13)
        dual_price_features = candidate_actions[:, :, 13:]  # (batch_size, num_candidates, 7)
        
        # 分别编码两类特征
        base_encoded = self.base_feature_extractor(base_features)  # (batch_size, num_candidates, hidden_dim//2)
        dual_encoded = self.dual_price_extractor(dual_price_features)  # (batch_size, num_candidates, hidden_dim//4)
        
        # 特征融合
        combined_features = torch.cat([base_encoded, dual_encoded], dim=-1)  # (batch_size, num_candidates, 3*hidden_dim//4)
        actions_encoded = self.feature_fusion(combined_features)  # (batch_size, num_candidates, hidden_dim)
        
        # 2. 添加位置编码
        actions_encoded = self.pos_encoding(actions_encoded)
        
        # 3. 准备查询：状态作为查询，动作作为键和值
        query = state_encoded.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # 4. 多层注意力处理
        attention_output = actions_encoded
        all_attention_weights = []
        
        for attention_layer, ff_layer in zip(self.attention_layers, self.ff_layers):
            # 自注意力：动作之间的关系
            attention_output, attn_weights = attention_layer(
                attention_output, attention_output, attention_output, action_mask
            )
            all_attention_weights.append(attn_weights)
            
            # 前馈网络
            attention_output = ff_layer(attention_output)
        
        # 5. 状态-动作交互注意力
        # 使用状态作为查询，处理后的动作作为键和值
        final_attention, final_weights = self.attention_layers[0](
            query, attention_output, attention_output, action_mask
        )
        
        # 6. 双头动作评分计算
        # 主要动作分数
        action_scores = self.action_scorer(attention_output).squeeze(-1)  # (batch_size, num_candidates)
        
        # 对偶价格感知的价值评估
        dual_values = self.dual_value_head(attention_output).squeeze(-1)  # (batch_size, num_candidates)
        
        # 自适应权重调整机制
        alpha, beta = self._compute_adaptive_weights(action_scores, dual_values, state)
        combined_scores = alpha * action_scores + beta * dual_values
        
        # 7. 应用动作掩码
        combined_scores = combined_scores.masked_fill(action_mask == 0, -1e10)
        
        # 8. 改进的数值稳定性处理和动作概率分布生成
        # 检查是否有有效动作
        valid_actions_count = action_mask.sum(dim=1)  # (batch_size,)
        
        # 处理没有有效动作的情况
        if torch.any(valid_actions_count == 0):
            # 创建确定性分布：选择第一个动作（如果存在）
            action_probs = torch.zeros_like(combined_scores)
            action_probs[:, 0] = 1.0  # 选择第一个动作
        else:
            # 正常情况下的改进softmax
            # 1. 数值稳定性：减去最大值
            masked_scores = combined_scores.masked_fill(action_mask == 0, -1e10)
            max_scores = torch.max(masked_scores, dim=1, keepdim=True)[0]
            stable_scores = masked_scores - max_scores
            
            # 2. 温度缩放以提高数值稳定性
            temperature = 1.0
            stable_scores = stable_scores / temperature
            
            # 3. 计算softmax
            action_probs = F.softmax(stable_scores, dim=-1)
            
            # 4. 检查并处理异常值
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print("Warning: NaN/Inf detected in action probabilities, using greedy selection")
                # 使用贪心策略：选择得分最高的有效动作
                action_probs = torch.zeros_like(combined_scores)
                best_actions = torch.argmax(masked_scores, dim=1)
                action_probs.scatter_(1, best_actions.unsqueeze(1), 1.0)
        
        # 最终的数值稳定性检查
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        
        try:
            dist = Categorical(probs=action_probs)
        except ValueError as e:
            print(f"Warning: Attention scoring failed: {e}. Using deterministic order.")
            # 创建确定性分布：选择第一个有效动作
            deterministic_probs = torch.zeros_like(action_probs)
            for i in range(batch_size):
                valid_indices = torch.where(action_mask[i])[0]
                if len(valid_indices) > 0:
                    deterministic_probs[i, valid_indices[0]] = 1.0
                else:
                    deterministic_probs[i, 0] = 1.0  # 如果没有有效动作，选择第一个
            dist = Categorical(probs=deterministic_probs)
        
        # --- Critic Logic ---
        # 基础状态价值
        state_value = self.critic(state)
        
        # 增强的状态-动作价值（使用注意力加权的动作信息）
        if num_candidates > 0:
            # 使用注意力权重加权动作特征
            weighted_actions = torch.sum(attention_output * action_probs.unsqueeze(-1), dim=1)
            state_action_features = torch.cat([state_encoded, weighted_actions], dim=-1)
            enhanced_value = self.state_action_critic(state_action_features)
            
            # 结合两种价值估计
            value = 0.7 * state_value + 0.3 * enhanced_value
        else:
            value = state_value
        
        return dist, value
    
    def _compute_adaptive_weights(self, action_scores, dual_values, state):
        """
        自适应计算双头权重
        
        参数:
        - action_scores: 主要动作评分
        - dual_values: 对偶价格评分
        - state: 当前状态
        
        返回:
        - alpha, beta: 自适应权重
        """
        batch_size = action_scores.shape[0]
        
        # 计算评分统计特征，添加数值稳定性
        action_std = torch.std(action_scores, dim=1, keepdim=True)  # (batch_size, 1)
        dual_std = torch.std(dual_values, dim=1, keepdim=True)      # (batch_size, 1)
        
        # 处理标准差为0的情况（所有值相同）
        action_std = torch.clamp(action_std, min=1e-8)
        dual_std = torch.clamp(dual_std, min=1e-8)
        
        # 检查并处理NaN值
        if torch.isnan(action_std).any() or torch.isnan(dual_std).any():
            print("Warning: NaN detected in standard deviation calculation")
            action_std = torch.ones_like(action_std) * 1e-8
            dual_std = torch.ones_like(dual_std) * 1e-8
        
        # 构建权重网络输入
        weight_input = torch.cat([
            state,           # 状态信息
            action_std,      # 主要评分的方差（反映决策确定性）
            dual_std         # 对偶价格评分的方差
        ], dim=1)  # (batch_size, state_dim + 2)
        
        # 检查输入是否包含NaN
        if torch.isnan(weight_input).any():
            print("Warning: NaN detected in weight adapter input, using default weights")
            alpha = torch.ones(batch_size, 1, device=self.device) * 0.7
            beta = torch.ones(batch_size, 1, device=self.device) * 0.3
            return alpha, beta
        
        try:
            # 计算自适应权重
            weights = self.weight_adapter(weight_input)  # (batch_size, 2)
            alpha = weights[:, 0].unsqueeze(1)  # (batch_size, 1)
            beta = weights[:, 1].unsqueeze(1)   # (batch_size, 1)
            
            # 确保权重有效
            if torch.isnan(alpha).any() or torch.isnan(beta).any():
                print("Warning: NaN detected in adaptive weights, using default values")
                alpha = torch.ones(batch_size, 1, device=self.device) * 0.7
                beta = torch.ones(batch_size, 1, device=self.device) * 0.3
            
            return alpha, beta
        except Exception as e:
            print(f"Warning: Error in adaptive weight computation: {e}, using default weights")
            alpha = torch.ones(batch_size, 1, device=self.device) * 0.7
            beta = torch.ones(batch_size, 1, device=self.device) * 0.3
            return alpha, beta
    
    def get_attention_weights(self, state, candidate_actions, action_mask):
        """获取注意力权重用于可视化和分析"""
        with torch.no_grad():
            state = state.to(self.device)
            candidate_actions = candidate_actions.to(self.device)
            action_mask = action_mask.to(self.device)
            
            state_encoded = self.state_encoder(state)
            actions_encoded = self.action_encoder(candidate_actions)
            actions_encoded = self.pos_encoding(actions_encoded)
            
            attention_weights = []
            attention_output = actions_encoded
            
            for attention_layer, ff_layer in zip(self.attention_layers, self.ff_layers):
                attention_output, attn_weights = attention_layer(
                    attention_output, attention_output, attention_output, action_mask
                )
                attention_weights.append(attn_weights)
                attention_output = ff_layer(attention_output)
            
            return attention_weights