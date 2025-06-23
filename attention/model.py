# model.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from . import config

class ActorCritic(nn.Module):
    """
    一个基于注意力机制的Actor-Critic网络。
    Actor使用一个基于查询-键-值（Query-Key-Value）的注意力模型来从候选动作中选择一个。
    Critic评估当前状态的价值。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=config.HIDDEN_DIM):
        """
        初始化Actor和Critic网络。
        
        参数:
        - state_dim (int): 状态向量的维度。
        - action_dim (int): 单个动作（任务）特征向量的维度。
        - hidden_dim (int): 神经网络中隐藏层的维度。
        """
        super(ActorCritic, self).__init__()
        self.device = config.DEVICE

        # --- Actor Network (策略网络) ---
        # Actor网络负责决定在给定状态下应该采取哪个动作。
        
        # 1. 状态编码器: 将输入的状态向量转换为一个高维的"上下文"或"查询"向量。
        # 这个向量代表了机长当前的所有情况。
        self.actor_state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)

        # 2. 动作编码器: 将每个候选动作的特征向量也转换为高维表示。
        # 这些向量将作为注意力的"键"和"值"。
        self.actor_action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)

        # 3. 注意力层: 这是将状态和动作联系起来的关键。
        # 我们使用一个简单的线性变换来生成最终的查询向量，
        # 也可以使用更复杂的结构，但线性层通常足够。
        # 这里的注意力机制是经典的Bahdanau或Luong风格的点积注意力。
        # query (from state) @ key (from action) -> attention_score
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device) # 生成Query
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device) # 生成Key
        
        self.sqrt_hidden_dim = math.sqrt(float(hidden_dim))

        # --- Critic Network (价值网络) ---
        # Critic网络负责评估当前状态的好坏，即预测从当前状态出发能获得的未来总奖励。
        # 这是一个标准的回归网络。
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出一个标量值
        ).to(self.device)
        
    def forward(self, state, candidate_actions, action_mask):
        """
        模型的前向传播。
        
        参数:
        - state (Tensor): 形状为 (batch_size, state_dim) 的状态张量。
        - candidate_actions (Tensor): 形状为 (batch_size, num_candidates, action_dim) 的候选动作特征张量。
        - action_mask (Tensor): 形状为 (batch_size, num_candidates) 的布尔或0/1张量，用于屏蔽无效动作。
        
        返回:
        - dist (torch.distributions.Categorical): 一个动作的概率分布。
        - value (Tensor): 形状为 (batch_size, 1) 的状态价值。
        """
        
        # 确保所有张量都在正确的设备上
        state = state.to(self.device)
        candidate_actions = candidate_actions.to(self.device)
        action_mask = action_mask.to(self.device)
        
        # --- Actor Logic ---
        # 1. 编码状态以生成查询 (Query)
        state_encoded = torch.relu(self.actor_state_encoder(state))
        query = self.W_q(state_encoded).unsqueeze(1)  # 形状: (batch_size, 1, hidden_dim)

        # 2. 编码所有候选动作以生成键 (Keys)
        actions_encoded = torch.relu(self.actor_action_encoder(candidate_actions))
        keys = self.W_k(actions_encoded)  # 形状: (batch_size, num_candidates, hidden_dim)

        # 3. 计算注意力分数
        # 使用缩放点积注意力 (Scaled Dot-Product Attention)
        # (Query @ Key.T) / sqrt(d_k)
        attention_scores = torch.bmm(query, keys.transpose(1, 2)) / self.sqrt_hidden_dim
        attention_scores = attention_scores.squeeze(1) # 形状: (batch_size, num_candidates)

        # 4. 应用动作掩码
        # 将无效动作的注意力分数设置为一个非常小的负数，
        # 这样在经过softmax后，它们的概率会趋近于0。
        attention_scores[action_mask == 0] = -1e10
        
        # 5. 生成动作概率分布
        # Softmax将分数转换为概率
        action_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # 使用Categorical分布，方便后续采样和计算log_prob
        dist = Categorical(probs=action_probs)
        
        # --- Critic Logic ---
        # 评估状态价值
        value = self.critic(state) # 形状: (batch_size, 1)
        
        return dist, value