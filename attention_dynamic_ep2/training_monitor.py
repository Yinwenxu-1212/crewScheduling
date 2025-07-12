import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import json
import time
from datetime import datetime

class TrainingMonitor:
    """
    双头架构训练监控器
    监控训练过程中的关键指标和双头架构特定的性能
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            # 基础训练指标
            'episode_rewards': deque(maxlen=window_size),
            'episode_lengths': deque(maxlen=window_size),
            'actor_losses': deque(maxlen=window_size),
            'critic_losses': deque(maxlen=window_size),
            
            # 双头架构特定指标
            'dual_price_contributions': deque(maxlen=window_size),
            'base_score_contributions': deque(maxlen=window_size),
            'adaptive_weights_alpha': deque(maxlen=window_size),
            'adaptive_weights_beta': deque(maxlen=window_size),
            
            # 约束和质量指标
            'constraint_violations': deque(maxlen=window_size),
            'solution_quality': deque(maxlen=window_size),
            'coverage_rates': deque(maxlen=window_size),
            
            # 特征重要性
            'feature_importance': defaultdict(list),
            'dual_price_feature_stats': defaultdict(list)
        }
        
        self.episode_count = 0
        self.start_time = time.time()
        self.best_reward = float('-inf')
        self.best_model_state = None
        
    def log_episode(self, episode_data):
        """
        记录单个episode的数据
        
        参数:
        - episode_data: 包含episode信息的字典
        """
        self.episode_count += 1
        
        # 基础指标
        reward = episode_data.get('total_reward', 0)
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(episode_data.get('episode_length', 0))
        
        # 损失
        self.metrics['actor_losses'].append(episode_data.get('actor_loss', 0))
        self.metrics['critic_losses'].append(episode_data.get('critic_loss', 0))
        
        # 双头架构指标
        self.metrics['dual_price_contributions'].append(
            episode_data.get('avg_dual_contribution', 0)
        )
        self.metrics['base_score_contributions'].append(
            episode_data.get('avg_base_contribution', 0)
        )
        self.metrics['adaptive_weights_alpha'].append(
            episode_data.get('avg_alpha', 0.7)
        )
        self.metrics['adaptive_weights_beta'].append(
            episode_data.get('avg_beta', 0.3)
        )
        
        # 约束和质量
        self.metrics['constraint_violations'].append(
            episode_data.get('constraint_violations', 0)
        )
        self.metrics['solution_quality'].append(
            episode_data.get('solution_quality', 0)
        )
        self.metrics['coverage_rates'].append(
            episode_data.get('coverage_rate', 0)
        )
        
        # 更新最佳模型
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_model_state = episode_data.get('model_state_dict')
            
    def log_dual_price_features(self, feature_stats):
        """
        记录伪对偶价格特征的统计信息
        
        参数:
        - feature_stats: 特征统计字典
        """
        for feature_name, value in feature_stats.items():
            self.metrics['dual_price_feature_stats'][feature_name].append(value)
            
    def analyze_feature_importance(self, model, sample_data):
        """
        分析特征重要性
        
        参数:
        - model: 训练的模型
        - sample_data: 样本数据
        """
        model.eval()
        with torch.no_grad():
            state, actions, mask = sample_data
            
            # 获取基线预测
            baseline_dist, baseline_value = model(state, actions, mask)
            baseline_probs = baseline_dist.probs
            
            # 分析每个特征维度的重要性
            feature_importance = {}
            
            # 分析基础特征（前13维）
            for i in range(13):
                perturbed_actions = actions.clone()
                perturbed_actions[:, :, i] = 0  # 置零该特征
                
                perturbed_dist, _ = model(state, perturbed_actions, mask)
                perturbed_probs = perturbed_dist.probs
                
                # 计算KL散度作为重要性指标
                kl_div = torch.nn.functional.kl_div(
                    torch.log(perturbed_probs + 1e-8),
                    baseline_probs,
                    reduction='mean'
                )
                feature_importance[f'base_feature_{i}'] = kl_div.item()
                
            # 分析对偶价格特征（后7维）
            for i in range(7):
                perturbed_actions = actions.clone()
                perturbed_actions[:, :, 13 + i] = 0  # 置零该特征
                
                perturbed_dist, _ = model(state, perturbed_actions, mask)
                perturbed_probs = perturbed_dist.probs
                
                kl_div = torch.nn.functional.kl_div(
                    torch.log(perturbed_probs + 1e-8),
                    baseline_probs,
                    reduction='mean'
                )
                feature_importance[f'dual_price_feature_{i}'] = kl_div.item()
                
            # 记录特征重要性
            for feature, importance in feature_importance.items():
                self.metrics['feature_importance'][feature].append(importance)
                
        model.train()
        return feature_importance
        
    def get_current_stats(self):
        """
        获取当前训练统计信息
        """
        if not self.metrics['episode_rewards']:
            return {}
            
        stats = {
            'episode_count': self.episode_count,
            'training_time': time.time() - self.start_time,
            'best_reward': self.best_reward,
            
            # 最近性能
            'recent_avg_reward': np.mean(list(self.metrics['episode_rewards'])[-20:]),
            'recent_avg_length': np.mean(list(self.metrics['episode_lengths'])[-20:]),
            
            # 双头架构性能
            'avg_alpha_weight': np.mean(list(self.metrics['adaptive_weights_alpha'])[-20:]),
            'avg_beta_weight': np.mean(list(self.metrics['adaptive_weights_beta'])[-20:]),
            'dual_price_contribution': np.mean(list(self.metrics['dual_price_contributions'])[-20:]),
            
            # 约束遵守情况
            'avg_constraint_violations': np.mean(list(self.metrics['constraint_violations'])[-20:]),
            'avg_coverage_rate': np.mean(list(self.metrics['coverage_rates'])[-20:]),
        }
        
        return stats
        
    def plot_training_progress(self, save_path=None):
        """
        绘制训练进度图表
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('双头架构训练监控', fontsize=16)
        
        # 奖励曲线
        axes[0, 0].plot(list(self.metrics['episode_rewards']))
        axes[0, 0].set_title('Episode奖励')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('奖励')
        
        # 损失曲线
        axes[0, 1].plot(list(self.metrics['actor_losses']), label='Actor Loss')
        axes[0, 1].plot(list(self.metrics['critic_losses']), label='Critic Loss')
        axes[0, 1].set_title('训练损失')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].legend()
        
        # 自适应权重
        axes[0, 2].plot(list(self.metrics['adaptive_weights_alpha']), label='Alpha (主要评分)')
        axes[0, 2].plot(list(self.metrics['adaptive_weights_beta']), label='Beta (对偶价格)')
        axes[0, 2].set_title('自适应权重变化')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('权重')
        axes[0, 2].legend()
        
        # 约束违反
        axes[1, 0].plot(list(self.metrics['constraint_violations']))
        axes[1, 0].set_title('约束违反次数')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('违反次数')
        
        # 覆盖率
        axes[1, 1].plot(list(self.metrics['coverage_rates']))
        axes[1, 1].set_title('任务覆盖率')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('覆盖率')
        
        # 双头贡献度对比
        axes[1, 2].plot(list(self.metrics['base_score_contributions']), label='基础评分贡献')
        axes[1, 2].plot(list(self.metrics['dual_price_contributions']), label='对偶价格贡献')
        axes[1, 2].set_title('双头贡献度对比')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('贡献度')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_training_log(self, filepath):
        """
        保存训练日志到文件
        """
        log_data = {
            'episode_count': self.episode_count,
            'training_time': time.time() - self.start_time,
            'best_reward': self.best_reward,
            'metrics': {k: list(v) for k, v in self.metrics.items() if not isinstance(v, defaultdict)},
            'feature_importance': {k: list(v) for k, v in self.metrics['feature_importance'].items()},
            'dual_price_stats': {k: list(v) for k, v in self.metrics['dual_price_feature_stats'].items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
    def print_progress_summary(self):
        """
        打印训练进度摘要
        """
        stats = self.get_current_stats()
        
        print(f"\n=== 双头架构训练进度摘要 ===")
        print(f"Episode: {stats.get('episode_count', 0)}")
        print(f"训练时间: {stats.get('training_time', 0):.2f}秒")
        print(f"最佳奖励: {stats.get('best_reward', 0):.4f}")
        print(f"最近平均奖励: {stats.get('recent_avg_reward', 0):.4f}")
        print(f"\n--- 双头架构性能 ---")
        print(f"平均Alpha权重: {stats.get('avg_alpha_weight', 0):.3f}")
        print(f"平均Beta权重: {stats.get('avg_beta_weight', 0):.3f}")
        print(f"对偶价格贡献度: {stats.get('dual_price_contribution', 0):.3f}")
        print(f"\n--- 约束遵守情况 ---")
        print(f"平均约束违反: {stats.get('avg_constraint_violations', 0):.2f}")
        print(f"平均覆盖率: {stats.get('avg_coverage_rate', 0):.3f}")
        print(f"="*40)

class FeatureAnalyzer:
    """
    特征重要性分析器
    专门分析双头架构中各特征的贡献度
    """
    
    def __init__(self):
        self.feature_names = {
            # 基础特征（0-12）
            0: '任务类型编码',
            1: '起飞机场编码', 
            2: '到达机场编码',
            3: '计划起飞时间',
            4: '计划到达时间',
            5: '飞行时长',
            6: '机组基地匹配度',
            7: '时间窗口匹配度',
            8: '连接可行性',
            9: '工作负载平衡',
            10: '约束余量',
            11: '历史偏好',
            12: '全局覆盖率',
            
            # 对偶价格特征（13-19）
            13: '任务稀缺性评分',
            14: '时间紧迫性评分', 
            15: '立即价值评分',
            16: '机组负载平衡评分',
            17: '全局覆盖压力',
            18: '连接效率评分',
            19: '约束风险评分'
        }
        
    def analyze_feature_correlation(self, feature_data):
        """
        分析特征间的相关性
        
        参数:
        - feature_data: 特征数据矩阵 (batch_size, num_actions, feature_dim)
        """
        # 重塑数据为 (samples, features)
        reshaped_data = feature_data.view(-1, feature_data.shape[-1]).cpu().numpy()
        
        # 计算相关性矩阵
        correlation_matrix = np.corrcoef(reshaped_data.T)
        
        return correlation_matrix
        
    def plot_feature_importance(self, importance_dict, save_path=None):
        """
        绘制特征重要性图表
        """
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        # 分离基础特征和对偶价格特征
        base_features = [(f, imp) for f, imp in zip(features, importances) if 'base_feature' in f]
        dual_features = [(f, imp) for f, imp in zip(features, importances) if 'dual_price_feature' in f]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 基础特征重要性
        if base_features:
            base_names = [self.feature_names.get(int(f.split('_')[-1]), f) for f, _ in base_features]
            base_imps = [imp for _, imp in base_features]
            
            ax1.barh(base_names, base_imps)
            ax1.set_title('基础特征重要性')
            ax1.set_xlabel('重要性分数')
            
        # 对偶价格特征重要性
        if dual_features:
            dual_names = [self.feature_names.get(13 + int(f.split('_')[-1]), f) for f, _ in dual_features]
            dual_imps = [imp for _, imp in dual_features]
            
            ax2.barh(dual_names, dual_imps, color='orange')
            ax2.set_title('对偶价格特征重要性')
            ax2.set_xlabel('重要性分数')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_feature_report(self, importance_data, correlation_matrix=None):
        """
        生成特征分析报告
        """
        report = "\n=== 双头架构特征分析报告 ===\n"
        
        # 特征重要性排序
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        
        report += "\n--- 特征重要性排名 ---\n"
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            feature_idx = int(feature.split('_')[-1])
            if 'base_feature' in feature:
                feature_name = self.feature_names.get(feature_idx, feature)
            else:
                feature_name = self.feature_names.get(13 + feature_idx, feature)
            report += f"{i+1:2d}. {feature_name}: {importance:.4f}\n"
            
        # 基础特征 vs 对偶价格特征对比
        base_importance = np.mean([imp for feat, imp in importance_data.items() if 'base_feature' in feat])
        dual_importance = np.mean([imp for feat, imp in importance_data.items() if 'dual_price_feature' in feat])
        
        report += f"\n--- 特征类别对比 ---\n"
        report += f"基础特征平均重要性: {base_importance:.4f}\n"
        report += f"对偶价格特征平均重要性: {dual_importance:.4f}\n"
        report += f"对偶价格特征相对贡献: {dual_importance/(base_importance+dual_importance)*100:.1f}%\n"
        
        return report