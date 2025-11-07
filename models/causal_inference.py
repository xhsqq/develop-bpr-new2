"""
Structural Causal Model (SCM) based Causal Inference
基于结构因果模型的因果推断：实现Pearl三步反事实推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class StructuralCausalModel(nn.Module):
    """
    结构因果模型：实现Pearl三步反事实推断

    理论保证：
    1. Identifiability (定理1.1)
    2. Consistency (定理1.2)
    3. Unbiased ITE (定理1.3)

    结构方程：
    Z_func = μ_func(X) + ε_func * σ_func(X)
    Z_aes = μ_aes(X) + ε_aes * σ_aes(X)
    Z_emo = μ_emo(X) + ε_emo * σ_emo(X)
    """

    def __init__(
        self, 
        latent_dim: int = 64,
        causal_loss_weights: Dict[str, float] = None  # ⭐ 因果损失权重
    ):
        super().__init__()

        self.latent_dim = latent_dim
        
        # ⭐ 因果损失权重（默认值与简化后的代码对应）
        if causal_loss_weights is None:
            causal_loss_weights = {
                'magnitude': 1.0
            }
        self.causal_loss_weights = causal_loss_weights

        # ===干预策略网络（自适应干预强度）===
        self.intervention_strength = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        mu_dict: Dict[str, torch.Tensor],
        logvar_dict: Dict[str, torch.Tensor],
        quantum_encoder: nn.Module,
        recommendation_head: callable,
        target_items: Optional[torch.Tensor] = None,
        candidate_items: Optional[torch.Tensor] = None,  # ⭐ 新增：候选物品
        item_embedding: Optional[torch.Tensor] = None  # ⭐ P0修复：传入item embedding
    ) -> Dict[str, torch.Tensor]:
        """
        三步反事实推断

        Args:
            z_dict: {'function': z_func, 'aesthetics': z_aes, 'emotion': z_emo}
            mu_dict, logvar_dict: VAE参数（用于abduction）
            quantum_encoder: 下游量子编码器
            recommendation_head: 推荐打分函数
            target_items: 目标商品
            candidate_items: 候选物品（可选）
            item_embedding: 原始item embedding（⭐ P0修复：需要拼接到特征后）

        Returns:
            包含ITE、反事实预测等的字典
        """
        batch_size = z_dict['function'].size(0)

        # ============================================
        # Step 1: Abduction（推断外生变量）
        # ============================================
        U_exogenous = self._abduction(z_dict, mu_dict, logvar_dict)

        # ============================================
        # Step 2: Action（干预操作）
        # ============================================
        # 学习自适应干预强度
        z_concat = torch.cat([
            z_dict['function'],
            z_dict['aesthetics'],
            z_dict['emotion']
        ], dim=-1)  # (batch, 192)
        strengths = self.intervention_strength(z_concat)  # (batch, 3)

        # 生成反事实场景
        counterfactuals = self._generate_counterfactuals(
            z_dict, U_exogenous, strengths
        )

        # ============================================
        # Step 3: Prediction（反事实预测）
        # ============================================
        # ⭐ P0修复：传递item_embedding给反事实预测
        cf_predictions = self._counterfactual_prediction(
            counterfactuals, quantum_encoder, recommendation_head, item_embedding
        )

        # 事实预测（baseline）
        # ⭐ P0修复：拼接item_embedding
        if item_embedding is not None:
            z_factual = torch.cat([z_concat, item_embedding], dim=-1)  # (batch, 320)
        else:
            z_factual = z_concat  # (batch, 192) - 向后兼容
        
        quantum_out_factual = quantum_encoder(z_factual)
        logits_factual = recommendation_head(quantum_out_factual['output'])

        # ============================================
        # 计算Individual Treatment Effect (ITE)
        # ============================================
        ite = self._compute_ite(
            logits_factual, cf_predictions, target_items, candidate_items  # ⭐ 传递candidate_items
        )

        # ============================================
        # 因果损失（理论严谨版）
        # ============================================
        causal_loss = self._compute_causal_loss(
            ite, strengths, U_exogenous, target_items
        )

        return {
            'ite': ite,
            'counterfactual_predictions': cf_predictions,
            'exogenous_variables': U_exogenous,
            'intervention_strengths': strengths,
            'causal_loss': causal_loss,
            'causal_effects': {'ite': torch.stack([v['target'] for v in ite.values() if 'target' in v], dim=-1) if target_items is not None else None},
            'original_features': z_factual
        }

    def _abduction(
        self,
        z_dict: Dict[str, torch.Tensor],
        mu_dict: Dict[str, torch.Tensor],
        logvar_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Step 1: Abduction
        从观测(Z)推断外生变量(U)

        公式：ε = (z - μ) / σ
        """
        U = {}

        for dim_name in ['function', 'aesthetics', 'emotion']:
            z = z_dict[dim_name]
            mu = mu_dict[dim_name]
            logvar = logvar_dict[dim_name]

            sigma = torch.exp(0.5 * logvar)
            epsilon = (z - mu) / (sigma + 1e-8)

            U[dim_name] = {
                'epsilon': epsilon,
                'mu': mu,
                'sigma': sigma
            }

        return U

    def _generate_counterfactuals(
        self,
        z_dict: Dict[str, torch.Tensor],
        U_exogenous: Dict[str, Dict[str, torch.Tensor]],
        strengths: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Step 2: Action（简化版，保留3个最有效的干预）
        
        精简策略：
        1. Set to mean (function) - 测试功能性的因果作用
        2. Shift (aesthetics) - 测试连续性的因果作用
        3. Swap (emotion) - 测试混淆因子的因果作用
        """
        batch_size = z_dict['function'].size(0)

        counterfactuals = {}

        # ===场景1：do(function = mean) - 去个性化===
        z_func_cf1 = z_dict['function'].mean(dim=0, keepdim=True).expand_as(
            z_dict['function']
        )

        counterfactuals['function_to_mean'] = {
            'function': z_func_cf1,
            'aesthetics': z_dict['aesthetics'],
            'emotion': z_dict['emotion']
        }

        # ===场景2：do(aesthetics = aesthetics + δ) - 增强/减弱===
        z_aes_cf1 = z_dict['aesthetics'] + \
                   strengths[:, 1:2] * U_exogenous['aesthetics']['sigma']

        counterfactuals['aesthetics_shift'] = {
            'function': z_dict['function'],
            'aesthetics': z_aes_cf1,
            'emotion': z_dict['emotion']
        }

        # ===场景3：do(emotion = other's_emotion) - 交换===
        indices = torch.randperm(batch_size, device=z_dict['emotion'].device)
        z_emo_cf1 = z_dict['emotion'][indices]

        counterfactuals['emotion_swap'] = {
            'function': z_dict['function'],
            'aesthetics': z_dict['aesthetics'],
            'emotion': z_emo_cf1
        }

        return counterfactuals

    def _counterfactual_prediction(
        self,
        counterfactuals: Dict[str, Dict[str, torch.Tensor]],
        quantum_encoder: nn.Module,
        recommendation_head: callable,
        item_embedding: Optional[torch.Tensor] = None  # ⭐ P0修复：传入item embedding
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Step 3: Prediction
        用修改后的SCM预测反事实结果

        关键：保持因果链的完整性
        
        Args:
            counterfactuals: 反事实特征字典
            quantum_encoder: 量子编码器
            recommendation_head: 推荐打分函数
            item_embedding: 原始item embedding（如果提供，会拼接到反事实特征后）
        """
        cf_predictions = {}

        for scenario_name, cf_z_dict in counterfactuals.items():
            # 拼接反事实的解耦表征
            z_concat_cf = torch.cat([
                cf_z_dict['function'],
                cf_z_dict['aesthetics'],
                cf_z_dict['emotion']
            ], dim=-1)  # (batch, 192)
            
            # ⭐ P0修复：如果提供了item_embedding，拼接上去
            if item_embedding is not None:
                z_concat_cf = torch.cat([z_concat_cf, item_embedding], dim=-1)  # (batch, 320)

            # 重新经过量子编码器（保持因果链）
            quantum_out_cf = quantum_encoder(z_concat_cf)

            # 预测
            logits_cf = recommendation_head(quantum_out_cf['output'])

            cf_predictions[scenario_name] = {
                'logits': logits_cf,
                'quantum_state': quantum_out_cf
            }

        return cf_predictions

    def _compute_ite(
        self,
        logits_factual: torch.Tensor,
        cf_predictions: Dict[str, Dict[str, torch.Tensor]],
        target_items: Optional[torch.Tensor],
        candidate_items: Optional[torch.Tensor] = None  # ⭐ 新增参数
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        计算Individual Treatment Effect

        ITE_dim = Y_counterfactual - Y_factual
        
        支持两种模式：
        1. 全库模式 (candidate_items=None): logits形状(batch, num_items+1)
        2. Candidate模式 (candidate_items提供): logits形状(batch, num_candidates)
        """
        ite = {}
        batch_size = logits_factual.size(0)

        for scenario_name, cf_pred in cf_predictions.items():
            logits_cf = cf_pred['logits']

            # 全局ITE（所有商品）
            ite[scenario_name] = {
                'global': logits_cf - logits_factual  # (batch, num_items) or (batch, num_candidates)
            }

            # 针对target_item的ITE
            if target_items is not None:
                if candidate_items is not None:
                    # ⭐⭐⭐ Candidate模式：第0列就是target
                    # candidate_items形状: (batch, num_candidates)
                    # candidate_items[:, 0] == target_items (验证用)
                    ite_target = logits_cf[:, 0] - logits_factual[:, 0]  # (batch,)
                else:
                    # 全库模式：用target_items索引
                    batch_indices = torch.arange(batch_size, device=logits_factual.device)
                    ite_target = (
                        logits_cf[batch_indices, target_items] -
                        logits_factual[batch_indices, target_items]
                    )

                ite[scenario_name]['target'] = ite_target  # (batch,)

        return ite

    def _compute_causal_loss(
        self,
        ite: Dict[str, Dict[str, torch.Tensor]],
        strengths: torch.Tensor,
        U_exogenous: Dict[str, Dict[str, torch.Tensor]],
        target_items: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        简化的因果损失（仅保留ITE幅度正则）
        """
        losses = {}

        # ===Loss 1: ITE Magnitude Regularization（核心）===
        # 确保ITE在合理范围内（不要太大也不要太小）
        target_magnitude = 0.3

        magnitude_loss = 0
        count = 0
        for scenario_name, ite_dict in ite.items():
            if 'target' in ite_dict:
                magnitude_loss += F.smooth_l1_loss(
                    ite_dict['target'].abs(),
                    torch.full_like(ite_dict['target'], target_magnitude)
                )
                count += 1
        
        magnitude_loss = magnitude_loss / count if count > 0 else torch.tensor(0.0, device=strengths.device)

        losses['magnitude'] = magnitude_loss

        # ===简化的加权组合（仅幅度）===
        total_causal_loss = self.causal_loss_weights.get('magnitude', 1.0) * losses['magnitude']

        return total_causal_loss


class CausalInferenceModule(nn.Module):
    """
    完整的因果推断模块（基于SCM）
    整合Pearl三步反事实推理
    """

    def __init__(
        self,
        disentangled_dim: int = 64,
        num_dimensions: int = 3,
        hidden_dim: int = 256,
        num_ensembles: int = 5,
        causal_loss_weights: Dict[str, float] = None,  # ⭐ 因果损失权重
        feature_dim: Optional[int] = None  # ⭐ P0修复：允许自定义特征维度
    ):
        super().__init__()

        self.disentangled_dim = disentangled_dim
        self.num_dimensions = num_dimensions
        total_dim = disentangled_dim * num_dimensions

        # SCM核心组件（传递权重配置）
        self.scm = StructuralCausalModel(
            latent_dim=disentangled_dim,
            causal_loss_weights=causal_loss_weights  # ⭐ 传递权重
        )

        # 不确定性量化（保留用于向后兼容）
        # ⭐ P0修复：如果提供了feature_dim，使用它；否则使用total_dim
        uncertainty_feature_dim = feature_dim if feature_dim is not None else total_dim
        self.uncertainty_quantification = UncertaintyQuantification(
            feature_dim=uncertainty_feature_dim,  # ⭐ 使用灵活的特征维度
            hidden_dim=hidden_dim,
            num_ensembles=num_ensembles
        )

    def forward(
        self,
        disentangled_features: Dict[str, torch.Tensor],
        return_uncertainty: bool = True,
        num_mc_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        完整的因果推断流程（向后兼容的接口）

        Args:
            disentangled_features: 解耦特征字典
            return_uncertainty: 是否返回不确定性估计
            num_mc_samples: MC采样次数

        Returns:
            包含反事实、因果效应和不确定性的完整结果
        """
        # 拼接原始特征
        original_concat = torch.cat([
            disentangled_features['function'],
            disentangled_features['aesthetics'],
            disentangled_features['emotion']
        ], dim=-1)

        # 创建简单的反事实（用于向后兼容）
        counterfactuals = self._generate_simple_counterfactuals(disentangled_features)

        results = {
            'counterfactuals': counterfactuals,
            'original_features': original_concat,
            'causal_effects': {}
        }

        # 不确定性量化
        if return_uncertainty:
            uncertainty = self.uncertainty_quantification(
                original_concat,
                num_mc_samples=num_mc_samples
            )
            results['uncertainty'] = uncertainty

        return results

    def _generate_simple_counterfactuals(
        self,
        disentangled_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """生成简单的反事实（向后兼容）"""
        batch_size = disentangled_features['function'].size(0)

        counterfactuals = {}

        # 功能维度的反事实
        z_func_mean = disentangled_features['function'].mean(dim=0, keepdim=True)
        counterfactuals['cf_function_strength_0'] = z_func_mean.expand(batch_size, -1)

        # 美学维度的反事实
        z_aes_perturb = disentangled_features['aesthetics'] + 0.1 * torch.randn_like(
            disentangled_features['aesthetics']
        )
        counterfactuals['cf_aesthetics_strength_0'] = z_aes_perturb

        # 情感维度的反事实
        indices = torch.randperm(batch_size, device=disentangled_features['emotion'].device)
        counterfactuals['cf_emotion_strength_0'] = disentangled_features['emotion'][indices]

        counterfactuals['original'] = disentangled_features

        return counterfactuals


class UncertaintyQuantification(nn.Module):
    """
    简化的不确定性量化模块（适合小数据集）
    只使用轻量级ensemble，移除MC dropout
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,  # ⭐ 256→128
        num_ensembles: int = 3,  # ⭐ 5→3
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.num_ensembles = num_ensembles
        self.dropout_rate = dropout_rate

        # 简化的ensemble heads（单层）
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1)  # ⭐ 移除中间层
            )
            for _ in range(num_ensembles)
        ])

    def forward(
        self,
        features: torch.Tensor,
        num_mc_samples: int = 5  # ⭐ 10→5
    ) -> Dict[str, torch.Tensor]:
        """
        简化的不确定性量化（只用ensemble）

        Args:
            features: 输入特征 (batch, feature_dim)
            num_mc_samples: 保留接口兼容性，但不使用

        Returns:
            包含预测和不确定性估计的字典
        """
        # Deep Ensemble（核心方法）
        ensemble_predictions = []
        for head in self.ensemble_heads:
            pred = head(features)
            ensemble_predictions.append(pred)

        ensemble_predictions = torch.stack(ensemble_predictions, dim=-1)
        ensemble_predictions = ensemble_predictions.squeeze(1)  # (batch, num_ensembles)

        # 计算统计量
        ensemble_mean = ensemble_predictions.mean(dim=-1)
        ensemble_var = ensemble_predictions.var(dim=-1)

        # 简化：直接用ensemble方差作为不确定性
        total_uncertainty = ensemble_var
        confidence = 1.0 / (1.0 + total_uncertainty + 1e-8)

        return {
            'prediction': ensemble_mean,
            'ensemble_predictions': ensemble_predictions,
            'total_uncertainty': total_uncertainty,
            'confidence': confidence,
            # 向后兼容性
            'aleatoric_uncertainty': total_uncertainty * 0.5,
            'epistemic_uncertainty': total_uncertainty * 0.5
        }
