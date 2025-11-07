"""
极简多模态解耦层：2独立 + 1门控

创新点：
- Emotion (Text独有)
- Aesthetics (Image独有)  
- Context (门控融合Text+Image)

优势：
- 架构简洁，易训练
- 门控机制自动学习模态可靠性
- 单一ELBO损失，无需调参
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VAEHead(nn.Module):
    """VAE头部（标准实现）"""
    
    def __init__(self, input_dim: int, latent_dim: int, dimension_name: str):
        super().__init__()
        self.dimension_name = dimension_name
        self.latent_dim = latent_dim
        
        # 投影层
        self.projector = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # μ和logσ²头部
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)
        
        # 初始化：避免KL坍塌
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, input_dim)
        
        Returns:
            z: 采样的潜在表征
            mu: 均值
            logvar: log方差
            epsilon: 采样噪声
        """
        h = self.projector(features)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        
        return {'z': z, 'mu': mu, 'logvar': logvar, 'epsilon': epsilon}


class ModalityGate(nn.Module):
    """
    轻量级门控：为Context维度计算Text/Image的融合权重
    
    理论：不同样本对Text和Image的依赖程度不同
    例如：
    - 功能性商品（充电器）→ 更依赖Text描述
    - 美观性商品（服装）→ 更依赖Image外观
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # 简单的2层MLP
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),  # ⭐ 同时看text和image
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, text_z: torch.Tensor, image_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_z, image_z: (batch, latent_dim)
        
        Returns:
            weights: (batch, 2) - [text_weight, image_weight]
        """
        # 拼接两个模态的context特征
        combined = torch.cat([text_z, image_z], dim=-1)
        weights = self.gate_net(combined)
        return weights


class SimpleGatedDisentangling(nn.Module):
    """
    极简多模态解耦：2独立 + 1门控
    
    架构：
    - Text  → Emotion（独有）+ Context（参与融合）
    - Image → Aesthetics（独有）+ Context（参与融合）
    
    创新点：
    - Context维度通过门控机制动态融合Text和Image
    - 门控自动学习每个样本的模态可靠性
    
    损失：
    - 只用ELBO = Reconstruction + β*KL
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        latent_dim: int = 64,
        beta: float = 0.1,
        gamma: float = 0.0,  # 兼容参数（不使用）
        use_kl_loss: bool = True,
        use_independence_loss: bool = False  # 兼容参数（不使用）
    ):
        """
        Args:
            input_dims: {'text': 768, 'image': 2048}
            latent_dim: 每个解耦维度的大小
            beta: KL散度权重
            gamma: 兼容参数（极简版不使用独立性损失）
            use_kl_loss: 是否使用KL损失（默认True）
            use_independence_loss: 兼容参数（极简版不使用）
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma  # 保存但不使用
        self.use_kl_loss = use_kl_loss
        
        # ===Text编码器（简化）===
        text_hidden = 256
        self.text_backbone = nn.Sequential(
            nn.Linear(input_dims['text'], text_hidden),
            nn.LayerNorm(text_hidden),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Text → Emotion（独有）
        self.text_emotion_encoder = VAEHead(text_hidden, latent_dim, 'emotion')
        
        # Text → Context（用于融合）
        self.text_context_encoder = VAEHead(text_hidden, latent_dim, 'context')
        
        # ===Image编码器（简化）===
        image_hidden = 256
        self.image_backbone = nn.Sequential(
            nn.Linear(input_dims['image'], image_hidden),
            nn.LayerNorm(image_hidden),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Image → Aesthetics（独有）
        self.image_aesthetics_encoder = VAEHead(image_hidden, latent_dim, 'aesthetics')
        
        # Image → Context（用于融合）
        self.image_context_encoder = VAEHead(image_hidden, latent_dim, 'context')
        
        # ===创新点：门控机制（轻量级）===
        self.context_gate = ModalityGate(latent_dim)
        
        # ===解码器（ELBO重构）===
        # Text从2个维度重构：emotion + context
        self.text_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, text_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_hidden, input_dims['text'])
        )
        
        # Image从2个维度重构：aesthetics + context
        self.image_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, image_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(image_hidden, input_dims['image'])
        )
    
    def forward(
        self,
        multimodal_features: Dict[str, torch.Tensor],
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：极简架构
        
        流程：
        1. Text → [Emotion, Context_text]
        2. Image → [Aesthetics, Context_image]
        3. Context = Gate(Context_text, Context_image)
        4. 输出：[Emotion, Aesthetics, Context]
        """
        text_feat = multimodal_features['text']
        image_feat = multimodal_features['image']
        batch_size = text_feat.size(0)
        
        # ===编码阶段===
        text_h = self.text_backbone(text_feat)
        image_h = self.image_backbone(image_feat)
        
        # Text编码
        text_emotion_out = self.text_emotion_encoder(text_h)
        text_context_out = self.text_context_encoder(text_h)
        
        # Image编码
        image_aesthetics_out = self.image_aesthetics_encoder(image_h)
        image_context_out = self.image_context_encoder(image_h)
        
        # ===3个维度===
        # 维度1: Emotion（Text独有）
        z_emotion = text_emotion_out['z']
        
        # 维度2: Aesthetics（Image独有）
        z_aesthetics = image_aesthetics_out['z']
        
        # 维度3: Context（门控融合，创新点）⭐⭐⭐
        gate_weights = self.context_gate(
            text_context_out['z'], 
            image_context_out['z']
        )  # (batch, 2)
        
        z_context = (
            gate_weights[:, 0:1] * text_context_out['z'] +
            gate_weights[:, 1:2] * image_context_out['z']
        )
        
        # ===拼接（顺序：Emotion, Aesthetics, Context）===
        z_concat = torch.cat([
            z_emotion,      # (batch, 64)
            z_aesthetics,   # (batch, 64)
            z_context       # (batch, 64)
        ], dim=-1)  # (batch, 192)
        
        # ===输出（兼容旧接口）===
        # Context的VAE参数（门控加权）
        context_mu = (
            gate_weights[:, 0:1] * text_context_out['mu'] +
            gate_weights[:, 1:2] * image_context_out['mu']
        )
        context_logvar = (
            gate_weights[:, 0:1] * text_context_out['logvar'] +
            gate_weights[:, 1:2] * image_context_out['logvar']
        )
        
        results = {
            # ===主输出：3个解耦维度===
            'z_emotion': z_emotion,
            'z_aesthetics': z_aesthetics,
            'z_context': z_context,
            'z_concat': z_concat,
            
            # ===完整VAE输出（用于因果推断）===
            'emotion_full': text_emotion_out,
            'aesthetics_full': image_aesthetics_out,
            'context_full': {
                'z': z_context,
                'mu': context_mu,
                'logvar': context_logvar,
                'epsilon': text_context_out['epsilon']  # 共享噪声
            },
            
            # ===向后兼容（z_function → z_context）===
            'z_function': z_context,
            'function_full': {
                'z': z_context,
                'mu': context_mu,
                'logvar': context_logvar,
                'epsilon': text_context_out['epsilon']
            },
            
            # ===门控权重（可视化分析）===
            'gate_weights': gate_weights,  # (batch, 2)
            
            # ===兼容接口（向后兼容）===
            'attention_maps': {},
            'modality_disentangled': {
                'text': {
                    'emotion': text_emotion_out,
                    'context': text_context_out
                },
                'image': {
                    'aesthetics': image_aesthetics_out,
                    'context': image_context_out
                }
            }
        }
        
        # ===单一ELBO损失===
        if return_loss:
            # 1. 重构损失
            # Text从emotion+context重构
            text_z = torch.cat([z_emotion, z_context], dim=-1)
            text_recon = self.text_decoder(text_z)
            text_recon_loss = F.mse_loss(text_recon, text_feat, reduction='mean')  # ⭐ 改为mean
            
            # Image从aesthetics+context重构
            image_z = torch.cat([z_aesthetics, z_context], dim=-1)
            image_recon = self.image_decoder(image_z)
            image_recon_loss = F.mse_loss(image_recon, image_feat, reduction='mean')  # ⭐ 改为mean
            
            recon_loss = (text_recon_loss + image_recon_loss) / 2.0
            
            # 2. KL散度（可选）
            if self.use_kl_loss and self.beta > 0:
                kl_terms = [
                    self._kl_divergence(text_emotion_out['mu'], text_emotion_out['logvar']),
                    self._kl_divergence(text_context_out['mu'], text_context_out['logvar']),
                    self._kl_divergence(image_aesthetics_out['mu'], image_aesthetics_out['logvar']),
                    self._kl_divergence(image_context_out['mu'], image_context_out['logvar'])
                ]
                kl_loss = sum(kl_terms) / len(kl_terms)
                kl_loss = torch.clamp(kl_loss, 0, 10.0)
            else:
                kl_loss = torch.tensor(0.0, device=text_feat.device)
            
            # 3. ELBO = Recon + β*KL
            elbo_loss = recon_loss + self.beta * kl_loss
            
            results.update({
                'loss': elbo_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'independence_loss': torch.tensor(0.0, device=text_feat.device)  # 兼容
            })
        
        return results
    
    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL散度: KL(q(z|x) || p(z))，其中p(z)=N(0,I)

        公式: -0.5 * sum(1 + log(σ²) - μ² - σ²)

        Note: 需要同时除以batch_size和latent_dim以正确归一化
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 修复：除以batch_size和latent_dim，防止KL损失随维度增长
        return kl / (mu.size(0) * mu.size(1))


# ===向后兼容别名===
DisentangledRepresentation = SimpleGatedDisentangling
DimensionSpecificMultimodalFusion = SimpleGatedDisentangling
