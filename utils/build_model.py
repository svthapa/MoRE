import torch 
import torch.nn as nn
import timm 
from collections import OrderedDict
from transformers import AutoModel, AutoConfig, RobertaTokenizerFast
from peft import LoraConfig, get_peft_model

class Attention(timm.models.vision_transformer.Attention):
    fused_attn: False

    def __init__(self, *args, qkv_bias = False, use_DropKey=True, mask_ratio=0.3, **kwargs):
        super().__init__(*args, qkv_bias=qkv_bias, **kwargs)
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.use_DropKey and self.training:
            m_r = torch.ones_like(attn)
            attn = attn + torch.bernoulli(m_r * self.mask_ratio) * -1e12
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(timm.models.vision_transformer.Block):
    def __init__(self, *args, qkv_bias = False, **kwargs):
        super().__init__(*args, qkv_bias = False, **kwargs)
        self.mask_ratio = 0.3
        self.attn = Attention(dim=768, num_heads=12, mask_ratio=self.mask_ratio)
        
class ViTWithDropKey(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, block_fn = Block)
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.forward_features(x)
    #     # x = self.forward_head(x)
    #     return x
    

class PatchEmbed(nn.Module):
    '''
    Patch Embedding for ECG data to match input for ViT Model. Converts (12,1000) to 196 patches of dim 768
    '''
    def __init__(self):
        super(PatchEmbed, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 12, out_channels = 256, kernel_size= 5, stride= 5),
            nn.ReLU(),
            nn.BatchNorm1d(256)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 768, kernel_size= 5, stride= 1),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            # Rearrange('b e d -> b d e')
            )

        self.relu = nn.ReLU()
        self.num_patches = 196

    def forward(self, x):
        conv1 = self.conv1(x)
        #skip = x + conv2
        out = self.conv2(conv1)
        out = out.permute(0,2,1)
        return out
    
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias = True, bn = True, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.use_bn = bn
        self.linear = nn.Linear(self.in_features, self.out_features, bias = self.bias)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)
            
    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, head_type = 'nonlinear', **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type
        
        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, True, False)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                    LinearLayer(self.in_features, self.hidden_features, False, True),
                    nn.ReLU(),
                    LinearLayer(self.hidden_features, self.out_features, False, True)
                )
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
class ViTModelEcg(nn.Module):
    def __init__(self, projector = True):
        super(ViTModelEcg, self).__init__()
        # self.vit_model = timm.create_model('vit_base_patch16_224', pretrained = True)
        self.vit_model = ViTWithDropKey()
        self.vit_model.patch_embed = PatchEmbed()
        self.vit_model.head = nn.Identity()
        self.projector = projector
        if self.projector:
            self.projector = ProjectionHead(768, 512, 768, head_type='nonlinear')
    
    def freeze_model(self):
        for param in list(self.vit_model.parameters()):#+list(self.projector.parameters()):
            param.requires_grad = False
        
    def forward(self, x):
        out = self.vit_model(x)
        # out = out[:,0,:]
        if self.projector:
            out = self.projector(out)
        
        return out

class ViTModelXray(nn.Module):
    def __init__(self, projector = True, proj_dim=128):
        super(ViTModelXray, self).__init__()
        # self.vit_model = timm.create_model('vit_base_patch16_224', pretrained = True)
        self.vit_model = ViTWithDropKey()
        self.vit_model.head = nn.Identity()
        self.projector = projector
        if self.projector:
            self.projector = ProjectionHead(768, proj_dim, 768, head_type='nonlinear')


    def freeze_model(self):
        for param in list(self.vit_model.parameters()):#+list(self.projector.parameters()):
            param.requires_grad = False
        
    def forward(self, x):
        out = self.vit_model(x)
        # out = out[:,0,:]
        if self.projector:
            out = self.projector(out)
        # out = self.fc(out)
        
        return out
    
class MultiModalHead(nn.Module):
    def __init__(self, in_dim = 1024*2, hid_dim = 256, out_dim = 4, proj_dim = 128, head_bias = True, dropout = 0.2, ecg_drop = 0.3, projector = False):
        super(MultiModalHead, self).__init__()
        self.xray_model = ViTModelXray(projector=projector)
        self.xray_model.freeze_model()
        self.ecg_model = ViTModelEcg(projector=projector)
        self.ecg_model.freeze_model()
        self.projector_xray = ProjectionHead(768, proj_dim, 768)
        self.projector_ecg = ProjectionHead(768, proj_dim, 768)
        self.dropout = dropout
        self.fc = nn.Sequential(
            # nn.Linear(in_dim, hid_dim),
            # # nn.BatchNorm1d(hid_dim),
            # nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, out_dim)
        )
        self.ecg_drop = ecg_drop
    
        
    def forward(self,xray, ecg):
        xray_out = self.projector_xray(self.xray_model(xray))
        ecg_out = self.projector_ecg(self.ecg_model(ecg))
        # xray_out = self.xray_model(xray)
        # ecg_out = self.ecg_model(ecg)
        if self.training and torch.rand(1).item() < self.ecg_drop:
            ecg_dropped = torch.zeros_like(ecg_out)
            combined = torch.cat([xray_out, ecg_dropped], axis = 1)
        else:
            combined = torch.cat([xray_out, ecg_out], axis = 1)
        out = self.fc(combined)

        return out
    
    
class MultiModal(nn.Module):
    def __init__(self):
        super(MultiModal, self).__init__()
        config = AutoConfig.from_pretrained(
                "./data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf",
            )
        self.text_model = AutoModel.from_pretrained("./data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf", config=config)
        # self.llm_model = AutoModel.from_pretrained("epfl-llm/meditron-7b")
        # for param in self.text_model.parameters():
        #     param.requires_grad = False
        # target_modules = ['q_proj','k_proj','v_proj', \
        #           'o_proj','gate_proj','down_proj', \
        #           'up_proj','lm_head']
        # target_modules = ["q_proj", "v_proj"]
        # lora_config = LoraConfig(
        #     r=16,
        #     target_modules = target_modules,
        #     lora_alpha=8,
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type="FEATURE_EXTRACTION",
        # )
        lora_config = LoraConfig(
            r=16,
            # target_modules = target_modules,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="all",
            task_type="FEATURE_EXTRACTION",
        )
        self.text_model = get_peft_model(self.text_model, lora_config)
        # del self.llm_model
        self.text_model.print_trainable_parameters()
        
        self.ecg_model = ViTModelEcg(projector = False)
        self.xray_model = ViTModelXray(projector = False)
        
#         ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
#         0.999 * averaged_model_parameter + 0.001 * model_parameter
#         self.ema_model_xray = torch.optim.swa_utils.AveragedModel(self.xray_model, avg_fn=ema_avg)
#         self.ema_model_ecg = torch.optim.swa_utils.AveragedModel(self.ecg_model, avg_fn=ema_avg)
        
#         self.ema_model_xray.eval(), self.ema_model_ecg.eval()

        self.projector_xray_text = ProjectionHead(768, 128, 768)
        # self.projector_xray = ProjectionHead(768, 128, 768)
        self.projector_ecg_text = ProjectionHead(768, 128, 768)
        # self.projector_ecg = ProjectionHead(768, 128, 768)
        self.projector_text = ProjectionHead(768, 128, 768)
        
        
    def forward(self, xray, ecg, note_id, mask):
        ecg_out = self.ecg_model(ecg)
        # ecg_t_out = self.ema_model_ecg(ecg_t)
        ecg_p_text = self.projector_ecg_text(ecg_out)
        # ecg_p = self.projector_ecg(ecg_out)
        # ecg_t_p = self.projector_ecg(ecg_t_out)
        
        xray_out = self.xray_model(xray)
        # xray_t_out = self.ema_model_xray(xray_t)
        xray_p_text = self.projector_xray_text(xray_out)
        # xray_p = self.projector_xray(xray_out)
        # xray_t_p = self.projector_xray(xray_t_out)
        
        note_out = self.text_model(note_id, attention_mask=mask).last_hidden_state.mean(dim=1)
        note_p = self.projector_text(note_out)
        
        return xray_out, xray_p_text, ecg_out, ecg_p_text, note_p
        # return xray_p, xray_t_p, xray_p_text, ecg_p, ecg_t_p, ecg_p_text, note_p
