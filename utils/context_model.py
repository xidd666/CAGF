import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel, WavLMModel
from utils.cross_attn_encoder import CMELayer, BertConfig


# Gated Fusion Module
class GatedFusionSchemeA(nn.Module):
    """Two Independent Gated Networks:
    - Branch 1 (original modality): text + audio
    - Branch 2 (context modality): text_context + audio_context
    
    **Each branch independently performs the following steps**:
    1. Compute c = concat[E_t, E_a]
    2. Compute three gates: G_t, G_a, G_int
    3. Compute gating output: G_fus = G_t_g + G_a_g + G_int_g
    4. Non-linear mapping: h = ReLU(W_fus[c; G_fus])
    5. Attention weighting: α = Sigmoid(W_att h), O = α ⊙ h
    
    **Final concatenation**: O_fus = concat[O_t-a; O_t-a_cont]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        in_dim = hidden_dim * 2  # c = concat(E_t, E_a)
        
        # ============ Branch 1 (original modality) - independent parameters ============
        # Gating weights
        self.b1_Wt = nn.Linear(in_dim, hidden_dim)
        self.b1_Wa = nn.Linear(in_dim, hidden_dim)
        self.b1_Wint = nn.Linear(in_dim, hidden_dim)
        # Step 4: Non-linear mapping weights (independent)
        self.b1_W_fus = nn.Linear(in_dim + hidden_dim, hidden_dim)
        # Step 5: Attention weights (independent)
        self.b1_W_att = nn.Linear(hidden_dim, hidden_dim)
        
        # ============ Branch 2 (context modality) - independent parameters ============
        # Gating weights
        self.b2_Wt = nn.Linear(in_dim, hidden_dim)
        self.b2_Wa = nn.Linear(in_dim, hidden_dim)
        self.b2_Wint = nn.Linear(in_dim, hidden_dim)
        # Step 4: Non-linear mapping weights (independent)
        self.b2_W_fus = nn.Linear(in_dim + hidden_dim, hidden_dim)
        # Step 5: Attention weights (independent)
        self.b2_W_att = nn.Linear(hidden_dim, hidden_dim)

        # Concatenate two branches' outputs and project from 2*hidden_dim to hidden_dim (to match classifier input)
        self.output_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text, audio, text_context, audio_context):
        """Gated Fusion Forward Pass
        
        Args:
            text: (B, D) Original text features
            audio: (B, D) Original audio features
            text_context: (B, D) Context text features
            audio_context: (B, D) Context audio features
        
        Returns:
            fused: (B, D) Fused features
        """
        # Branch1: original
        c1 = torch.cat([text, audio], dim=-1)  # (B, 2D)
        Gt1 = torch.sigmoid(self.b1_Wt(c1))
        Ga1 = torch.sigmoid(self.b1_Wa(c1))
        Gint1 = torch.sigmoid(self.b1_Wint(c1))
        Gt1_g = Gt1 * text
        Ga1_g = Ga1 * audio
        Gint1_g = Gint1 * (text * audio)
        Gfus1 = Gt1_g + Ga1_g + Gint1_g  # (B, D)

        # === Branch 1 independently computes Step 4-5 (using independent parameters) ===
        # Step 4: h = ReLU(W_fus [c; G_fus])
        h1_in = torch.cat([c1, Gfus1], dim=-1)  # (B, 3D)
        h1 = torch.relu(self.b1_W_fus(h1_in))     # (B, D) - Using Branch 1's independent W_fus
        
        # Step 5: α = Sigmoid(W_att h), O = α ⊙ h
        alpha1 = torch.sigmoid(self.b1_W_att(h1))  # (B, D)
        O1 = alpha1 * h1  # Branch 1's final output O_t-a

        # Branch2: context
        c2 = torch.cat([text_context, audio_context], dim=-1)
        Gt2 = torch.sigmoid(self.b2_Wt(c2))
        Ga2 = torch.sigmoid(self.b2_Wa(c2))
        Gint2 = torch.sigmoid(self.b2_Wint(c2))
        Gt2_g = Gt2 * text_context
        Ga2_g = Ga2 * audio_context
        Gint2_g = Gint2 * (text_context * audio_context)
        Gfus2 = Gt2_g + Ga2_g + Gint2_g

        # === Branch 2 independently computes Step 4-5 (using independent parameters) ===
        # Step 4: h = ReLU(W_fus [c; G_fus])
        h2_in = torch.cat([c2, Gfus2], dim=-1)  # (B, 3D)
        h2 = torch.relu(self.b2_W_fus(h2_in))     # (B, D) - Using Branch 2's independent W_fus
        
        # Step 5: α = Sigmoid(W_att h), O = α ⊙ h
        alpha2 = torch.sigmoid(self.b2_W_att(h2))  # (B, D)
        O2 = alpha2 * h2  # Branch 2's final output O_t-a_cont

        # === Final concatenation ===
        # O_fus = concat[O_t-a; O_t-a_cont]
        # Concatenate the independently computed O from both branches, then project to hidden_dim to match classifier input
        O_fus = torch.cat([O1, O2], dim=-1)  # (B, 2*hidden_dim)
        fused = self.output_proj(O_fus)      # (B, hidden_dim)
        fused = self.layer_norm(fused)       # Normalize output to stabilize magnitude
        return fused

# English text model + context
class roberta_en_context(nn.Module):
            
    def __init__(self):
        super().__init__() 
        self.roberta_model = AutoModel.from_pretrained('roberta-large')
        self.classifier = nn.Linear(1024*2, 1)    
   
    def forward(self, input_ids, attention_mask, context_input_ids, context_attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        context_output = self.roberta_model(context_input_ids, context_attention_mask, return_dict=True)
        context_pooler = context_output["pooler_output"]   # Shape is [batch_size, 1024]

        pooler = torch.cat((input_pooler, context_pooler), dim=1)
        output = self.classifier(pooler)                    # Shape is [batch_size, 1]
        return output
    

# English text+audio model + context
class rob_wavlm_cc_context(nn.Module):            
    def __init__(self, config):        
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.wavlm_model = WavLMModel.from_pretrained('microsoft/wavlm-base')
        
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 1)
          )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2+768*2, 1024*2),
            nn.ReLU(),
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
        
    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        context_pooler = raw_output_context["pooler_output"]    # Shape is [batch_size, 1024]

        # audio feature extraction
        audio_out = self.wavlm_model(audio_inputs, attention_mask=audio_mask)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        A_features = []
        for batch in range(A_hidden_states.shape[0]):
            padding_idx = int(audio_mask[batch].sum().item())
            if padding_idx == 0:
                padding_idx = A_hidden_states.shape[1]
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features,0).to(audio_inputs.device)   # Shape is [batch_size, 768]
        
        # audio context feature extraction
        audio_context_out = self.wavlm_model(audio_context_inputs, attention_mask=audio_context_mask)
        A_context_hidden_states = audio_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_context_features = []
        for batch in range(A_context_hidden_states.shape[0]):
            padding_idx = int(audio_context_mask[batch].sum().item())
            if padding_idx == 0:
                padding_idx = A_context_hidden_states.shape[1]
            truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_context_features.append(truncated_feature)
        A_context_features = torch.stack(A_context_features,0).to(audio_context_inputs.device)   # Shape is [batch_size, 768]

        T_features = torch.cat((input_pooler, context_pooler), dim=1)    # Shape is [batch_size, 1024*2]
        A_features = torch.cat((A_features, A_context_features), dim=1)  # Shape is [batch_size, 768*2]
        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 1]
        
        fused_features = torch.cat((T_features, A_features), dim=1)    # Shape is [batch_size, 1024*2+768*2]
        fused_output = self.fused_output_layers(fused_features)        # Shape is [batch_size, 1]

        return {
                'T': T_output, 
                'A': A_output, 
                'M': fused_output
        }


# English text+audio model + context + cme
class rob_wavlm_cme_context(nn.Module):            
    def __init__(self, config, use_gated_fusion=False):        
        super().__init__()
        self.use_gated_fusion = use_gated_fusion  # Flag to control whether to use gated fusion
        
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1)
          )
        
        # Choose different fusion output layer based on whether to use gated fusion
        if self.use_gated_fusion:
            # Output layer when using gated fusion (input dimension is 1024, since gated fusion outputs 1024)
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(1024, 768),
                nn.ReLU(),
                nn.Linear(768, 1)
            )
            # Initialize gated fusion module (hidden_dim=1024)
            self.gated_fusion = GatedFusionSchemeA(hidden_dim=1024)
        else:
            # Original concatenation fusion output layer
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(1024*4, 768),
                nn.ReLU(),
                nn.Linear(768, 1)
            )
        
        # Audio Feature Adapter
        self.audio_adapter = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 1024)
        )
        
        # Hierarchical feature weights (learnable) - weighted fusion of 24 layers
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers, hidden_size=1024, intermediate_size=4096, num_attention_heads=16)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )
        
        
    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        T_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"]    # Shape is [batch_size, 1024]

        # audio feature extraction with hierarchical fusion
        audio_out = self.wavlm_model(
            audio_inputs, 
            attention_mask=audio_mask,
            output_hidden_states=True,  # Get outputs from all 24 layers
            return_dict=True
        )
        
        # Hierarchical weighted fusion (weighted average of 24 layers)
        # hidden_states contains 25 states: embedding layer + 24 transformer layers, we only use the last 24 layers
        all_hidden_states = torch.stack(audio_out.hidden_states[1:], dim=0)  # [24, batch, seq, 1024]
        weights = torch.softmax(self.layer_weights, dim=0)  # Learn weights for each layer
        A_hidden_states = torch.sum(
            weights.view(-1, 1, 1, 1) * all_hidden_states, 
            dim=0
        )  # [batch, seq, 1024]
        
        # Feature adaptation (dimensionality reduction then expansion, adding non-linear transformation)
        A_hidden_states = self.audio_adapter(A_hidden_states)
        
        ## Average over unmasked audio tokens (using attention_mask to compute valid length)
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            # Directly compute valid length from attention_mask
            padding_idx = int(audio_mask[batch].sum().item())
            if padding_idx == 0:  # If mask is all zeros, use the entire sequence
                padding_idx = A_hidden_states.shape[1]
            audio_mask_idx_new.append(padding_idx)
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx], 0)  # Shape is [1024]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features, 0).to(audio_inputs.device)   # Shape is [batch_size, 1024]
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(audio_inputs.device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1

        # Audio context feature extraction with hierarchical fusion
        audio_context_out = self.wavlm_model(
            audio_context_inputs, 
            attention_mask=audio_context_mask,
            output_hidden_states=True,  # Get outputs from all 24 layers
            return_dict=True
        )
        
        # Hierarchical weighted fusion (skip embedding layer, only use 24 transformer layers)
        all_context_hidden_states = torch.stack(audio_context_out.hidden_states[1:], dim=0)
        A_context_hidden_states = torch.sum(
            weights.view(-1, 1, 1, 1) * all_context_hidden_states, 
            dim=0
        )
        
        # Feature adaptation
        A_context_hidden_states = self.audio_adapter(A_context_hidden_states)
        
        ## Average over unmasked audio tokens (using attention_mask to compute valid length)
        A_context_features = []
        audio_context_mask_idx_new = []
        for batch in range(A_context_hidden_states.shape[0]):
            # Directly compute valid length from attention_mask
            padding_idx = int(audio_context_mask[batch].sum().item())
            if padding_idx == 0:  # If mask is all zeros, use the entire sequence
                padding_idx = A_context_hidden_states.shape[1]
            audio_context_mask_idx_new.append(padding_idx)
            truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx], 0)  # Shape is [1024]
            A_context_features.append(truncated_feature)
        A_context_features = torch.stack(A_context_features, 0).to(audio_context_inputs.device)   # Shape is [batch_size, 1024]
        audio_context_mask_new = torch.zeros(A_context_hidden_states.shape[0], A_context_hidden_states.shape[1]).to(audio_context_inputs.device)
        for batch in range(audio_context_mask_new.shape[0]):
            audio_context_mask_new[batch][:audio_context_mask_idx_new[batch]] = 1

        T_features = torch.cat((input_pooler, context_pooler), dim=1)    # Shape is [batch_size, 1024*2]
        A_features = torch.cat((A_features, A_context_features), dim=1)  # Shape is [batch_size, 1024*2]
        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 1]
        
        # CME layers
        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text') # Add CLS token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio') # Add CLS token

        text_context_inputs, text_context_attn_mask = self.prepend_cls(T_context_hidden_states, text_context_mask, 'text') # Add CLS token
        audio_context_inputs, audio_context_attn_mask = self.prepend_cls(A_context_hidden_states, audio_context_mask_new, 'audio') # Add CLS token
        
        for layer_module in self.CME_layers:
            text_inputs, audio_inputs = layer_module(text_inputs, text_attn_mask,
                                                audio_inputs, audio_attn_mask)
        
        for layer_module in self.CME_layers:
            text_context_inputs, audio_context_inputs = layer_module(text_context_inputs, text_context_attn_mask,
                                                audio_context_inputs, audio_context_attn_mask)

        # Choose different fusion method based on whether to use gated fusion
        if self.use_gated_fusion:
            # Use gated fusion
            # Extract CLS token representations
            text_cls = text_inputs[:, 0, :]           # [batch_size, 1024]
            audio_cls = audio_inputs[:, 0, :]         # [batch_size, 1024]
            text_context_cls = text_context_inputs[:, 0, :]   # [batch_size, 1024]
            audio_context_cls = audio_context_inputs[:, 0, :] # [batch_size, 1024]
            
            # Gated fusion: two independent gated networks (original modality + context modality)
            fused_hidden_states = self.gated_fusion(
                text_cls, 
                audio_cls, 
                text_context_cls, 
                audio_context_cls
            )  # Shape is [batch_size, 1024]
        else:
            # Original concatenation fusion method
            fused_hidden_states = torch.cat(
                (text_inputs[:,0,:], text_context_inputs[:,0,:], 
                 audio_inputs[:,0,:], audio_context_inputs[:,0,:]), 
                dim=1
            )  # Shape is [batch_size, 1024*4]

        # last linear output layer
        fused_output = self.fused_output_layers(fused_hidden_states) # Shape is [batch_size, 1]
        
        return {
                'T': T_output, 
                'A': A_output, 
                'M': fused_output
        }