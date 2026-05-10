import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel
from utils.cross_attn_encoder import CMELayer, BertConfig
# from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class rob_hub_cc(nn.Module):            
    def __init__(self, config):        
        super().__init__()
        model_path="audio_model"
        self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-large')
        
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
          )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_features = raw_output["pooler_output"]  # Shape is [batch_size, 1024]

        # audio feature extraction
        audio_out = self.hubert_model(audio_inputs, audio_mask)
        A_hidden_states = audio_out.last_hidden_state
        ## use _get_feat_extract_output_lengths to get the exact number of valid frames after downsampling
        audio_lengths = audio_mask.sum(dim=1).long()
        feat_lengths = self.hubert_model._get_feat_extract_output_lengths(audio_lengths)
        A_features = []
        for batch in range(A_hidden_states.shape[0]):
            feat_len = max(int(feat_lengths[batch].item()), 1)
            truncated_feature = torch.mean(A_hidden_states[batch][:feat_len], 0)  # [1024]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features, 0).to(device)

        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 1]
        
        fused_features = torch.cat((T_features, A_features), dim=1)    # Shape is [batch_size, 1024*2]
        fused_output = self.fused_output_layers(fused_features)        # Shape is [batch_size, 1]

        return {
                'T': T_output, 
                'A': A_output, 
                'M': fused_output
        }


    
class rob_hub_cme(nn.Module):            
    def __init__(self, config):        
        super().__init__()
        self.version = config.cme_version

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

        # load audio pre-trained model
        self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-large')
        
        # output layers for each single modality
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
          )
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)
        self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024*2)
        self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024*2)

        # position encoding
        #self.pos_enc = Summer(PositionalEncoding1D(1024))

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers, hidden_size=1024, num_attention_heads=16)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )

        # fused method V2
        self.text_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1024),
            nn.ReLU()
        )
        self.audio_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1024),
            nn.ReLU()
        )
        
        # fusion method V3
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024*2, nhead=16, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2,enable_nested_tensor=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024*2, nhead=16, batch_first=True)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2,enable_nested_tensor=False)
        
        # last linear output layer
        if self.version == 'v3':
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(1024*4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(1024*2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        
        # v1: fused = (T_pred + A_pred)/2 + correction
        if self.version == 'v1':
            nn.init.zeros_(self.fused_output_layers[-1].weight)
            nn.init.zeros_(self.fused_output_layers[-1].bias)
        
    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'text_mixed':
            embedding_layer = self.text_mixed_cls_emb
        elif layer_name == 'audio_mixed':
            embedding_layer = self.audio_mixed_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)
        T_hidden_states = raw_output.last_hidden_state
        T_features = raw_output["pooler_output"]  # Shape is [batch_size, 1024]
                
        # audio feature extraction
        audio_out = self.hubert_model(audio_inputs, audio_mask)
        A_hidden_states = audio_out.last_hidden_state
        audio_lengths = audio_mask.sum(dim=1).long()
        feat_lengths = self.hubert_model._get_feat_extract_output_lengths(audio_lengths)
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            feat_len = max(int(feat_lengths[batch].item()), 1)
            audio_mask_idx_new.append(feat_len)
            A_features.append(torch.mean(A_hidden_states[batch][:feat_len], 0))  # [1024]
        A_features = torch.stack(A_features, 0).to(device)
        ## create new audio mask
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1
                
        # text output layer
        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 2]
        
        # audio output layer
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 2]
        
        # CME layers
        text_hidden_for_cme = T_hidden_states.detach()
        audio_hidden_for_cme = A_hidden_states.detach()
        ## prepend cls tokens
        text_inputs, text_attn_mask = self.prepend_cls(text_hidden_for_cme, text_mask, 'text') # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(audio_hidden_for_cme, audio_mask_new, 'audio') # add cls token

        # position encoding
        # pos_enc_text = Summer(PositionalEncodingPermute1D(text_inputs.shape[1]))
        # text_inputs = pos_enc_text(text_inputs)
        # pos_enc_audio = Summer(PositionalEncodingPermute1D(audio_inputs.shape[1]))
        # audio_inputs = pos_enc_audio(audio_inputs)

        # pass through CME layers
        for layer_module in self.CME_layers:
            text_inputs, audio_inputs = layer_module(text_inputs, text_attn_mask,
                                                audio_inputs, audio_attn_mask)
        # different fusion methods
        if self.version == 'v1':
            # fused features
            fused_hidden_states = torch.cat((text_inputs[:,0,:], audio_inputs[:,0,:]), dim=1) # Shape is [batch_size, 1024*2]
        elif self.version == 'v2':
            # concatenate original features with fused features
            text_concat_features = torch.cat((T_features, text_inputs[:,0,:]), dim=1) # Shape is [batch_size, 1024*2]
            audio_concat_features = torch.cat((A_features, audio_inputs[:,0,:]), dim=1) # Shape is [batch_size, 1024*2]
            text_mixed_features = self.text_mixed_layer(text_concat_features) # Shape is [batch_size, 1024]
            audio_mixed_features = self.audio_mixed_layer(audio_concat_features) # Shape is [batch_size, 1024]
            fused_hidden_states = torch.cat((text_mixed_features, audio_mixed_features), dim=1) # Shape is [batch_size, 1024*2]
        else:
            # concatenate original features with fused features
            text_concat_features = torch.cat((T_hidden_states, text_inputs[:,1:,:]), dim=2) # Shape is [batch_size, text_length, 1024*2]
            audio_concat_features = torch.cat((A_hidden_states, audio_inputs[:,1:,:]), dim=2) # Shape is [batch_size, audio_length, 1024*2]
            text_concat_features, text_attn_mask = self.prepend_cls(text_concat_features, text_mask, 'text_mixed') # add cls token
            audio_concat_features, audio_attn_mask = self.prepend_cls(audio_concat_features, audio_mask_new, 'audio_mixed') # add cls token
            text_mixed_features = self.text_encoder(text_concat_features, src_key_padding_mask=(1-text_attn_mask).bool())
            audio_mixed_features = self.audio_encoder(audio_concat_features, src_key_padding_mask=(1-audio_attn_mask).bool())
            # fused features
            fused_hidden_states = torch.cat((text_mixed_features[:,0,:], audio_mixed_features[:,0,:]), dim=1) # Shape is [batch_size, 1024*4]

        # last linear output layer
        fused_correction = self.fused_output_layers(fused_hidden_states) # Shape is [batch_size, 1]
        if self.version == 'v1':
            fused_output = (T_output + A_output) / 2 + fused_correction
        else:
            fused_output = fused_correction
        
        return {
                'T': T_output, 
                'A': A_output, 
                'M': fused_output
        }
    


