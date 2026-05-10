# CAGF
CAGF: Context-Aware Gated Fusion Network for Multimodal Sentiment Analysis

## Environment setup
1. create a new environment using conda
2. ```pip install -r requirements.txt```

## Download Data
The three datasets (CMU-MOSI, CMU-MOSEI, and CH-SIMS) are available from this link: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

## Data Directory
```
/data/
    mosi/
        raw/
        label.csv
    mosei/
        raw/
        label.csv
    sims/
        raw/
        label.csv
```

## Settings
### GPU: NVIDIA H20

### Pre-trained models:

#### For CMU-MOSI, CMU-MOSEI
1. Text Encoder: RoBERTa-large, link: https://huggingface.co/FacebookAI/roberta-large
2. Audio Encoder: WavLM-large, link: https://huggingface.co/microsoft/wavlm-large

#### For CH-SIMS
1. Text Encoder: Chinese-RoBERTa-WWM-EXT-Large, link: https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
2. Audio Encoder: Chinese-HuBERT-large, link: https://huggingface.co/TencentGameMate/chinese-hubert-large

#### Note: All pre-trained models were initialized with their publicly available checkpoints

## Audio Extraction
Before running the training code, please extract the audio from the raw video files.

```
python extract_audio.py

options:
  --dataset DATASET     dataset name (mosi, mosei, or sims)
```

## Train with both text and audio features
```
python run.py

options (optional):
  --seed SEED           random seed (default: 1)
  
  --batch_size BATCH_SIZE
                        batch size (default: 8)
                        
  --lr LR               learning rate (default: 5e-6)
  
  --model MODEL         concatenate(cc) or cross-modality encoder(cme) (default: cme)
  
  --cme_version VERSION
                        version (default: v1)
                        
  --dataset DATASET     dataset name: mosi, mosei, sims (default: mosi)
  
  --num_hidden_layers NUM_HIDDEN_LAYERS
                        number of hidden layers for cross-modality encoder (default: 5)
                        
  --tasks TASKS        losses to train: M: multi-modal, T: text, A: audio
                       (default: MTA)
                       
  --context CONTEXT    incorporate context or not (default: True)
  
  --text_context_len TEXT_CONTEXT_LEN
                        (default: 2)
  
  --audio_context_len AUDIO_CONTEXT_LEN
                        (default: 1)
                        
  --use_gated_fusion    use gated fusion network for rob_wavlm_cme_context model
  
  --gpu_ids GPU_IDS     gpu ids for multi-gpu training, e.g., "0,1,2,3" or "0" for single gpu (default: 0)
```
