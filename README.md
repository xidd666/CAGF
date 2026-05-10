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
                        
  --use_gated_fusion    Use gated fusion network for rob_wavlm_cme_context model
  
  --gpu_ids GPU_IDS     GPU ids for multi-gpu training, e.g., "0,1,2,3" or "0" for single gpu (default: 0)
```

## Save waveform images for audio clips

You can generate a full waveform or extract and save the waveform of a specific word spoken in the audio.

```
python save_waveform.py

  --video_id VIDEO_ID

  --clip_id CLIP_ID

  --dataset DATASET    Dataset name (mosi, mosei)

  --word TARGET_WORD    (default: None)
```

## t-SNE Feature Distribution Visualization for CMU-MOSI or CMU-MOSEI

Generate t-SNE visualizations comparing feature representations across three processing methods: raw features, concatenation fusion, and CAGF (CME + Gated Fusion).

Creates a 3-subplot figure showing how different fusion strategies affect feature distributions.

```
python tsne_visualization.py

  --checkpoint CHECKPOINT         Path to the CAGF model checkpoint
                                  Example: checkpoint/cagf_model.pth
                                  
  --checkpoint_cc CHECKPOINT_CC   Path to the Concatenation fusion model checkpoint
                                  Example: checkpoint/cc_model.pth

  --dataset DATASET               Dataset name (mosi, mosei)
                                  
  --seed SEED                     Random seed for reproducibility (default: 1)
  
  --batch_size BATCH_SIZE         Batch size for feature extraction (default: 8)
  
  --text_context_len LENGTH       Length of text context window (default: 5)
  
  --audio_context_len LENGTH      Length of audio context window (default: 5)
  
  --perplexity PERPLEXITY         t-SNE perplexity parameter (default: 30)
                                  Recommended range: 20-50
                                  
  --output OUTPUT                 Output image path (default: tsne_mosi.png)
```


## t-SNE Feature Distribution Visualization for CH-SIMS

Generate t-SNE visualizations comparing feature representations across three processing methods for the CH-SIMS dataset: raw features, concatenation fusion, and CME (Cross-Modal Encoder). 

Creates a 3-subplot figure showing how different fusion strategies affect feature distributions in Chinese multimodal sentiment analysis.

```
python tsne_visualization_sims.py

  --checkpoint CHECKPOINT         Path to the CME (Cross-Modal Encoder) checkpoint
                                  Example: checkpoint/cme_sims.pth
                                  
  --checkpoint_cc CHECKPOINT_CC   Path to the Concatenation (CC) fusion model checkpoint
                                  Example: checkpoint/cc_sims.pth

  --seed SEED                     Random seed for reproducibility (default: 1)
  
  --batch_size BATCH_SIZE         Batch size for feature extraction (default: 8)
  
  --num_hidden_layers NUM_HIDDEN_LAYERS
                                  Number of CME cross-modal encoder layers (default: 5)
                                  Should match the training configuration
                                  
  --perplexity PERPLEXITY         t-SNE perplexity parameter (default: 30)
                                  Recommended range: 20-50
                                  
  --output OUTPUT                 Output image path (default: tsne_sims.png)

```
