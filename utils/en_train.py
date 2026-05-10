import torch
from torch import nn
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.context_model import rob_wavlm_cc_context, rob_wavlm_cme_context
from utils.en_model import rob_wavlm_cc, rob_wavlm_cme
import random
import numpy as np
from utils.data_loader import data_loader
from itertools import chain

# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str


class EnConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode = 'regression',
                loss_weights = {
                    'M': 1.0,
                    'T': 1.0,
                    'A': 1.0,
                },
                 model_save_path = 'checkpoint/',
                 learning_rate = 1e-5,
                 epochs = 20,
                 dataset_name = 'mosei',
                 early_stop = 10,
                 seed = 0,
                 dropout=0.3,
                 model='cc',
                 batch_size = 16,
                 multi_task = True,
                 model_size = 'small',
                 cme_version = 'v1',
                 num_hidden_layers = 1,
                 tasks = 'M',  # 'M' or 'MTA'
                 context = True,
                 text_context_len = 2,
                 audio_context_len = 1,
                 use_gated_fusion = False,  # Flag to enable gated fusion network
                 gpu_ids = '0',  # GPU IDs for multi-GPU training
                ):

        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.model_size = model_size
        self.cme_version = cme_version
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len
        self.use_gated_fusion = use_gated_fusion
        self.gpu_ids = [int(x) for x in gpu_ids.split(',')]
        self.use_multi_gpu = len(self.gpu_ids) > 1

        
        
class EnTrainer():
    def __init__(self, config):
 
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks
        
    def do_train(self, model, data_loader):    
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        total_loss = 0
        # Iterate through all batches
        for batch in tqdm(data_loader):                    
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)

            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            audio_context_mask = batch["audio_context_masks"].to(device)

            targets = batch["targets"].to(device).view(-1, 1)

            optimizer.zero_grad()  # Reset gradients to zero

            if self.config.context:
                outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask)
            else:
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
            
            # Compute loss with multi-task or single-task mode
            if self.config.multi_task:
                loss = 0.0         
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
                    # train_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item()*text_inputs.size(0)  
            else:
                loss = self.criterion(outputs['M'], targets)        
                total_loss += loss.item()*text_inputs.size(0)
        
            loss.backward()                   
            optimizer.step()  # Update model parameters
                
        total_loss = round(total_loss / len(data_loader.dataset), 4)
        # print('TRAIN loss:', total_loss)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()  # Set model to evaluation mode
        if self.config.multi_task:
            y_pred = {'M': [], 'T': [], 'A': []}
            y_true = {'M': [], 'T': [], 'A': []}
            total_loss = 0
            val_loss = {
                'M':0,
                'T':0,
                'A':0
            }
        else:
            y_pred = []
            y_true = []
            total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader):  # Iterate through all batches
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)

                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)

                targets = batch["targets"].to(device).view(-1, 1)

                # Forward pass with or without context
                if self.config.context:
                    outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                                    audio_mask, audio_context_inputs, audio_context_mask)
                else:
                    outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                
                # Compute validation loss
                if self.config.multi_task:
                    loss = 0.0         
                    for m in self.tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item()*text_inputs.size(0)
                    total_loss += loss.item()*text_inputs.size(0)
                    # Collect predictions
                    for m in self.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs['M'], targets)        
                    total_loss += loss.item()*text_inputs.size(0)

                    # Collect predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())

        if self.config.multi_task:
            for m in self.tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'], "  A_loss: ", val_loss['A'])

            eval_results = {}
            for m in self.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.tasks[0]]
            eval_results['Loss'] = total_loss 
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss
        
        return eval_results


def EnRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name,
                                                        text_context_length=config.text_context_len,
                                                        audio_context_length=config.audio_context_len)

    if config.context:
        if config.model == 'cc':
            model = rob_wavlm_cc_context(config).to(device)
        else:
            model = rob_wavlm_cme_context(config, use_gated_fusion=config.use_gated_fusion).to(device)
        for param in model.wavlm_model.feature_extractor.parameters():
            param.requires_grad = False
    else:
        if config.model == 'cc':
            model = rob_wavlm_cc(config).to(device)
        else:
            model = rob_wavlm_cme(config).to(device)
        for param in model.wavlm_model.feature_extractor.parameters():
            param.requires_grad = False
    
    # Setup for multi-GPU training if needed
    if config.use_multi_gpu:
        print(f"Using multi-GPU training: GPUs {config.gpu_ids}")
        print(f"  Primary GPU: cuda:{config.gpu_ids[0]}")
        print(f"  Number of secondary GPUs: {len(config.gpu_ids)-1}")
        print(f"  Effective batch_size: {config.batch_size} * {len(config.gpu_ids)} = {config.batch_size * len(config.gpu_ids)}")
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    else:
        print(f"Using single GPU training: GPU {config.gpu_ids[0]}")

    trainer = EnTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader,"VAL")

        if eval_results['Loss']<lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            # Handle DataParallel wrapped model
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            # Add suffix to checkpoint filename based on gated fusion flag
            gate_suffix = '_gated' if config.use_gated_fusion else ''
            torch.save(model_to_save.state_dict(), config.model_save_path+f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}{gate_suffix}.pth')
            best_epoch = epoch
        if eval_results['Has0_acc_2']>=highest_eval_acc:
            highest_eval_acc = eval_results['Has0_acc_2']
            # Handle DataParallel wrapped model
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            # Add suffix to checkpoint filename based on gated fusion flag
            gate_suffix = '_gated' if config.use_gated_fusion else ''
            torch.save(model_to_save.state_dict(), config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}{gate_suffix}.pth')
        if epoch - best_epoch >= config.early_stop:
            break
    
    # Handle DataParallel wrapping when loading checkpoint
    gate_suffix = '_gated' if config.use_gated_fusion else ''
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    # Load best model based on highest validation accuracy
    model_to_load.load_state_dict(torch.load(config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}{gate_suffix}.pth'))        
    test_results_loss = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))

    # Load best model based on lowest validation loss
    model_to_load.load_state_dict(torch.load(config.model_save_path+f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}{gate_suffix}.pth'))
    test_results_acc = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))