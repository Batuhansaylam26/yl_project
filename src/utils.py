import os
import sys
import logging
import optuna
import torch
import logging.handlers
import time 
import wandb
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from neuralforecast.losses.pytorch import MSE
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from JsonHandler import JsonHandler
from lightning.pytorch.loggers import WandbLogger
def convert_to_nf_dataframe(
        df: pd.DataFrame, 
        time_col: str, 
        value_col: str, 
        exogenous_cols: Optional[list] = None,
        id_col: Optional[str] = None
) -> pd.DataFrame:
    nf_df = df.rename(columns={time_col: 'ds', value_col: 'y'})
    if id_col:
        nf_df = nf_df.rename(columns={id_col: 'unique_id'})
    else:
        nf_df['unique_id'] = 'series_1'  # Varsayılan seri kimliği
    columns = ['unique_id', 'ds']
    if exogenous_cols:
        columns.extend(exogenous_cols)
    if nf_df['ds'].dtype is not int or nf_df['ds'].dtype is not pd.Timestamp:
        nf_df['ds'] = nf_df['ds'].astype('int64')
    return nf_df[columns] , nf_df['y']





def setup_logging(log_dir: str, verbose: bool = False):

    os.makedirs(log_dir, exist_ok=True)
    
    # Timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console Handler (INFO level, simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    #Main Log File (DEBUG level, detailed format)
    main_file_handler = logging.FileHandler(
        f"{log_dir}/training_{timestamp}.log",
        mode='w'
    )
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_file_handler)
    
    #Episode Logger (separate file)
    episode_logger = logging.getLogger('episode')
    episode_logger.setLevel(logging.INFO)
    episode_handler = logging.FileHandler(
        f"{log_dir}/episodes_{timestamp}.log",
        mode='w'
    )
    episode_handler.setFormatter(detailed_formatter)
    episode_logger.addHandler(episode_handler)
    episode_logger.propagate = False  # Don't propagate to root
    
    #Step Logger (separate file)
    step_logger = logging.getLogger('step')
    step_logger.setLevel(logging.DEBUG)
    step_handler = logging.FileHandler(
        f"{log_dir}/steps_{timestamp}.log",
        mode='w'
    )
    step_handler.setFormatter(detailed_formatter)
    step_logger.addHandler(step_handler)
    step_logger.propagate = False  # Don't propagate to root
    
    # 5. JSON Handler (custom for structured logging)
    json_handler = JsonHandler(f"{log_dir}/training_{timestamp}.json")
    json_handler.setLevel(logging.INFO)
    json_logger = logging.getLogger('json')
    json_logger.addHandler(json_handler)
    json_logger.propagate = False
    
    logging.info("="*70)
    logging.info("🚀 LOGGING INITIALIZED")
    logging.info(f"   Log Directory: {log_dir}")
    logging.info(f"   Timestamp: {timestamp}")
    logging.info("="*70)
    
    return root_logger

def create_episode_summary_table(episode_stats: list):
    logger = logging.getLogger()
    
    logger.info("\n" + "="*70)
    logger.info("EPISODE SUMMARY TABLE")
    logger.info("="*70)
    
    # Header
    header = f"{'Episode':<10} {'Avg Reward':<15} {'Total Reward':<15} {'Best Model':<20} {'Steps':<10} {'Duration':<12}"
    logger.info(header)
    logger.info("-"*70)
    
    # Rows
    for stat in episode_stats:
        row = (f"{stat['episode']:<10} "
               f"{stat['avg_reward']:<15.6f} "
               f"{stat['total_reward']:<15.6f} "
               f"{stat['best_model']:<20} "
               f"{stat['steps']:<10} "
               f"{stat['duration']:<12.2f}s")
        logger.info(row)
    
    logger.info("="*70)
    
    # Overall stats
    avg_rewards = [s['avg_reward'] for s in episode_stats]
    
    logger.info("\nOVERALL STATISTICS:")
    logger.info(f"   Total Episodes: {len(episode_stats)}")
    logger.info(f"   Mean Avg Reward: {np.mean(avg_rewards):.6f} ± {np.std(avg_rewards):.6f}")
    logger.info(f"   Best Episode: {episode_stats[np.argmax(avg_rewards)]['episode']} "
                f"(Reward: {max(avg_rewards):.6f})")
    logger.info(f"   Total Training Time: {sum(s['duration'] for s in episode_stats):.2f}s")
    logger.info("="*70 + "\n")


def save_episode_csv(episode_stats: list, log_dir: str):
    df = pd.DataFrame(episode_stats)
    # Remove complex columns for CSV
    df_simple = df.drop(columns=['rewards', 'actions', 'action_distribution'], errors='ignore')
    csv_path = f"{log_dir}/episode_summary.csv"
    df_simple.to_csv(csv_path, index=False)
    logging.info(f"✓ Episode summary CSV saved: {csv_path}")


def nested_func_to_get_config(
        n_trials:int, 
        model_name: str,
        episode: int,
        step: int,
        experiment_name:str,
        early_stop_patience: int = 5,
        horizon:int = 20,
        futr_exog_list: Optional[list] = None
    ):
    def create_callbacks():
        callbacks = []

        dirpath = 'checkpoints/'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        # 1. Early Stopping
        early_stop_callback = EarlyStopping(
            monitor='valid_loss',           # Validation loss'u izle
            patience=early_stop_patience,  # N epoch iyileşme yoksa dur
            mode='min',                   # Loss minimize ediliyor
            verbose=True,
            min_delta=0.0001,             # Minimum değişim
            check_on_train_epoch_end=False
        )
        callbacks.append(early_stop_callback)
        
        # 2. Model Checkpoint (en iyi modeli kaydet)
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_loss',
            mode='min',
            save_top_k=1,                # Sadece en iyi modeli kaydet
            dirpath=dirpath,
            filename=f'best-{model_name}-ep{episode}_step{step}'+'{epoch:02d}-{valid_loss:.4f}',
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks




    base_trainer_config = {
        'optimizer': torch.optim.Adadelta,
        'optimizer_kwargs': {
            'rho': 0.75,
            'lr': 1e-3,
        },
        'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
        'lr_scheduler_kwargs': {
            'T_max': 10
        },
        'max_steps': 250,
        'val_check_steps': 10,
        'early_stop_patience_steps': early_stop_patience,
        'scaler_type': 'standard',
        'random_seed': 26,
        'accelerator': 'auto',
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'enable_checkpointing': True,
    }

    def timesNet_config(trial: optuna.trial.Trial):
        #if wandb.run is not None:
#            wandb.finish()
        if hasattr(trial, 'number'):
            trial_id_str = f"T{trial.number:02d}"
        else:
            # MockTrial durumu: WandB'de çöp veri oluşmaması için rastgele bir etiket
            trial_id_str = f"mock_{int(time.time())}"

        # 2. WandB için benzersiz bir ID (id) ve görünen isim (name)
        # 'id' aynı kalırsa WandB üzerine yazar. 'id' değişirse yeni kayıt açar.
        run_id = f"run_{model_name}_ep{episode:03d}_s{step:03d}_{trial_id_str}"
        run_name = f"{model_name}_Ep{episode}_S{step}_{trial_id_str}"
        
        wandb_logger = WandbLogger(
            project=experiment_name,
            name=run_name,
            id=run_id,              # Benzersiz ID: Üst üste yazmayı engeller
            group=f"Ep{episode:03d}_Step{step:03d}_{model_name}", # Modelleri gruplar
            reinit=True,            # Aynı süreçte yeni bir run başlatmaya izin verir
            settings=wandb.Settings(start_method="thread") # Çakışmaları önlemek için thread modu
        )
        config = {
            'input_size': trial.suggest_categorical('input_size', [24, 48, 72, 96]),
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'conv_hidden_size': trial.suggest_categorical('conv_hidden_size', [16, 32, 64, 128]),
            'top_k': trial.suggest_int('top_k', 3, 7),
            'num_kernels': trial.suggest_int('num_kernels', 4, 8),
            'encoder_layers': trial.suggest_int('encoder_layers', 1, 4),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            **base_trainer_config,
            # Lightning Trainer ayarları
            #'logger': wandb_logger,
            'logger': None,
            'callbacks' : create_callbacks()
        }
        if futr_exog_list is not None:
            config['futr_exog_list'] = futr_exog_list
        return config

    def GRU_config(trial: optuna.trial.Trial):
        #if wandb.run is not None:
#            wandb.finish()
        if hasattr(trial, 'number'):
            trial_id_str = f"T{trial.number:02d}"
        else:
            # MockTrial durumu: WandB'de çöp veri oluşmaması için rastgele bir etiket
            trial_id_str = f"mock_{int(time.time())}"

        # 2. WandB için benzersiz bir ID (id) ve görünen isim (name)
        # 'id' aynı kalırsa WandB üzerine yazar. 'id' değişirse yeni kayıt açar.
        run_id = f"run_{model_name}_ep{episode:03d}_s{step:03d}_{trial_id_str}"
        run_name = f"{model_name}_Ep{episode}_S{step}_{trial_id_str}"
        
        wandb_logger = WandbLogger(
            project=experiment_name,
            name=run_name,
            id=run_id,              # Benzersiz ID: Üst üste yazmayı engeller
            group=f"Ep{episode:03d}_Step{step:03d}_{model_name}", # Modelleri gruplar
            reinit=True,            # Aynı süreçte yeni bir run başlatmaya izin verir
            settings=wandb.Settings(start_method="thread") # Çakışmaları önlemek için thread modu
        )
        config = {
            'input_size': trial.suggest_categorical('input_size', [24, 48, 72, 96]),
            'encoder_hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
            'encoder_n_layers': trial.suggest_int('encoder_n_layers', 1, 3),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            **base_trainer_config,
            # Lightning Trainer ayarları
            #'logger': wandb_logger,
            'logger': None,
            'callbacks' : create_callbacks()
        }
        if futr_exog_list is not None:
            config['futr_exog_list'] = futr_exog_list
        return config

    def LSTM_config(trial: optuna.trial.Trial):
        #if wandb.run is not None:
#            wandb.finish()
        if hasattr(trial, 'number'):
            trial_id_str = f"T{trial.number:02d}"
        else:
            # MockTrial durumu: WandB'de çöp veri oluşmaması için rastgele bir etiket
            trial_id_str = f"mock_{int(time.time())}"

        # 2. WandB için benzersiz bir ID (id) ve görünen isim (name)
        # 'id' aynı kalırsa WandB üzerine yazar. 'id' değişirse yeni kayıt açar.
        run_id = f"run_{model_name}_ep{episode:03d}_s{step:03d}_{trial_id_str}"
        run_name = f"{model_name}_Ep{episode}_S{step}_{trial_id_str}"
        
        wandb_logger = WandbLogger(
            project=experiment_name,
            name=run_name,
            id=run_id,              # Benzersiz ID: Üst üste yazmayı engeller
            group=f"Ep{episode:03d}_Step{step:03d}_{model_name}", # Modelleri gruplar
            reinit=True,            # Aynı süreçte yeni bir run başlatmaya izin verir
            settings=wandb.Settings(start_method="thread") # Çakışmaları önlemek için thread modu
        )
        config = {
            'input_size': trial.suggest_categorical('input_size', [24, 48, 72, 96]),
            'encoder_hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
            'encoder_n_layers': trial.suggest_int('encoder_n_layers', 1, 3),
            'decoder_hidden_size': trial.suggest_categorical('decoder_hidden_size', [32, 64, 128, 256]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            **base_trainer_config,
            # Lightning Trainer ayarları
            #'logger': wandb_logger,
            'logger': None,
            'callbacks' : create_callbacks()
        }
        if futr_exog_list is not None:
            config['futr_exog_list'] = futr_exog_list
        return config

    def KAN_config(trial: optuna.trial.Trial):
        #if wandb.run is not None:
#            wandb.finish()
        if hasattr(trial, 'number'):
            trial_id_str = f"T{trial.number:02d}"
        else:
            # MockTrial durumu: WandB'de çöp veri oluşmaması için rastgele bir etiket
            trial_id_str = f"mock_{int(time.time())}"

        # 2. WandB için benzersiz bir ID (id) ve görünen isim (name)
        # 'id' aynı kalırsa WandB üzerine yazar. 'id' değişirse yeni kayıt açar.
        run_id = f"run_{model_name}_ep{episode:03d}_s{step:03d}_{trial_id_str}"
        run_name = f"{model_name}_Ep{episode}_S{step}_{trial_id_str}"
        
        wandb_logger = WandbLogger(
            project=experiment_name,
            name=run_name,
            id=run_id,              # Benzersiz ID: Üst üste yazmayı engeller
            group=f"Ep{episode:03d}_Step{step:03d}_{model_name}", # Modelleri gruplar
            reinit=True,            # Aynı süreçte yeni bir run başlatmaya izin verir
            settings=wandb.Settings(start_method="thread") # Çakışmaları önlemek için thread modu
        )
        config = {
            'input_size': trial.suggest_categorical('input_size', [24, 48, 72, 96]),
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
            'grid_size': trial.suggest_categorical('grid_size', [5, 10, 15, 20]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'spline_order': trial.suggest_categorical('spline_order', [2, 3, 4]),
            **base_trainer_config,
            # Lightning Trainer ayarları
            #'logger': wandb_logger,
            'logger': None,
            'callbacks' : create_callbacks()
        }
        if futr_exog_list is not None:
            config['futr_exog_list'] = futr_exog_list
        return config

    def VanillaTransformer_config(trial: optuna.trial.Trial):
        #if wandb.run is not None:
#            wandb.finish()
        if hasattr(trial, 'number'):
            trial_id_str = f"T{trial.number:02d}"
        else:
            # MockTrial durumu: WandB'de çöp veri oluşmaması için rastgele bir etiket
            trial_id_str = f"mock_{int(time.time())}"

        # 2. WandB için benzersiz bir ID (id) ve görünen isim (name)
        # 'id' aynı kalırsa WandB üzerine yazar. 'id' değişirse yeni kayıt açar.
        run_id = f"run_{model_name}_ep{episode:03d}_s{step:03d}_{trial_id_str}"
        run_name = f"{model_name}_Ep{episode}_S{step}_{trial_id_str}"
        
        wandb_logger = WandbLogger(
            project=experiment_name,
            name=run_name,
            id=run_id,              # Benzersiz ID: Üst üste yazmayı engeller
            group=f"Ep{episode:03d}_Step{step:03d}_{model_name}", # Modelleri gruplar
            reinit=True,            # Aynı süreçte yeni bir run başlatmaya izin verir
            settings=wandb.Settings(start_method="thread") # Çakışmaları önlemek için thread modu
        )
        config = {
            'input_size': trial.suggest_categorical('input_size', [24, 48, 72, 96]),
            'n_head': trial.suggest_categorical('nhead', [2, 4, 8]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'windows_batch_size': trial.suggest_categorical('windows_batch_size', [8, 16, 32]),
            **base_trainer_config,
            # Lightning Trainer ayarları
            #'logger': wandb_logger,
            'logger': None,
            'callbacks' : create_callbacks()
        }
        if futr_exog_list is not None:
            config['futr_exog_list'] = futr_exog_list
        return config
    # Configuration
    model_config = None
    if model_name == 'TimesNet':
        model_config = timesNet_config
    elif model_name == 'VanillaTransformer':
        model_config = VanillaTransformer_config
    elif model_name == 'GRU':
        model_config = GRU_config
    elif model_name == 'LSTM':
        model_config = LSTM_config
    elif model_name == 'KAN':
        model_config = KAN_config
    return {
        'h': horizon,
        'backend' : 'optuna',
        'loss': MSE(),                    # Loss function
        'valid_loss': MSE(),              # Validation loss
        'num_samples': n_trials,     # Optuna trial sayısı
        'config': model_config   # Model yapılandırması
    }
