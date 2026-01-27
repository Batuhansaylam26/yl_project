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
from nolitsa import delay, dimension, utils as nolitsa_utils
import matplotlib.pyplot as plt
def convert_to_nf_dataframe(
        df: pd.DataFrame, 
        time_col: str, 
        value_col: str, 
        use_exogenous_cols: bool = False,
        id_col: Optional[str] = None
) -> pd.DataFrame:
    
    nf_df = df.copy()
    nf_df = nf_df.rename(columns={time_col: 'ds', value_col: 'y'})
    
    if id_col:
        nf_df = nf_df.rename(columns={id_col: 'unique_id'})
    else:
        nf_df['unique_id'] = 'series_1'
    
    # Sütun listesi
    columns = ['unique_id', 'ds', 'y']  # 'y' eksikti!
    
    if use_exogenous_cols:
        exogenous_cols = [col for col in nf_df.columns if col not in ['unique_id', 'ds', 'y']]
        columns.extend(exogenous_cols)
    
    # dtype kontrolü düzeltildi
    if nf_df['ds'].dtype is not int or nf_df['ds'].dtype is not pd.Timestamp:
        nf_df['ds'] = nf_df['ds'].astype('int64')
    
    return nf_df[columns]




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
        input_size:int = None,
        early_stop_patience: int = 5,
        model_max_steps: int = 250,
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
        'max_steps': model_max_steps,
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
            'input_size': input_size if input_size else trial.suggest_categorical('input_size', [24, 48, 72, 96]),
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
            'input_size': input_size if input_size else trial.suggest_categorical('input_size', [24, 48, 72, 96]),
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
            'input_size': input_size if input_size else trial.suggest_categorical('input_size', [24, 48, 72, 96]),
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
            'input_size': input_size if input_size else trial.suggest_categorical('input_size', [24, 48, 72, 96]),
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
            'input_size': input_size if input_size else trial.suggest_categorical('input_size', [24, 48, 72, 96]),
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
def opt_ami(signal,max_tau, local_min=True):
    ami_values = delay.dmi(signal, maxtau=max_tau)
    if local_min:
        tau_opt = np.where(np.diff(ami_values) > 0)[0][0]
        return ami_values, tau_opt, None, None
    else:
        threshold_ami = ami_values[0] * (1 / np.exp(1))
        idx = np.where(ami_values <= threshold_ami)[0]
        tau_opt = idx[0]
    return ami_values, tau_opt, threshold_ami, idx
def corr_plot(name, y_label_title, tau_search, corr_values, threshold, tau_opt, output_path ):
    plt.figure(figsize=(10, 6))
    plt.title('Kaos Analizi Parametre Seçimi', fontsize=16)
    if threshold:
        plt.axhline(y=threshold, color='green', linestyle='--', label='1/e Eşiği (Traditional)')
    plt.axvline(x=tau_opt, color='r', linestyle='--', label=f'Belirlenen Tau {tau_opt}')
    plt.plot(tau_search, corr_values, 'b-', linewidth=2, label='Correlation Value')
    plt.plot(tau_opt, corr_values[tau_opt], 'ro', markersize=8, label=f'Optimal Tau = {tau_opt}')
    plt.xlabel('Gecikme (Tau)')
    plt.ylabel(y_label_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if output_path:
        path = os.path.join(output_path, name)
        plt.savefig(path)
    plt.show()

def fnn_plot(name, dims, fnn_values, m_opt, output_path):
    plt.figure(figsize=(10, 6))
    plt.title('False Nearest Neighbors (Boyut Seçimi)', fontsize=16)
    plt.plot(dims, fnn_values * 100, 'go-', linewidth=2, label='FNN Oranı')
    plt.axhline(y=1, color='r', linestyle=':', label='%1 Eşiği')
    plt.plot(m_opt, fnn_values[m_opt-1] * 100, 'rs', markersize=8, label=f'Optimal m = {m_opt}')
    plt.xlabel('Boyut (m)')
    plt.ylabel('Yalancı Komşu Oranı (%)')
    plt.yscale('log') # Küçük değerleri görmek için logaritmik ölçek
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if output_path:
        path = os.path.join(output_path, name)
        plt.savefig(path)
    plt.show()
def get_takens_embedding(
        time_series: pd.DataFrame,
        max_tau: int = 1000,
        output_path:str = None
    ) -> pd.DataFrame:
    if not output_path:
        output_path = '../preprocess_plots'
        os.makedirs(output_path, exist_ok=True)
    maxtau = max_tau
    signal  = time_series['y'].values
    tau_search = np.arange(maxtau)
    values_local_min, opt_ami_local_min, _, _ = opt_ami(signal, maxtau, local_min=True)
    corr_plot('Local_Mini.png', 'Average Mutual Information (AMI)', tau_search, values_local_min, None, opt_ami_local_min, None, output_path=output_path)    
    values_threshold, opt_ami_threshold, threshold_ami, idx_ami = opt_ami(signal, maxtau, local_min=False)
    corr_plot('Threshold.png', 'Average Mutual Information (AMI)', tau_search, values_threshold, threshold_ami, opt_ami_threshold, None, output_path=output_path)
    tau_candidates = {
        'AMI Local Min': opt_ami_local_min,
        'AMI Threshold': opt_ami_threshold,
    }

    # None/invalid olanları filtrele
    valid = {k: v for k, v in tau_candidates.items() if v is not None and v > 0}

    # Minimum olanı bul
    tau_key = min(valid, key=valid.get)
    tau_opt = valid[tau_key]
    print(f"Seçilen: {tau_key} = {tau_opt}")
    print(f"Tümü: {tau_candidates}")




    dims = np.arange(1, 10)
    f1, f2, f3 = dimension.fnn(signal, 
                               tau=tau_opt, 
                               dim=dims, 
                               R=10.0, 
                               A=2.0, 
                               window=50, 
                               metric='euclidean', 
                               parallel=True)
    
    # 4. Optimal m değerini belirleme (%1 eşiği)
    # f1 değerinin 0.01'in altına düştüğü ilk indeksi buluyoruz
    threshold = 0.01
    valid_dims = np.where(f1 < threshold)[0]

    
    
    if len(valid_dims) > 0:
        optimal_m = dims[valid_dims[0]]
    else:
        # Eğer %1 altına düşmüyorsa en düşük değeri veren boyutu al
        optimal_m = dims[np.argmin(f1)]
        print(f"Uyarı: FNN oranı %1'in altına düşmedi. En düşük değer seçildi.")

    

    print(f"Hesaplanan Optimal Boyut (m): {optimal_m}")
    print(f"FNN Oranları (f1): {np.round(f1, 4)}")
    fnn_plot('FNN_plot.png', dims, f1, optimal_m, None)
    reconstructed_z = nolitsa_utils.reconstruct(signal, tau=tau_opt, dim=optimal_m)

    # Çıktı Shape: (N - (m-1)*tau, m)
    # Yani her bir satır, m-boyutlu bir koordinattır.
    print(f"Orijinal veri uzunluğu: {len(signal)}")
    print(f"Embedding sonrası yapı: {reconstructed_z.shape}")

        # Embedding sonrası kayıp: (m-1) * tau
    offset = (optimal_m - 1) * tau_opt

    # Zaman sütunlarını al ve index'i resetle
    time_df = time_series.loc[offset:, ['unique_id', 'U', 'ds']].reset_index(drop=True)

    # Embedding dataframe
    emb_df = pd.DataFrame(reconstructed_z, columns=[f'Z_emb_{i+1}' for i in range(optimal_m)])
    emb_df = emb_df.rename(columns={'Z_emb_1': 'y'})  # İlk sütunu 'y' olarak yeniden adlandır

    # Uzunlukları kontrol et
    print(f"Time df: {len(time_df)}, Emb df: {len(emb_df)}")

    # Birleştir
    new_df = pd.concat([time_df, emb_df['y']], axis=1)

    

   
    
    return new_df, tau_opt, offset