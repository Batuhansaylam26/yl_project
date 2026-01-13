import os
import sys
import logging
import logging.handlers
import pandas as pd
from datetime import datetime
from typing import Optional
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from neuralforecast.losses.pytorch import MSE
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from JsonHandler import JsonHandler
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
    columns = ['unique_id', 'ds', 'y']
    if exogenous_cols:
        columns.extend(exogenous_cols)
    if nf_df['ds'].dtype is not int or nf_df['ds'].dtype is not pd.Timestamp:
        nf_df['ds'] = nf_df['ds'].astype('int64')
    return nf_df[columns]


def create_callbacks(early_stop_patience: int = 5, model_name: str = ''):
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
            filename=f'best-{model_name}-'+'{epoch:02d}-{valid_loss:.4f}',
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks

def get_logger(log_dir: str = 'logs/', name: str = 'neuralforecast'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=name,
        version=None,
        default_hp_metric=False
    )
    return logger

# Configuration
def get_auto_model_config(  
                          n_trials, 
                          callbacks, 
                          horizon:int = 20,
                          logger_name: str = 'neuralforecast'
                          
) -> dict:
    return {
        'h': horizon,
        'loss': MSE(),                    # Loss function
        'valid_loss': MSE(),              # Validation loss
        'num_samples': n_trials,     # Optuna trial sayısı
        'max_steps': 1000,
        'config': {
            'val_check_steps': 1,            # Her  step'te validation
            'batch_size': tune.choice([32, 64, 128]),               # Auto-tune
            'learning_rate': tune.loguniform(1e-5, 1e-1),            # Auto-tune
            'callbacks': callbacks,
            'accelerator': 'auto',        # GPU varsa kullan
            'enable_progress_bar': True, # Progress bar kapat (gym için)
            'input_size': tune.randint(7, 24),
            'enable_model_summary': True,
            'logger': get_logger(logger_name),
            'enable_checkpointing': True,
            'enable_progress_bar': True
        }
    }

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