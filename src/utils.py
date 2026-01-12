import os
import pandas as pd
from typing import Optional
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from neuralforecast.losses.pytorch import MSE
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
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