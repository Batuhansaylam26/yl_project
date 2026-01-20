import numpy as np
import pandas as pd 
from gymnasium import Env, spaces
from neuralforecast.auto import AutoTimesNet, AutoVanillaTransformer, AutoGRU, AutoLSTM, AutoKAN
from neuralforecast import NeuralForecast
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import mean_squared_error, r2_score
from utils import create_callbacks, timesNet_config, LSTM_config, GRU_config, KAN_config, VanillaTransformer_config, get_auto_model_config, setup_logging, get_logger, create_callbacks
import os
import time
import json
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from math import pi


class TestEnv(Env):
    def __init__(self, 
                 train_data: pd.DataFrame,
                 val_size: int,
                 test_data: pd.DataFrame,
                 horizon: int = 1,
                 max_steps: int = 1000,
                 model_n_trials: int = 10,
                 model_patience: int = 5,
                 exog_vars: Optional[List[str]] = None,  
                 use_exog: bool = False,
                 experiment_name = "Experiment_v3"
    ):
        super(TestEnv, self).__init__()
        
        self.train_data = train_data
        self.val_size = val_size
        self.test_data = test_data
        self.horizon = horizon
        self.max_steps = max_steps
        self.model_n_trials = model_n_trials
        self.model_patience = model_patience
        self.use_exog = use_exog
        self.exog_vars = exog_vars if exog_vars else []
        self.experiment_name = experiment_name

        # Kullanılacak modellerin isim listesi
        self.models = [
            "TimesNet",
            "VanillaTransformer",
            "GRU",
            "LSTM",
            "KAN"
        ]
        self.len_event_space = len(self.models)

        # Action Space tanımlamaları
        self.action_space = spaces.Discrete(self.len_event_space)  

        # Observation Space tanımlamaları
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(self.len_event_space*5,),
            dtype=np.float32
        )
        
        # Episode verileri
        self.current_step = 0
        self.episode_rewards = []
        self.episode_actions = []

        # Model performans verileri
        self.models_performance = {
            model: {
                "mse_history": [],
                "r2_history": [],
                "reward_history": [],  
                "count": 0,
                "total_rewards": 0,
                "total_time": 0.0  
            } for model in self.models
        }
        
        # Genel istatistikler
        self.all_results = []
        
        self.episode_start_time = None
        self.model_training_times = {}
        # Loglama ve çıktı dizini ayarları
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"outputs/run_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        
        # Setup logging system
        setup_logging(log_dir=f"{self.output_dir}/logs", verbose=True)
        
        # Get loggers
        self.logger = logging.getLogger(__name__)
        self.episode_logger = logging.getLogger('episode')
        self.step_logger = logging.getLogger('step')
        self.json_logger = logging.getLogger('json')
        
        # Log initial configuration
        self.logger.info("="*70)
        self.logger.info("🚀 TRAINING ENVIRONMENT INITIALIZED")
        self.logger.info(f"   Timestamp: {timestamp}")
        self.logger.info(f"   Output Directory: {self.output_dir}")
        self.logger.info("="*70)
        self.logger.info("Configuration:")
        self.logger.info(f"   Horizon: {self.horizon}")
        self.logger.info(f"   Max Steps: {self.max_steps}")
        self.logger.info(f"   Model n_trials: {self.model_n_trials}")
        self.logger.info(f"   Model patience: {self.model_patience}")
        self.logger.info(f"   Models: {', '.join(self.models)}")
        self.logger.info(f"   Use Exogenous: {self.use_exog}")
        self.logger.info(f"   Exogenous vars: {self.exog_vars if self.use_exog else 'None'}")
        self.logger.info("="*70)



    def get_observation_data(self):
        self.logger.debug(f"Generating observation data at step {self.current_step}")
        data = []
        for model in self.models:
            perf = self.models_performance[model]
            mean_mse = np.mean(perf["mse_history"]) if perf["mse_history"] else 0
            mean_r2 = np.mean(perf["r2_history"]) if perf["r2_history"] else 0
            count = perf["count"]
            last_mse = perf["mse_history"][-1] if perf["mse_history"] else 0
            last_r2 = perf["r2_history"][-1] if perf["r2_history"] else 0
            
            data.extend([mean_mse, mean_r2, count, last_mse, last_r2])
        
        return np.array(data, dtype=np.float32)
    
    def get_info(self):
        self.logger.debug(f"Generating info at step {self.current_step}")
        return {
            "step": self.current_step,
            "episode_rewards": self.episode_rewards,
            "episode_actions": self.episode_actions,
            "best_model": self.get_best_model()
        }
    
    def get_best_model(self) -> str:
        self.logger.debug("Determining best model based on average rewards")
        best_model = None
        best_reward = -np.inf
        
        for model, stats in self.models_performance.items():
            if stats['count'] > 0:
                avg_reward = stats['total_rewards'] / stats['count']
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_model = model
        
        return best_model if best_model else self.models[0]
    
    def create_model(self, model_name: str, episode:int):
        self.logger.debug(f"Creating model: {model_name}") 
        if model_name == 'TimesNet':
            config = get_auto_model_config(
                horizon=self.horizon,
                n_trials=self.model_n_trials,
                model_name=model_name
            )
            if self.use_exog and self.exog_vars:
                config['hist_exog_list'] = self.exog_vars
            name = f"TimesNet_ep{episode}_step{self.current_step}"
            logger = get_logger(name=name, project=self.experiment_name)
            config['logger'] = logger
            callbacks = create_callbacks(early_stop_patience=self.model_patience, model_name=name)
            config['callbacks'] = callbacks
            self.logger.debug(f"TimesNet config: {config}")
            return AutoTimesNet(**config)
        elif model_name == 'VanillaTransformer':
            config = get_auto_model_config(
                horizon=self.horizon,
                n_trials=self.model_n_trials,
                model_name=model_name
            )
            if self.use_exog and self.exog_vars:
                config['hist_exog_list'] = self.exog_vars
            name = f"VanillaTransformer_ep{episode}_step{self.current_step}"
            logger = get_logger(name=name, project=self.experiment_name)
            config['logger'] = logger
            callbacks = create_callbacks(early_stop_patience=self.model_patience, model_name=name)
            config['callbacks'] = callbacks
            self.logger.debug(f"VanillaTransformer config: {config}")
            return AutoVanillaTransformer(**config)
        elif model_name == 'GRU':
            config = get_auto_model_config(
                horizon=self.horizon,
                n_trials=self.model_n_trials,
                model_name=model_name
            )
            if self.use_exog and self.exog_vars:
                config['hist_exog_list'] = self.exog_vars
            name = f"GRU_ep{episode}_step{self.current_step}"
            logger = get_logger(name=name, project=self.experiment_name)
            config['logger'] = logger
            callbacks = create_callbacks(early_stop_patience=self.model_patience, model_name=name)
            config['callbacks'] = callbacks
            self.logger.debug(f"GRU config: {config}")
            return AutoGRU(**config)
        elif model_name == 'LSTM':
            config = get_auto_model_config(
                horizon=self.horizon,
                n_trials=self.model_n_trials,
                model_name=model_name
            )
            if self.use_exog and self.exog_vars:
                config['hist_exog_list'] = self.exog_vars
            name = f"LSTM_ep{episode}_step{self.current_step}"
            logger = get_logger(name=name, project=self.experiment_name)
            config['logger'] = logger
            callbacks = create_callbacks(early_stop_patience=self.model_patience, model_name=name)
            config['callbacks'] = callbacks
            self.logger.debug(f"LSTM config: {config}")
            return AutoLSTM(**config)
        elif model_name == 'KAN':
            config = get_auto_model_config(
                horizon=self.horizon,
                n_trials=self.model_n_trials,
                model_name=model_name
            )
            if self.use_exog and self.exog_vars:
                config['hist_exog_list'] = self.exog_vars
            name = f"KAN_ep{episode}_step{self.current_step}"
            logger = get_logger(name=name, project=self.experiment_name)
            config['logger'] = logger
            callbacks = create_callbacks(early_stop_patience=self.model_patience, model_name=name)
            config['callbacks'] = callbacks
            self.logger.debug(f"KAN config: {config}")
            return AutoKAN(**config)
        else:
            self.logger.debug(f"Error: Unknown model {model_name}")
            raise ValueError(f"Unknown model: {model_name}")
    
    def calculate_reward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        self.logger.debug("Calculating reward based on MSE and R²")
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MSE'yi normalize et (düşük iyi)
        mse_normalized = 1 / (1 + mse)
        
        # R²'yi normalize et (yüksek iyi, -inf ile 1 arası olabilir)
        r2_normalized = max(0, min(1, (r2 + 1) / 2))
        
        # Reward: ağırlıklı kombinasyon
        reward = 0.3 * mse_normalized + 0.7 * r2_normalized
        
        return reward, mse, r2
    
    def step(self, action: int, episode:int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.logger.debug(f"Step {self.current_step + 1}: Action received - {action}")
        model_name = self.models[action]
        
        self.logger.debug(f"\n{'='*70}")
        self.logger.debug(f"Step {self.current_step + 1}/{self.max_steps}: Training {model_name}")
        self.logger.debug(f"{'='*70}")
        
        # Model training start
        model_start_time = time.time()
        
        try:
            # Modeli oluştur ve eğit
            model = self.create_model(model_name, episode)
            nf = NeuralForecast(models=[model], freq=1)
            nf.fit(self.train_data, val_size=self.val_size, verbose=False)
            
            # Model training end
            model_training_time = time.time() - model_start_time
            
            # Store training time
            if model_name not in self.model_training_times:
                self.model_training_times[model_name] = []
            self.model_training_times[model_name].append(model_training_time)
            
            self.logger.debug(f"{model_name} training completed in {model_training_time:.2f}s")
            
            # Tahmin yap
            forecasts = nf.predict()
            y_pred = forecasts[model_name].values
            y_true = self.test_data['y'].values[:len(y_pred)]
            
            # Reward hesapla
            reward, mse, r2 = self.calculate_reward(y_true, y_pred)
            
            # İstatistikleri güncelle
            self.models_performance[model_name]['mse_history'].append(mse)
            self.models_performance[model_name]['r2_history'].append(r2)
            self.models_performance[model_name]['reward_history'].append(reward)  
            self.models_performance[model_name]['count'] += 1
            self.models_performance[model_name]['total_rewards'] += reward
            self.models_performance[model_name]['total_time'] += model_training_time  
            
            self.logger.debug(f"📊 Results: MSE={mse:.6f}, R²={r2:.6f}, Reward={reward:.6f}")
            
            # Episode bilgilerini kaydet
            self.episode_rewards.append(reward)
            self.episode_actions.append(action)
            
            # Genel sonuçları kaydet
            self.all_results.append({
                'step': self.current_step,
                'model': model_name,
                'mse': mse,
                'r2': r2,
                'reward': reward,
                'training_time': model_training_time  
            })
            
        except Exception as e:
            self.logger.debug(f"Error training {model_name}: {str(e)}")
            reward = -1.0
            mse, r2 = np.inf, -np.inf
            model_training_time = time.time() - model_start_time
        
        # Step'i artır
        self.current_step += 1
        
        # Episode bitip bitmediğini kontrol et
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Yeni observation
        observation = self.get_observation_data()
        info = self.get_info()
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.logger.debug("Resetting environment for new episode")
        super().reset(seed=seed)
        
        self.episode_start_time = time.time()
        
        self.current_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        
        # Model istatistiklerini SIFIRLA
        self.models_performance = {
            model: {
                'mse_history': [],
                'r2_history': [],
                'reward_history': [],  
                'count': 0,
                'total_rewards': 0.0,
                'total_time': 0.0  
            }
            for model in self.models
        }

        self.logger.debug("\n" + "="*70)
        self.logger.debug("Environment Reset")
        self.logger.debug("="*70 + "\n")

        observation = self.get_observation_data()
        info = self.get_info()
        
        return observation, info
    
    def render(self):
        if len(self.all_results) == 0:
            self.logger.debug("Henüz sonuç yok!")
            return
        
        # Episode toplam süre
        episode_total_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        # ============ TEXT REPORT ============
        self.logger.debug("\n" + "="*70)
        self.logger.debug(f"EPISODE SUMMARY - Step: {self.current_step}/{self.max_steps}")
        self.logger.debug("="*70)
        self.logger.debug(f"Total Episode Time: {episode_total_time:.2f}s ({episode_total_time/60:.2f}m)")
        self.logger.debug("="*70)
        
        summary_data = []
        
        for model in self.models:
            stats = self.models_performance[model]
            if stats['count'] > 0:
                avg_mse = np.mean(stats['mse_history'])
                avg_r2 = np.mean(stats['r2_history'])
                avg_reward = stats['total_rewards'] / stats['count']
                total_time = stats.get('total_time', 0)
                avg_time = total_time / stats['count'] if stats['count'] > 0 else 0
                
                self.logger.debug(f"\n🤖 {model}:")
                self.logger.debug(f"   Selection: {stats['count']} times")
                self.logger.debug(f"   Average MSE: {avg_mse:.6f}")
                self.logger.debug(f"   Average R²:  {avg_r2:.6f}")
                self.logger.debug(f"   Average Reward: {avg_reward:.6f}")
                self.logger.debug(f"   Total Training Time: {total_time:.2f}s")
                self.logger.debug(f"   Average Training Time: {avg_time:.2f}s")
                
                summary_data.append({
                    'model': model,
                    'count': stats['count'],
                    'avg_mse': avg_mse,
                    'avg_r2': avg_r2,
                    'avg_reward': avg_reward,
                    'total_time': total_time,
                    'avg_time': avg_time
                })
        
        best_model = self.get_best_model()
        self.logger.debug(f"\n🏆 Best Model: {best_model}")
        self.logger.debug("="*70)
        
        # ============ SAVE JSON SUMMARY ============
        json_summary = {
            'timestamp': datetime.now().isoformat(),
            'episode_total_time': episode_total_time,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'best_model': best_model,
            'models': summary_data,
            'all_results': self.all_results
        }
        
        with open(f"{self.output_dir}/summary.json", "w") as f:
            json.dump(json_summary, f, indent=2)
        
        self.logger.debug(f"\nSummary saved to: {self.output_dir}/summary.json")
        
        # ============ PLOTS ============
        if summary_data:
            self._create_plots(summary_data)
            self.logger.debug(f"Plots saved to: {self.output_dir}/plots/")
        
        self.logger.debug("="*70 + "\n")
    
    def _create_plots(self, summary_data):
        self.logger.debug("Creating performance plots")
        if not summary_data:
            return
        
        # Renk paleti
        colors = plt.cm.Set3(range(len(summary_data)))
        
        # ============ PLOT 1: Model Selection Count ============
        fig, ax = plt.subplots(figsize=(10, 6))
        models = [d['model'] for d in summary_data]
        counts = [d['count'] for d in summary_data]
        
        bars = ax.bar(models, counts, color=colors)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Selection Count', fontsize=12)
        ax.set_title('Model Selection Frequency', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/01_selection_count.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ============ PLOT 2: Average Rewards ============
        fig, ax = plt.subplots(figsize=(10, 6))
        rewards = [d['avg_reward'] for d in summary_data]
        
        bars = ax.bar(models, rewards, color=colors)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Average Reward by Model', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/02_avg_reward.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ============ PLOT 3: MSE and R² ============
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # MSE
        mses = [d['avg_mse'] for d in summary_data]
        bars1 = ax1.bar(models, mses, color=colors)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Average MSE', fontsize=12)
        ax1.set_title('Average MSE by Model (Lower is Better)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax1.tick_params(axis='x', rotation=45)
        
        # R²
        r2s = [d['avg_r2'] for d in summary_data]
        bars2 = ax2.bar(models, r2s, color=colors)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Average R²', fontsize=12)
        ax2.set_title('Average R² by Model (Higher is Better)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/03_mse_r2.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ============ PLOT 4: Training Time ============
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_times = [d['avg_time'] for d in summary_data]
        
        bars = ax.bar(models, avg_times, color=colors)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Average Training Time (s)', fontsize=12)
        ax.set_title('Average Training Time by Model', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/04_training_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ============ PLOT 5: Reward History (Line) ============
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model in self.models:
            stats = self.models_performance[model]
            if len(stats['reward_history']) > 0:
                ax.plot(range(1, len(stats['reward_history']) + 1),
                       stats['reward_history'],
                       marker='o',
                       label=model,
                       linewidth=2,
                       markersize=6)
        
        ax.set_xlabel('Selection Number', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Reward History Over Selections', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/05_reward_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ============ PLOT 6: Comprehensive Comparison (Radar) ============
        if len(summary_data) >= 3:
            self._create_radar_plot(summary_data)
    
    def _create_radar_plot(self, summary_data):
        self.logger.debug("Creating radar plot for model comparison")
        labels = ['Avg Reward', 'Avg R²', 'Speed\n(1/time)']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angles += angles[:1]
        
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        
        colors = plt.cm.Set3(range(len(summary_data)))
        
        # Normalize speed (inverse of avg_time)
        max_speed = max([1.0 / d['avg_time'] if d['avg_time'] > 0 else 0 for d in summary_data])
        
        for idx, data in enumerate(summary_data):
            speed_normalized = (1.0 / data['avg_time'] / max_speed) if data['avg_time'] > 0 and max_speed > 0 else 0
            
            values = [
                data['avg_reward'],
                max(0, min(1, (data['avg_r2'] + 1) / 2)),  # Normalize R² to 0-1
                speed_normalized
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=data['model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title('Model Performance Comparison\n(Normalized)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/06_radar_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def close(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"All outputs saved to: {self.output_dir}")
            self.logger.info(f"{'='*70}")
            self.logger.info("Closing log file.")
            logging.shutdown()