import numpy as np
import json
from typing import Dict


class UCBAgent:
    # UCB (Upper Confidence Bound) tabanlı ajan 
    # Çok kollu bandit problemleri için uygundur    
    def __init__(self, n_actions: int, c: float = 2.0):
        self.n_actions = n_actions # Action (model) sayısı
        self.c = c # Keşif katsayısı
        self.counts = np.zeros(n_actions) # Her actionın seçilme sayısı
        self.total_rewards = np.zeros(n_actions) # Her actionın toplam ödülü
        self.mean_rewards = np.zeros(n_actions) # Her actionın ortalama ödülü
        self.t = 0 # Toplam adım sayısı
    
    def select_action(self):
        # İlk turda her action'ı en az bir kez dene
        if np.min(self.counts) == 0:
            return np.argmin(self.counts)
        
        # UCB hesapla
        # faydalanma terimi = ortalama ödülü = Exploition (Faydalanma)
        # keşif terimi = keşif katsayısı * sqrt(log(toplam adım sayısı) / her actionın seçilme sayısı) = Exploration (deneme)
        # UCB = faydalanma terimi + keşif terimi
        # En yüksek UCB değerine sahip action'ı seç

        ucb_values = self.mean_rewards + \
            self.c * np.sqrt(
            np.log(self.t) / self.counts
        )
        
        return np.argmax(ucb_values)
    
    def update(self, action: int, reward: float):
        self.t += 1
        self.counts[action] += 1
        self.total_rewards[action] += reward
        self.mean_rewards[action] = self.total_rewards[action] / self.counts[action]

    def get_stats(self) -> Dict:
        """Agent istatistikleri"""
        return {
            'strategy': 'UCB',
            'total_steps': self.t,
            'c_parameter': self.c,
            'action_counts': self.counts.tolist(),
            'mean_rewards': self.mean_rewards.tolist(),
            'total_rewards': self.total_rewards.tolist(),
            'best_action': int(np.argmax(self.mean_rewards)) if self.t > 0 else 0,
        }



    def save(self, filepath: str):
        """Agent'ı kaydet"""
        state = {
            'n_actions': self.n_actions,
            'c': self.c,
            'counts': self.counts.tolist(),
            'total_rewards': self.total_rewards.tolist(),
            'mean_rewards': self.mean_rewards.tolist(),
            't': self.t,
            'action_history': self.action_history,
            'reward_history': self.reward_history
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✓ Agent saved: {filepath}")


    def load(self, filepath: str):
        """Agent'ı yükle"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.n_actions = state['n_actions']
        self.c = state['c']
        self.counts = np.array(state['counts'])
        self.total_rewards = np.array(state['total_rewards'])
        self.mean_rewards = np.array(state['mean_rewards'])
        self.t = state['t']
        self.action_history = state['action_history']
        self.reward_history = state['reward_history']
        print(f"✓ Agent loaded: {filepath}")

    def reset(self):
        """Agent'ı sıfırla"""
        self.counts = np.zeros(self.n_actions)
        self.total_rewards = np.zeros(self.n_actions)
        self.mean_rewards = np.zeros(self.n_actions)
        self.t = 0
        self.action_history = []
        self.reward_history = []