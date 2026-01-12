import numpy as np

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
