from test_env import TestEnv
from agent import UCBAgent
from utils import convert_to_nf_dataframe
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import ray
ray.init(include_dashboard=False, _metrics_export_port=None)
os.environ['RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS'] = '0'


if __name__ == "__main__":
    # Örnek veri yükleme
    data = pd.read_csv('data/lorenze_attractor.csv')
    
    # Veri çerçevesini NeuralForecast formatına dönüştürme
    nf_data = convert_to_nf_dataframe(data, 
                                      time_col='t', 
                                      value_col='Z', 
                                      exogenous_cols=['X', 'Y', 'U'],
                                      id_col=None
    )
    train_df, test_df = train_test_split(nf_data, test_size=0.2, shuffle=False)
    val_size = len(train_df) // 10 * 2
    # Ortamı oluşturma
    env = TestEnv(
        train_data=train_df,
        val_size=1000, #val_size,
        test_data=test_df,
        horizon=20,
        max_steps=10,
        exog_vars=['X', 'Y', 'U'],
        model_n_trials=5

    )
    models = [
        'TimesNet',
        'GRU',
        'LSTM',
        'KAN',
        'VanillaTransformer'
    ]
    # Ajanı oluşturma
    agent = UCBAgent(n_actions=3, c=2.0)
    
    # Ortamda etkileşimde bulunma
    observation, info = env.reset()
    done = False
    
    while not done:
        action = agent.select_action()
        observation, reward, terminated, truncated, info = env.step(action)
        agent.update(action, reward)
        
        if terminated:
            break


    env.render()

    env.close()
    
    print("Eğitim tamamlandı.")