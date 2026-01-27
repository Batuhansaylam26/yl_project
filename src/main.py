from test_env import TestEnv
from agent import UCBAgent
from utils import convert_to_nf_dataframe, setup_logging, create_episode_summary_table,save_episode_csv, get_takens_embedding
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import ray
import logging
import argparse
#ray.init(include_dashboard=False, _metrics_export_port=None)
#os.environ['RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS'] = '0'


def main() -> None:
    # Data args
    parser = argparse.ArgumentParser(description="UCB Agent ile Test Ortamı")
    parser.add_argument('--data_path', type=str, default='data/lorenze_attractor.csv',
                        help='Data File Path (CSV)')
    
    parser.add_argument('--time_col', type=str, default='t',
                        help='Time Column Name')
    parser.add_argument('--value_col', type=str, default='Z',
                        help='Value Column Name')
    parser.add_argument('--use_exogenous_cols', type=bool, default=True,
                        help='Use Exogenous Columns or Not')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Forecast Horizon')
    parser.add_argument('--max_steps', type=int, default=5,
                        help='Max Steps per Episode')
    parser.add_argument('--model_n_trials', type=int, default=5,
                        help='Number of Trials for Model Hyperparameter Tuning')
    parser.add_argument('--val_size', type=int, default=None,
                        help='Validation Set Size')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test Set Size (Fraction)')
    parser.add_argument('--n_episodes', type=int, default=5,
                        help='Number of Episodes to Run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random Seed for Reproducibility')
    parser.add_argument('--model_max_steps', type=int, default=250,
                        help='Maximum Training Steps for Each Model')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early Stopping Patience for Model Training')
    parser.add_argument('--user_takens_embedding', type=bool, default=False,
                        help='Use Takens Embedding for Exogenous Variables')
    # UCB parameters
    parser.add_argument('--ucb-c', type=float, default=2.0,
                       help='UCB exploration parameter')
    parser.add_argument(
        '--experiment_name', type=str, default='ucb_test_env',
        help='Experiment name for logging'
    )
    args = parser.parse_args()





    if not ray.is_initialized():
        ray.init(include_dashboard=False, _metrics_export_port=None)
    os.environ['RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS'] = '0'
    # Örnek veri yükleme
    data = pd.read_csv(args.data_path)

    

    
    # Veri çerçevesini NeuralForecast formatına dönüştürme
    data = convert_to_nf_dataframe(data, 
                                      time_col=args.time_col, 
                                      value_col=args.value_col,
                                      use_exogenous_cols=args.use_exogenous_cols,
                                      id_col=None
    )
    val_size = None
    input_size = None
    if args.user_takens_embedding:
        data, input_size, val_size = get_takens_embedding(
            data, 
            max_tau=1000
        )
        
    y = data['y']
    X = data.drop(
        'y',
        inplace = False,
        axis =1 
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    if  not val_size:
        if args.val_size is None:
            val_size = len(train_df) // 10 * 2
        else:
            val_size = args.val_size

    
    # Ortamı oluşturma
    env = TestEnv(
        train_data=train_df,
        val_size=val_size,
        test_data=test_df,
        input_size= input_size,
        horizon=args.horizon,
        max_steps=args.max_steps,
        use_exog=args.use_exogenous_cols,
        model_n_trials=args.model_n_trials,
        experiment_name=args.experiment_name,
        model_max_steps=args.model_max_steps,
        model_patience=args.patience,

    )
    models = [
        'TimesNet',
        'GRU',
        'LSTM',
        'KAN',
        'VanillaTransformer'
    ]
    n_actions = len(models)

    step_count = 0

    # Ajanı oluşturma
    agent = UCBAgent(n_actions=n_actions, c=args.ucb_c)
    for episode in range(args.n_episodes):
        print(f"\n=== Episode {episode+1}/{args.n_episodes} ===")


        # Ortamda etkileşimde bulunma
        observation, info = env.reset()
        done = False
        
        while not done:
            action = agent.select_action()
            observation, reward, terminated, truncated, info = env.step(action, episode=episode)
            agent.update(action, reward)
            print(f"Step: {step_count} | Action: {action} ({env.models[action]}) | Reward: {reward:.6f}")
            step_count += 1
            
            if terminated:
                done = True
        env.render()

    # Agent stats
    stats = agent.get_stats()
    print(f"\n{'='*70}")
    print(f"STATISTICS")
    print(f"{'='*70}")
    print(f"Total Steps: {stats['total_steps']}")
    print(f"Best Action: {stats['best_action']} ({env.models[stats['best_action']]})")
    print(f"Action Counts: {stats['action_counts']}")
    print(f"Mean Rewards: {[f'{r:.6f}' for r in stats['mean_rewards']]}")
    print(f"Total Rewards: {[f'{r:.6f}' for r in stats['total_rewards']]}")
    print(f"{'='*70}\n")
    
    # Save agent
    agent.save(f"{env.output_dir}/agent.json")


    env.close()
    
    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    main()