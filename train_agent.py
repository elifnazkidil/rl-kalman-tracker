from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_env.autonomous_env import AutonomousEnv
import os

def train():
    print("RL Ortamı başlatılıyor...")
    env = AutonomousEnv()
    
    # Ortamın standartlara uygunluğunu kontrol et
    check_env(env)
    print("Ortam testi başarılı!")
    
    print("PPO Modeli oluşturuluyor...")
    # Multi-Layer Perceptron (MlpPolicy) tabanlı PPO Ajanı
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    print("Eğitim başlıyor... (Bu işlem kısa sürecek bir deneme sürümüdür)")
    # Gerçek bir projede total_timesteps 1,000,000 civarı olur.
    # Portfolyo ve iyi bir öğrenme için 200,000 adıma çıkarıldı.
    model.learn(total_timesteps=200000)
    
    model_path = "ppo_autonomous_agent"
    model.save(model_path)
    print(f"Model başarıyla kaydedildi: {model_path}.zip")

if __name__ == "__main__":
    train()
