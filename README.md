# RL-Kalman Tracker Projesi

Bu proje, bir pekiştirmeli öğrenme (Reinforcement Learning - PPO) ajanı ile Kalman Filtresi (Kalman Filter) tabanlı çoklu nesne takibini (Multi-Object Tracking) birleştiren otonom bir simülasyon içerir.

## Mimari
- **`tracker/kalman_filter.py`**: Doğrusal hareket modeli için konum ve hız tahmini yapan temel Kalman Filtresi.
- **`tracker/object_tracker.py`**: Tespit edilen hedefleri (sahte gürültülü sensör verisi) mevcut hedeflerle Hungarian algoritmasıyla eşleştirip takip eden (SORT algoritması benzeri) yapı.
- **`rl_env/autonomous_env.py`**: Gymnasium tabanlı RL ortamı. Ajan hareketli hedeflerden kaçınırken ulaşması gereken noktaya gitmeye çalışır.
- **`train_agent.py`**: Stable Baselines3 kütüphanesi kullanılarak ajanın eğitildiği betik.
- **`main_sim.py`**: Sistemi birleştiren Pygame simülasyonu.

## Ajanın Eğitimi
PPO algoritmasının tatmin edici bir şekilde öğrenebilmesi (convergence) için `train_agent.py` içindeki `total_timesteps` parametresi **200,000** olarak ayarlanmıştır. Portfolyo veya demo amaçlı daha kısa denemeler yapmak isterseniz bu rakamı düşürebilirsiniz ancak ajanın karar mekanizmasının zayıflayacağını unutmayın.
Eğitim sonucunda `ppo_autonomous_agent.zip` adında bir model dosyası oluşturulacaktır.

## Çalıştırma Adımları
1. Ortamı kurmak için: `pip install -r requirements.txt`
2. Modeli eğitmek için: `python train_agent.py` (200.000 adım için biraz beklemeniz gerekecektir).
3. Simülasyonu izlemek için: `python main_sim.py`
