import pygame
import numpy as np
import os
from stable_baselines3 import PPO
from rl_env.autonomous_env import AutonomousEnv
from tracker.object_tracker import ObjectTracker

# Pygame ayarları
WIDTH, HEIGHT = 400, 400
FPS = 30

# Renkler
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class MovingObstacle:
    """Sahte hareketli engeller (Kamera algılamasını simüle eder)"""
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def move(self):
        self.x += self.vx
        self.y += self.vy
        # Sınırda sekme
        if self.x <= 0 or self.x >= WIDTH: self.vx *= -1
        if self.y <= 0 or self.y >= HEIGHT: self.vy *= -1
        
    def get_noisy_detection(self):
        # Gerçek sensör gürültüsünü simüle et
        noise_x = np.random.normal(0, 2.0)
        noise_y = np.random.normal(0, 2.0)
        return (self.x + noise_x, self.y + noise_y)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL-Kalman Otonom İzleme Simülasyonu")
    clock = pygame.time.Clock()

    tracker = ObjectTracker(max_age=10, min_hits=2, dist_threshold=50)
    env = AutonomousEnv(grid_size=WIDTH, max_obstacles=5)
    obs, _ = env.reset()

    # RL Modelini Yükle (Eğer eğitilmişse)
    model_path = "ppo_autonomous_agent.zip"
    if os.path.exists(model_path):
        print("Eğitilmiş model bulundu, yükleniyor...")
        model = PPO.load("ppo_autonomous_agent")
    else:
        print("Model bulunamadı, ajan rastgele hareket edecek. Önce train_agent.py çalıştırın!")
        model = None

    # Performans optimizasyonu: Fontu döngü dışında bir kez yaratıyoruz
    font = pygame.font.SysFont(None, 20)

    # Sahte engeller yarat (Bunu YOLOv11 tespitleri gibi düşünün)
    mock_obstacles = [
        MovingObstacle(100, 100, 2, 1.5),
        MovingObstacle(300, 200, -1.5, 2),
        MovingObstacle(200, 300, 1, -1)
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 1. Sahte tespitleri al (Sensör / YOLO Simülasyonu)
        detections = []
        for mo in mock_obstacles:
            mo.move()
            detections.append(mo.get_noisy_detection())

        # 2. Kalman Filtresi ve Tracker ile Güncelle
        active_tracks = tracker.update(detections)
        
        # 3. İzlenen konumları RL ortamına besle
        tracked_positions = [track.get_pos() for track in active_tracks]
        env.update_obstacles(tracked_positions)

        # 4. RL Ajanı Karar Alsın
        if model:
            action, _states = model.predict(obs, deterministic=True)
            action = action.item()
        else:
            action = env.action_space.sample() # Rastgele hareket
            
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()

        # --- ÇİZİM ---
        screen.fill(WHITE)
        
        # Hedefi çiz
        pygame.draw.circle(screen, GREEN, (int(env.target_pos[0]), int(env.target_pos[1])), 15)
        
        # Sensör tespitlerini çiz (Noktalar halinde, gürültülü)
        for det in detections:
            pygame.draw.circle(screen, BLACK, (int(det[0]), int(det[1])), 2)
            
        # İzlenen Nesneleri (Kalman Tahminleri) Çiz
        for track in active_tracks:
            tx, ty = track.get_pos()
            pygame.draw.circle(screen, RED, (int(tx), int(ty)), 12)
            # ID numarası yaz
            img = font.render(str(track.track_id), True, BLACK)
            screen.blit(img, (int(tx)-5, int(ty)-5))
            
        # RL Ajanını Çiz
        pygame.draw.rect(screen, BLUE, (int(env.agent_pos[0])-10, int(env.agent_pos[1])-10, 20, 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
