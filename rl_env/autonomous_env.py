import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AutonomousEnv(gym.Env):
    """
    Otonom Navigasyon için Custom Reinforcement Learning Ortamı.
    Ajan (Drone/Araç), hareketli engellerden kaçarak hedefe ulaşmaya çalışır.
    """
    def __init__(self, grid_size=400, max_obstacles=5):
        super(AutonomousEnv, self).__init__()
        self.grid_size = grid_size
        self.max_obstacles = max_obstacles
        self.agent_step = 10.0 # Ajanın hızı
        
        # Action Space: 0: Dur, 1: Yukarı, 2: Aşağı, 3: Sola, 4: Sağa
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: [agent_x, agent_y, target_x, target_y] + [obs1_x, obs1_y, ...]
        obs_dim = 4 + (self.max_obstacles * 2)
        # Normalize edilmiş kordinatlar (0 ile 1 arası)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self.agent_pos = np.array([50.0, 50.0])
        self.target_pos = np.array([350.0, 350.0])
        self.obstacles = np.zeros((self.max_obstacles, 2))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([50.0, 50.0])
        # Rastgele hedef
        self.target_pos = np.random.uniform(100, 380, size=(2,))
        
        # Başlangıç engelleri rastgele (Dışarıdan da beslenebilir)
        self.obstacles = np.random.uniform(50, 350, size=(self.max_obstacles, 2))
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Durumu normalize et (0-1)
        obs = [
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.target_pos[0] / self.grid_size,
            self.target_pos[1] / self.grid_size,
        ]
        for obs_pos in self.obstacles:
            obs.extend([obs_pos[0] / self.grid_size, obs_pos[1] / self.grid_size])
        return np.array(obs, dtype=np.float32)
        
    def step(self, action):
        reward = -0.1 # Zaman cezası (hızlı gitmesi için)
        
        # Ajanı hareket ettir
        if action == 0: # Dur
            reward -= 0.4 # Durma cezası (toplam -0.5)
        elif action == 1: # Yukarı
            self.agent_pos[1] -= self.agent_step
        elif action == 2: # Aşağı
            self.agent_pos[1] += self.agent_step
        elif action == 3: # Sol
            self.agent_pos[0] -= self.agent_step
        elif action == 4: # Sağ
            self.agent_pos[0] += self.agent_step
            
        # Sınırları kontrol et
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
        
        reward = -0.1 # Zaman cezası (hızlı gitmesi için)
        terminated = False
        truncated = False
        
        # Hedefe varış kontrolü
        dist_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        if dist_to_target < 20.0:
            reward = 100.0
            terminated = True
            
        # Çarpışma kontrolü
        for obs_pos in self.obstacles:
            dist_to_obs = np.linalg.norm(self.agent_pos - obs_pos)
            if dist_to_obs < 15.0: # Çarpışma yarıçapı
                reward = -50.0
                terminated = True
                break
                
        return self._get_obs(), reward, terminated, truncated, {}
        
    def update_obstacles(self, tracked_obstacles):
        """
        Ana simülasyondan gelen (Kalman ile takip edilen) engel konumlarını günceller.
        tracked_obstacles: [(x1, y1), (x2, y2)...] max_obstacles kadar.
        """
        self.obstacles = np.zeros((self.max_obstacles, 2))
        for i, pos in enumerate(tracked_obstacles[:self.max_obstacles]):
            self.obstacles[i] = [pos[0], pos[1]]
