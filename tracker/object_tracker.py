import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker.kalman_filter import KalmanFilter

class TrackedObject:
    def __init__(self, track_id, initial_pos):
        self.track_id = track_id
        # Hızlı hareket eden nesneler için process noise'u yüksek tutuyoruz
        self.kf = KalmanFilter(dt=0.1, std_acc=5.0, std_meas=0.5)
        self.kf.x[:2] = np.array(initial_pos).reshape(2, 1)
        self.time_since_update = 0
        self.hits = 1
        self.history = []
        
    def predict(self):
        pred = self.kf.predict()
        self.time_since_update += 1
        self.history.append((pred[0][0], pred[1][0]))
        return pred
        
    def update(self, detection):
        self.kf.update(detection)
        self.time_since_update = 0
        self.hits += 1
        
    def get_pos(self):
        return self.kf.x[0,0], self.kf.x[1,0]

class ObjectTracker:
    """
    Basit bir Multi-Object Tracker (SORT algoritmasının temeli).
    Tespit edilen noktaları mevcut izlenen nesnelerle (Tracks) eşleştirir.
    """
    def __init__(self, max_age=5, min_hits=3, dist_threshold=50):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections):
        """
        Detections: [(x1, y1), (x2, y2), ...]
        """
        # 1. Mevcut tüm track'ler için tahminde bulun
        for track in self.tracks:
            track.predict()
            
        if len(detections) == 0:
            # Hiç tespit yoksa yaşlanan track'leri sil
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]
            
        # 2. Maliyet (Mesafe) matrisi oluştur
        if len(self.tracks) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    tx, ty = track.get_pos()
                    dx, dy = det
                    cost_matrix[t_idx, d_idx] = np.sqrt((tx - dx)**2 + (ty - dy)**2)
                    
            # Hungarian Algoritması ile eşleştir (Scipy linear_sum_assignment)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            unmatched_tracks = set(range(len(self.tracks)))
            unmatched_detections = set(range(len(detections)))
            
            matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] > self.dist_threshold:
                    pass # Eşleşmeyi reddet
                else:
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_detections.discard(c)
        else:
            matches = []
            unmatched_tracks = set()
            unmatched_detections = set(range(len(detections)))
            
        # 3. Eşleşenleri güncelle
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])
            
        # 4. Eşleşmeyen tespitlerden yeni track oluştur
        for d_idx in unmatched_detections:
            self.tracks.append(TrackedObject(self.next_id, detections[d_idx]))
            self.next_id += 1
            
        # 5. Ölü track'leri temizle
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Aktif olanları döndür (yeni oluşanlar hemen min_hits'i karşılamayabilir)
        return [t for t in self.tracks if t.hits >= self.min_hits or t.time_since_update == 0]
