import numpy as np
from tracker.kalman_filter import KalmanFilter

def test_kalman_filter():
    print("Kalman Filter İzole Testi Başlatılıyor...")
    kf = KalmanFilter(dt=1.0)
    
    # Doğrusal hareket eden bir nesne (Her saniye x ve y'de 1 birim ilerliyor)
    # Başlangıç: x=0, y=0, dx=1, dy=1
    true_positions = [(i, i) for i in range(1, 10)]
    
    for i, (true_x, true_y) in enumerate(true_positions):
        # Ölçüme biraz gürültü ekleyelim
        noise = np.random.normal(0, 0.5, 2)
        measured_x = true_x + noise[0]
        measured_y = true_y + noise[1]
        
        # Kalman filter tahmini
        pred = kf.predict()
        
        # Güncelleme
        updated = kf.update([measured_x, measured_y])
        
        print(f"Adım {i+1}:")
        print(f"  Gerçek Konum: ({true_x:.2f}, {true_y:.2f})")
        print(f"  Gürültülü Ölçüm: ({measured_x:.2f}, {measured_y:.2f})")
        print(f"  Kalman Tahmini (Güncellenmiş): ({updated[0][0]:.2f}, {updated[1][0]:.2f})")
        print("-" * 30)

if __name__ == "__main__":
    test_kalman_filter()
