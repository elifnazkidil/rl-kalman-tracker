import numpy as np

class KalmanFilter:
    """
    Basit 2D (x, y) Sabit Hız Modeli Kalman Filtresi.
    Durum Vektörü (State): [x, y, dx, dy] (Konum ve Hız)
    Ölçüm Vektörü (Measurement): [x, y] (Sadece konumu ölçebildiğimizi varsayıyoruz)
    """
    def __init__(self, dt=1.0, u_x=0, u_y=0, std_acc=1, std_meas=0.1):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        
        # Durum Vektörü Başlangıcı (x, y, dx, dy)
        self.x = np.array([[0.], [0.], [0.], [0.]])
        
        # Durum Geçiş Matrisi (State Transition Matrix) - F
        # x_new = x + dt*dx
        # y_new = y + dt*dy
        self.F = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        # Kontrol Giriş Matrisi (Control Input Matrix) - B
        self.B = np.array([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])
        
        # Ölçüm Matrisi (Measurement Matrix) - H
        # Sadece x ve y'yi ölçüyoruz.
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        # Süreç Gürültüsü Kovaryans Matrisi (Process Noise Covariance) - Q
        q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                       [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                       [(self.dt**3)/2, 0, self.dt**2, 0],
                       [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        self.Q = q
        
        # Ölçüm Gürültüsü Kovaryans Matrisi (Measurement Noise Covariance) - R
        self.R = np.array([[std_meas**2, 0],
                            [0, std_meas**2]])
        
        # Tahmin Hatası Kovaryans Matrisi (Estimate Error Covariance) - P
        self.P = np.eye(self.F.shape[1])

    def predict(self):
        """
        Durum tahmin (Prediction) adımı.
        Bir sonraki adımı tahmin eder.
        """
        # Durum tahmini: x_k = F * x_{k-1} + B * u
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        
        # Hata kovaryans tahmini: P_k = F * P_{k-1} * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x.tolist()

    def update(self, z):
        """
        Ölçüm güncelleme (Update/Correction) adımı.
        Gerçek ölçüm değeri z = [x, y] geldiğinde tahmini düzeltir.
        """
        z = np.array(z).reshape(2, 1) # 2x1 vektör
        
        # Kalman Kazancı (Kalman Gain) - K
        # K = P * H^T * (H * P * H^T + R)^-1
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Durum güncelleme
        # x_k = x_k + K * (z - H * x_k)
        y = z - np.dot(self.H, self.x) # Innovation (Ölçüm kalıntısı)
        self.x = self.x + np.dot(K, y)
        
        # Hata kovaryans güncelleme
        # P_k = (I - K * H) * P_k
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x.tolist()
