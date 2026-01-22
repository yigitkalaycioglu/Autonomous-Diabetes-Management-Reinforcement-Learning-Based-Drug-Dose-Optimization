import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
import os


class DiabetesEnv(gym.Env):
    """
    OpenAI Gym uyumlu Diyabet Simülasyon Ortamı.
    Durum (State): [normalize_glukoz, normalize_hba1c, ...]
    Eylem (Action): 0 (Azalt), 1 (Koru), 2 (Artır)
    Dinamik: 'glucose_dynamics_model.pkl' (Random Forest) üzerinden tahmin edilir.
    """

    def __init__(self, df):
        super(DiabetesEnv, self).__init__()

        self.df = df

        # --- Model Yükleme ---
        model_path = 'glucose_dynamics_model.pkl'
        if os.path.exists(model_path):
            self.dynamics_model = joblib.load(model_path)
            print("Environment: Dynamics Model başarıyla yüklendi.")
        else:
            raise FileNotFoundError(
                "HATA: 'glucose_dynamics_model.pkl' bulunamadı! Önce train_dynamics_model.py çalıştırılmalı.")

        # Sabitler
        self.target_glucose = 110
        self.safe_min = 70
        self.safe_max = 180
        self.abs_min = 40
        self.abs_max = 400
        self.max_steps = 50  # Her epizot max adım sayısı

        # Aksiyon Uzayı: 0=Azalt, 1=Koru, 2=Artır
        self.action_space = spaces.Discrete(3)

        # Gözlem Uzayı: Normalize edilmiş değerler (Min-Max varsayımı ile 0-1 arası)
        # Glukoz, HbA1c, Yaş, BMI, Hipertansiyon, Kalp Hastalığı
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        self.state = None
        self.current_glucose = None
        self.step_count = 0
        self.patient_data = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Rastgele bir hasta seç
        self.patient_data = self.df.sample(1).iloc[0]

        # Başlangıç glukozunu veriden al
        self.current_glucose = float(self.patient_data['blood_glucose_level'])

        # Başlangıç durumunu biraz randomize et (Ezberi bozmak için)
        self.current_glucose += np.random.uniform(-10, 10)
        self.current_glucose = np.clip(self.current_glucose, self.abs_min, self.abs_max)

        self.step_count = 0
        self.state = self._get_observation()

        return self.state, {}

    def step(self, action):
        # --- 1. Model Tabanlı Geçiş (Dynamics Model) ---
        # Eğitilmiş model tahmin ediyor.

        # Model girdisi hazırla
        input_df = pd.DataFrame({
            'current_glucose': [self.current_glucose],
            'action': [action]
        })

        # Modelden değişim miktarını tahmin et (Predict)
        predicted_delta = self.dynamics_model.predict(input_df)[0]

        # Stokastik yapı (Hafif belirsizlik ekle)
        noise = np.random.normal(0, 1.5)

        # Yeni glukoz değeri
        self.current_glucose += (predicted_delta + noise)
        self.current_glucose = np.clip(self.current_glucose, self.abs_min, self.abs_max)

        self.step_count += 1
        self.state = self._get_observation()

        # --- 2. Ödül Fonksiyonu (Reward Engineering) ---
        reward = 0
        dist = abs(self.current_glucose - self.target_glucose)

        # A. Bölgesel Ödüller
        if dist < 15:  # 95-125 arası (Mükemmel)
            reward = 5.0
        elif dist < 30:  # 80-140 arası (İyi)
            reward = 1.0
        else:
            # Hedef dışı (Uzaklaştıkça artan ceza)
            reward = -0.05 * dist

        # B. Stabilite Bonusu (Gereksiz İlaç Kullanımını Önleme)
        # Eğer yeşil alandaysak (dist < 30) ve eylem "Koru (1)" ise ödül ver
        if action == 1:
            if dist < 30:
                reward += 2.0
            else:
                # Yeşil alan dışında "beklemek" kötü olabilir
                reward -= 1.0

        # C. Kritik Güvenlik Cezaları
        terminated = False

        # Hipoglisemi Riski (Düşük Şeker - Çok Tehlikeli)
        if self.current_glucose < self.safe_min:  # < 70
            reward -= 10.0
            if self.current_glucose < self.abs_min + 5:  # < 45 (Koma)
                reward -= 50.0
                terminated = True  # Hasta komaya girdi, oyun biter

        # Hiperglisemi Riski (Yüksek Şeker)
        if self.current_glucose > 300:
            reward -= 5.0
            if self.current_glucose > self.abs_max - 10:  # > 390
                reward -= 20.0
                terminated = True

        # D. Zaman Kısıtı
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True

        return self.state, reward, terminated, truncated, {}

    def _get_observation(self):
        # Durum vektörünü oluştur ve 0-1 arasına normalize et
        # ['blood_glucose_level', 'HbA1c_level', 'age', 'bmi', 'hypertension', 'heart_disease']

        # Basit Min-Max Normalizasyon (Yaklaşık değerlerle)
        norm_glucose = self.current_glucose / 400.0
        norm_hba1c = self.patient_data['HbA1c_level'] / 9.0
        norm_age = self.patient_data['age'] / 100.0
        norm_bmi = self.patient_data['bmi'] / 50.0  # Ortalama max BMI varsayımı

        obs = np.array([
            norm_glucose,
            norm_hba1c,
            norm_age,
            norm_bmi,
            self.patient_data['hypertension'],
            self.patient_data['heart_disease']
        ], dtype=np.float32)

        return obs