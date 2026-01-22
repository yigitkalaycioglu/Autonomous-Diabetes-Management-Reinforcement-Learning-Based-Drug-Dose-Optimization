import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_and_save_model():
    print("Veri seti yükleniyor ve sentetik geçişler üretiliyor...")

    # 1. Veri Setini Yükle
    try:
        df = pd.read_csv('diabetes_prediction_dataset.csv')
    except FileNotFoundError:
        print("HATA: 'diabetes_prediction_dataset.csv' dosyası bulunamadı!")
        return

    # Sadece Diyabet olanları al (Simülasyon mantığı için)
    df = df[df['diabetes'] == 1].sample(n=2000, random_state=42).reset_index(drop=True)

    # 2. Eğitim Verisi Oluştur (RL Mantığını Modele Öğretme)
    # Model: (Şu anki Şeker, Eylem) -> (Şeker Değişimi)

    current_glucose_list = []
    actions_list = []
    delta_glucose_list = []

    # Veri setini çoğaltarak farklı senaryolar üret
    for _ in range(5):
        for val in df['blood_glucose_level'].values:
            # Gürültülü başlangıç değeri
            current_g = float(val) + np.random.normal(0, 5)

            for action in [0, 1, 2]:
                # --- Biyolojik Mantık (Model bunu öğrenecek) ---
                natural_drift = np.random.uniform(2, 5)  # Yemek etkisi

                if action == 0:  # Doz Azalt
                    effect = 0
                elif action == 1:  # Doz Koru
                    effect = -2.0
                elif action == 2:  # Doz Artır
                    effect = -12.0

                # Gürültü ekle ki model ezberlemesin, genellesin
                noise = np.random.normal(0, 2.0)

                # Gerçek değişim
                delta = natural_drift + effect + noise

                current_glucose_list.append(current_g)
                actions_list.append(action)
                delta_glucose_list.append(delta)

    X = pd.DataFrame({
        'current_glucose': current_glucose_list,
        'action': actions_list
    })
    y = np.array(delta_glucose_list)

    # 3. Modeli Eğit (Random Forest)
    print("Dynamics Model (Random Forest) eğitiliyor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    # 4. Kaydet
    joblib.dump(model, 'glucose_dynamics_model.pkl')
    print("BAŞARILI: 'glucose_dynamics_model.pkl' dosyası oluşturuldu.")

    # Test
    test_val = model.predict(pd.DataFrame({'current_glucose': [150], 'action': [2]}))[0]
    print(f"Test Tahmini (150 mg/dL + Doz Artır): Değişim {test_val:.2f}")


if __name__ == "__main__":
    train_and_save_model()