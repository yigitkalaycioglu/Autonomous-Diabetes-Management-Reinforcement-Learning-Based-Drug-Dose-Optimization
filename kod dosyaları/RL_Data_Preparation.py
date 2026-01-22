import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


def prepare_data():
    print("--- Veri Hazırlama Başlıyor ---")

    # 1. Veri Setini Yükle
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    print(f"Orijinal Veri Boyutu: {df.shape}")

    # 2. Veri Temizliği
    # Tekrar eden satırları kaldır
    df = df.drop_duplicates()

    # Cinsiyette 'Other' olanları (çok az sayıda ise) temizleyebiliriz veya tutabiliriz.
    # RL için netlik iyidir, 'Other' zaten çok az olduğu için siliyoruz.
    df = df[df['gender'] != 'Other']

    # 3. Encoding (Kategorik -> Sayısal)
    # Cinsiyet: Male=1, Female=0
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])

    # Sigara Geçmişi: never, current vs. -> 0, 1, 2...
    le_smoking = LabelEncoder()
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

    print("Kategorik veriler sayısal hale getirildi.")

    # 4. Scaling (Ölçeklendirme)
    # DİKKAT: 'blood_glucose_level' burada ölçeklendirilmez!
    # Çünkü Environment, başlangıç şekeri olarak gerçek mg/dL değerine ihtiyaç duyar.
    # Ajanın State'ine girerken Env içinde normalize edeceğiz.

    cols_to_scale = ['age', 'bmi', 'HbA1c_level']
    scaler = StandardScaler()

    # Fit ve Transform işlemi
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    print("Age, BMI ve HbA1c standartlaştırıldı (Mean=0, Std=1).")

    # 5. Kontrol
    print("\n--- İşlenmiş Veri Önizleme ---")
    print(df.head())
    print("-" * 30)
    print(df.describe())

    # 6. Kaydetme
    output_filename = 'diabetes_rl_data.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nDosya başarıyla kaydedildi: {output_filename}")

    # Scaler nesnesini de kaydet (İleride tersine çevirmek gerekirse diye)
    with open("scaler_rl.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler nesnesi kaydedildi: scaler_rl.pkl")


if __name__ == "__main__":
    prepare_data()