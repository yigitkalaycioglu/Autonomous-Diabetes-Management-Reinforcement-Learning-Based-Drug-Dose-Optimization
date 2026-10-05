# Pekiştirmeli Öğrenme ile Dinamik İnsülin Dozu Ayarı

Sakarya Üniversitesi ISE 427 Tıpta Yapay Zeka dersi (2025-2026 Güz) dönem projesi. Hiperglisemi hastalarında kan şekerini güvenli aralıkta tutmak için her adımda "dozu azalt / koru / artır" kararını veren bir REINFORCE ajanı eğittim. Gerçek hastalar üzerinde deneme yapılamayacağı için önce veriden bir simülasyon ortamı kurdum, ajanı bu ortamda eğittim.

**Canlı rapor:** https://yigitkalaycioglu.github.io/Autonomous-Diabetes-Management-Reinforcement-Learning-Based-Drug-Dose-Optimization/ — sonuçlar, grafikler ve çıktılarıyla birlikte not defterleri tarayıcıda görüntülenebilir.

## Yaklaşım

Kaggle'daki Diabetes Prediction Dataset 100.000 hastanın tek bir andaki değerlerini içeriyor (yaş, cinsiyet, BMI, HbA1c, kan şekeri, hipertansiyon, kalp hastalığı, sigara). Zaman serisi olmadığı için önce glukozun bir eyleme nasıl tepki vereceğini tahmin eden bir model eğittim, ortamı bu modelin üstüne kurdum.

1. `RL_Data_Preparation.py`: veriyi temizleyip ölçekliyor, `diabetes_rl_data.csv` ve `scaler_rl.pkl` dosyalarını üretiyor.
2. `train_dynamics_model.py.py`: mevcut glukoz ve uygulanan eylemden glukoz değişimini tahmin eden bir `RandomForestRegressor` eğitiyor (`glucose_dynamics_model.pkl`).
3. `diabetes_env.py`: Gymnasium tabanlı `DiabetesEnv`. Durum 6 boyutlu (glukoz, HbA1c, yaş, BMI, hipertansiyon, kalp hastalığı), 3 ayrık eylem var.
4. `train_reinforce.py`: PyTorch ile REINFORCE (policy gradient). Politika ağı 6 → 128 → 128 → 3 (ReLU, softmax), Adam optimizer, öğrenme oranı 0.001, gamma 0.99, 10.000 epizot.
5. `test_agent.py`: eğitilen ajanı yeni bir hasta profilinde çalıştırıp sonucu `test_result.png` olarak çiziyor.

Ödül fonksiyonu (`diabetes_env.py`), hedef glukoz 110 mg/dL ve bir epizot en fazla 50 adım:

- Hedefe uzaklık 15'ten az (95-125 mg/dL): +5
- Hedefe uzaklık 30'dan az (80-140 mg/dL): +1
- Daha uzaksa: -0,05 x uzaklık
- "Koru" seçildiğinde 80-140 mg/dL aralığındaysa +2 (gereksiz doz değişikliğini azaltmak için), aralık dışındaysa -1
- 70 mg/dL altı: -10, 45 mg/dL altında ek -50 ve epizot biter
- 300 mg/dL üstü: -5, 390 mg/dL üstünde ek -20 ve epizot biter

Her adımda dinamik modelin tahminine küçük bir gürültü ekleniyor, böylece ortam tamamen deterministik olmuyor.

## Sonuçlar

![Eğitim grafiği](kod%20dosyalar%C4%B1/training_plot.png)

İlk 2.500 epizotta ortalama ödül -448 civarında, ajan sık sık güvenlik sınırlarını aşıyor. 2.500 ile 3.000 arasında doz ile glukoz düşüşü arasındaki ilişkiyi öğreniyor ve ödül pozitife dönüyor. Sonrasında ortalama ödül 296'ya kadar çıkıp sabitleniyor.

![Test epizodu](kod%20dosyalar%C4%B1/test_result.png)

Test hastasında başlangıç glukozu 127,8 mg/dL. Ajan ilk iki adımda dozu artırıp şekeri 111,7 mg/dL'ye indiriyor, sonraki 48 adımda çoğunlukla "koru" seçerek değeri 105-114 mg/dL arasında tutuyor. Epizodun toplam ödülü 326.

Bu sonuçlar sadece bu simülasyon için geçerli. Ortam kesitsel veriden türetildiği için gerçek bir hastanın glukoz dinamiğini temsil etmiyor, klinik bir öneri olarak okunmamalı. Bir sonraki adım olarak sürekli glukoz ölçüm (CGM) verisiyle zaman serisi tabanlı bir model denemek mantıklı olur.

## Çalıştırma

```bash
pip install pandas numpy scikit-learn gymnasium torch matplotlib joblib
cd "kod dosyaları"
python RL_Data_Preparation.py
python train_dynamics_model.py.py
python train_reinforce.py
python test_agent.py
```

Betikler dosyaları bulundukları klasörden okuyor. Eğitilmiş modeller (`.pth` ve `.pkl`) repoda olduğu için sadece `python test_agent.py` ile test de çalıştırılabilir. Keşifsel veri analizi `EDA_Diabetes_Prediction.ipynb` içinde, proje raporu `rapor.docx`, sunum `sunum.pptx`.

## Veri seti

[Diabetes Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Veri seti kendi lisansına tabidir, repodaki MIT lisansı sadece kod için geçerlidir.
