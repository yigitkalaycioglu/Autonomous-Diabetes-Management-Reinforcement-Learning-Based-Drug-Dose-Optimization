import torch
import numpy as np
import pandas as pd  # EKLENDİ
import matplotlib.pyplot as plt
from diabetes_env import DiabetesEnv


# --- Policy Network Mimarisi (Train dosyasındaki ile AYNI olmalı) ---
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        probs = self.softmax(self.fc3(x))
        return probs


def test_model():
    print("--- Test Süreci Başlıyor ---")

    # 1. Veri Setini Yükle
    try:
        df = pd.read_csv('diabetes_prediction_dataset.csv')
    except FileNotFoundError:
        print("HATA: 'diabetes_prediction_dataset.csv' dosyası bulunamadı!")
        return

    # 2. Ortamı Başlat
    env = DiabetesEnv(df)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3. Modeli Yükle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNetwork(state_dim, action_dim).to(device)

    try:
        model.load_state_dict(torch.load('reinforce_diabetes_model.pth', map_location=device))
        print("Model başarıyla yüklendi: reinforce_diabetes_model.pth")
    except FileNotFoundError:
        print("HATA: Model dosyası bulunamadı! Önce train_reinforce.py çalıştırın.")
        return

    model.eval()  # Test modu (Dropout vb. kapatır)

    # 4. Test Epizodu Başlat
    state, _ = env.reset()
    glucose_history = [env.current_glucose]
    actions_history = []
    rewards = 0

    print(f"\nBaşlangıç Glukoz: {env.current_glucose:.2f} mg/dL")
    print(f"Hedef Glukoz: {env.target_glucose} mg/dL")

    for t in range(50):  # 50 Adımlık test
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = model(state_tensor)
            action = torch.argmax(probs).item()  # En yüksek olasılıklı eylemi seç (Greedy)

        next_state, reward, done, truncated, _ = env.step(action)

        glucose_history.append(env.current_glucose)
        actions_history.append(action)
        rewards += reward
        state = next_state

        # Eylem metni
        act_str = ["AZALT", "KORU", "ARTIR"][action]
        print(f"Adım {t + 1}: Glukoz={env.current_glucose:.1f} -> Eylem={act_str} -> Ödül={reward:.1f}")

        if done or truncated:
            break

    print(f"\nTest Bitti. Toplam Ödül: {rewards:.2f}")

    # 5. Görselleştirme
    plt.figure(figsize=(12, 6))

    # Glukoz Seviyesi
    plt.plot(glucose_history, label='Glukoz Seviyesi', marker='o')

    # Hedef Aralığı (Yeşil Bölge)
    plt.axhline(y=140, color='g', linestyle='--', alpha=0.5, label='Hedef Üst Sınır (140)')
    plt.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Hedef Alt Sınır (80)')
    plt.axhline(y=110, color='black', linestyle='-', alpha=0.3, label='Tam Hedef (110)')

    # Kritik Sınırlar (Kırmızı)
    plt.axhline(y=70, color='r', linestyle=':', label='Hipoglisemi Sınırı (70)')

    plt.title(f'Test Sonucu - Toplam Ödül: {rewards:.1f}')
    plt.xlabel('Adım')
    plt.ylabel('Glukoz (mg/dL)')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_result.png')
    print("Test grafiği kaydedildi: test_result.png")
    plt.show()


if __name__ == '__main__':
    test_model()