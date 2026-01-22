import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import os

# Kendi modüllerimiz
from diabetes_env import DiabetesEnv


# --- 1. REINFORCE Ajanı (Policy Network) ---
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


class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Eğitim hafızası
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = deque()

        # Geriye dönük kümülatif ödül (Discounted Return)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)

        returns = torch.tensor(returns).to(self.device)

        # Normalizasyon (Stabilite için kritik)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()

        if len(policy_loss) > 0:
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            self.optimizer.step()

        self.saved_log_probs = []
        self.rewards = []


# --- 2. Eğitim Döngüsü ---
def train():
    # Veri setini yükle
    try:
        df = pd.read_csv('diabetes_prediction_dataset.csv')
    except FileNotFoundError:
        print("HATA: 'diabetes_prediction_dataset.csv' dosyası bulunamadı!")
        return

    # Ortamı başlat
    env = DiabetesEnv(df)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- AYARLAR ---
    # Episod sayısını artırdık, Learning Rate'i düşürdük
    num_episodes = 10000
    learning_rate = 0.001

    agent = REINFORCEAgent(state_dim, action_dim, learning_rate=learning_rate, gamma=0.99)

    all_rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')  # En iyi modeli takip etmek için

    print(f"Eğitim Başlıyor... (Episod Hedefi: {num_episodes})")

    for i_episode in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0

        for t in range(100):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_reward(reward)
            state = next_state
            ep_reward += reward

            if done or truncated:
                break

        agent.update_policy()

        all_rewards.append(ep_reward)

        # Son 100 epizodun ortalaması
        avg_reward = np.mean(all_rewards[-100:])
        avg_rewards.append(avg_reward)

        # Loglama
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}\tLast Reward: {ep_reward:.2f}\tAverage Reward: {avg_reward:.2f}')

            # --- YENİ ÖZELLİK: En İyi Modeli Kaydet ---
            # Eğer şu anki ortalama performans, rekorumuzdan iyiyse kaydet
            if avg_reward > best_avg_reward and i_episode > 500:
                best_avg_reward = avg_reward
                torch.save(agent.policy_net.state_dict(), 'reinforce_diabetes_model_best.pth')
                # Ana model ismiyle de kaydet ki test kodu bulabilsin
                torch.save(agent.policy_net.state_dict(), 'reinforce_diabetes_model.pth')
                # print(f"  -> Yeni Rekor! Model güncellendi. (Avg: {best_avg_reward:.2f})")

    print("Eğitim Tamamlandı!")

    # Son durumu da kaydet
    torch.save(agent.policy_net.state_dict(), 'reinforce_diabetes_model_final.pth')
    print(f"En yüksek ortalama ödül: {best_avg_reward:.2f}")

    # Grafiği Çiz
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards, label='Epizot Ödülü', alpha=0.2, color='gray')
    plt.plot(avg_rewards, label='Ortalama Ödül (Moving Avg)', color='blue', linewidth=2)
    plt.xlabel('Epizot')
    plt.ylabel('Ödül')
    plt.title(f'REINFORCE Eğitim Performansı ({num_episodes} Episod)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_plot.png')
    print("Eğitim grafiği kaydedildi: training_plot.png")
    plt.show()


if __name__ == '__main__':
    train()