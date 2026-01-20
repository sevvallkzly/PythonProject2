import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import os

# ======================================================
# 1. VERİ YÜKLEME
# ======================================================
base_path = os.path.dirname(os.path.abspath(__file__))
file_past = os.path.join(base_path, 'silver_prices_10years.csv')
file_forecast = os.path.join(base_path, 'silver_price_forecast_jan_mar_2026.csv')

if not os.path.exists(file_past) or not os.path.exists(file_forecast):
    raise FileNotFoundError("CSV dosyaları kodla aynı klasörde olmalı!")

df_past = pd.read_csv(file_past)
df_forecast = pd.read_csv(file_forecast)

df_past['Date'] = pd.to_datetime(df_past['Date'])
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
df_past = df_past.sort_values('Date')

print("Veriler başarıyla yüklendi.")

# ======================================================
# 2. LOG GETİRİ HESAPLAMA
# ======================================================
df_past['Returns'] = np.log(df_past['Close'] / df_past['Close'].shift(1))
df_past = df_past.dropna()

# Excel çıktısı
df_past[['Date', 'Close', 'Returns']].to_excel(
    'silver_log_returns.xlsx', index=False
)
print("Log getiriler Excel dosyasına yazıldı.")

# ======================================================
# 3. HMM MODELİ
# ======================================================
X = df_past['Returns'].values.reshape(-1, 1)

model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=1000,
    random_state=42
)
model.fit(X)

df_past['State'] = model.predict(X)

# ======================================================
# 4. MODEL PARAMETRELERİ
# ======================================================
print("\n==============================")
print("HMM MODEL PARAMETRELERİ")
print("==============================")

for i in range(model.n_components):
    mean = model.means_[i][0]
    var = np.diag(model.covars_[i])[0]
    print(f"Durum {i}: Ortalama Getiri = {mean:.6f}, Varyans = {var:.6f}")

print("\nDurum Geçiş Matrisi (A):")
A = pd.DataFrame(model.transmat_,
                 columns=[f"To {i}" for i in range(3)],
                 index=[f"From {i}" for i in range(3)])
print(A)

# ======================================================
# 5. REJİM ETİKETLEME
# ======================================================
state_stats = pd.DataFrame({
    "State": range(3),
    "Mean_Return": model.means_.flatten(),
    "Variance": [np.diag(c)[0] for c in model.covars_]
}).sort_values("Mean_Return")

state_labels = {
    state_stats.iloc[0]["State"]: "Ayı Piyasası",
    state_stats.iloc[1]["State"]: "Yatay Piyasa",
    state_stats.iloc[2]["State"]: "Boğa Piyasası"
}

df_past['Regime'] = df_past['State'].map(state_labels)

print("\nRejim Etiketleri:")
print(state_stats)

# ======================================================
# 6. BEKLENEN REJİM SÜRELERİ
# ======================================================
print("\nBeklenen Rejim Süreleri (gün):")
for i in range(3):
    duration = 1 / (1 - model.transmat_[i, i])
    print(f"{state_labels[i]}: {duration:.2f} gün")

# ======================================================
# 7. GÖRSELLEŞTİRME
# ======================================================
plt.figure(figsize=(14, 7))
colors = ['red', 'blue', 'green']

for i in range(3):
    mask = df_past['State'] == i
    plt.scatter(df_past['Date'][mask],
                df_past['Close'][mask],
                s=8, color=colors[i],
                label=state_labels[i])

plt.plot(df_forecast['Date'],
         df_forecast['Predicted_Close'],
         'k--', linewidth=2, label="2026 Projeksiyonu")

plt.title("Gümüş Fiyatları – Saklı Markov Rejim Analizi")
plt.xlabel("Tarih")
plt.ylabel("Fiyat ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# NaN olmayan durumları al
states = df_past['State'].dropna().astype(int).values

n_states = model.n_components

# Geçiş sayıları matrisi
transition_counts = np.zeros((n_states, n_states))

for s1, s2 in zip(states[:-1], states[1:]):
    transition_counts[s1, s2] += 1

# DataFrame'e çevir
transition_counts_df = pd.DataFrame(
    transition_counts,
    index=[f"From {i}" for i in range(n_states)],
    columns=[f"To {i}" for i in range(n_states)]
)

print("\n--- Rejim Geçiş Sayıları ---")
print(transition_counts_df)
plt.figure(figsize=(14, 4))

plt.plot(df_past['Date'], df_past['State'], drawstyle='steps-post')

plt.yticks(
    ticks=[0, 1, 2],
    labels=[state_labels[0], state_labels[1], state_labels[2]]
)

plt.xlabel("Tarih")
plt.ylabel("Piyasa Rejimi")
plt.title("Gizli Markov Modeli – Rejim Zaman Serisi")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. 2026 VERİSİ İÇİN REJİM TAHMİNİ
# ======================================================

# 2026 için log getiri hesapla
df_forecast['Returns'] = np.log(
    df_forecast['Predicted_Close'] /
    df_forecast['Predicted_Close'].shift(1)
)

df_forecast = df_forecast.dropna()

# HMM sadece sınıflandırma yapıyor (fit YOK)
X_2026 = df_forecast['Returns'].values.reshape(-1, 1)
df_forecast['Predicted_State'] = model.predict(X_2026)

# Rejim etiketlerini ekle
df_forecast['Predicted_Regime'] = df_forecast['Predicted_State'].map(state_labels)

print("\n==============================")
print("2026 REJİM DAĞILIMI")
print("==============================")
# Son gözlemin rejimi (2025 sonu)
last_state = df_past['State'].iloc[-1]

n_states = model.n_components

# Başlangıç olasılık vektörü (one-hot)
pi_0 = np.zeros(n_states)
pi_0[int(last_state)] = 1.0

# Geçiş matrisi
A = model.transmat_

# 1 adım sonrası (ör: 1 ay / 1 dönem sonrası)
pi_1 = pi_0 @ A

# 3 adım sonrası (ör: 2026 ilk çeyrek)
pi_3 = pi_0 @ np.linalg.matrix_power(A, 3)

# 6 adım sonrası (2026 ortası varsayımı)
pi_6 = pi_0 @ np.linalg.matrix_power(A, 6)

# Sonuçları tabloya al
regime_prob_df = pd.DataFrame({
    "Rejim": [state_labels[i] for i in range(n_states)],
    "1 Adım Sonrası Olasılık": pi_1,
    "3 Adım Sonrası Olasılık (2026 Q1)": pi_3,
    "6 Adım Sonrası Olasılık (2026 Ortası)": pi_6
})

print(regime_prob_df)

# ======================================================
# 9. 2026 REJİM OLASILIKLARI GRAFİĞİ
# ======================================================

regime_prob_df.set_index("Rejim").plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("2026 Rejim Olasılık Tahmini (HMM)")
plt.ylabel("Olasılık")
plt.xlabel("Piyasa Rejimi")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# ======================================================
# 10. 2026 GÜMÜŞ FİYAT TAHMİNİ (REJİM AĞIRLIKLI)
# ======================================================

print("\n==============================")
print("2026 GÜMÜŞ FİYAT TAHMİNİ")
print("==============================")

# Son bilinen fiyat
last_price = df_past['Close'].iloc[-1]

# Rejim ortalama getirileri
state_means = model.means_.flatten()

# 2026 Q1 için rejim olasılıkları (önceden hesaplanan)
prob_2026 = pi_3   # 3 adım sonrası

# Beklenen günlük log getiri
expected_log_return = np.sum(prob_2026 * state_means)

# Tahmin süresi (örnek: 60 işlem günü)
forecast_days = 60

# Tahmini fiyat
predicted_price_2026 = last_price * np.exp(forecast_days * expected_log_return)

print(f"Son Bilinen Gümüş Fiyatı: {last_price:.2f} $")
print(f"Beklenen Günlük Log Getiri: {expected_log_return:.6f}")
print(f"2026 Q1 Tahmini Gümüş Fiyatı: {predicted_price_2026:.2f} $")
