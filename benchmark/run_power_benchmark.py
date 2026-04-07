"""
MP-LRNet Benchmark — UCI Household Electric Power Consumption
Gunluk enerji tuketimi — 1433 gun, 7 ozellik
Haftalik dongu BELIRGIN: hafta sonu yuksek, hafta ici dusuk
[7,14] patch = haftalik/iki haftalik dongu — mobilya verisiyle AYNI mantik

Literaturde raporlanan skorlar:
  - CNN-LSTM-Transformer: R2=0.9928
  - DNN-LSTM: R2=0.9991
  - CNN-LSTM: R2=0.894
  - CNN-GRU: R2=0.922

5 konfigurasyon x 10 iterasyon = 50 deney
Metrikler: R2, MSE, MAE, RMSE, MAPE
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from math import sqrt
import time
import json
import zipfile

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# YARDIMCI FONKSIYONLAR
# =====================================================
def create_time_series_data(data, time_steps=50):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, :])
        y.append(data[i, :])
    return np.array(X), np.array(y)

def run_experiment(train_scaled, test_scaled, scaler, target_col_idx, n_features,
                   patch_sizes, use_rf, n_iters=10, dataset_name=""):
    all_metrics = {'r2': [], 'mse': [], 'mae': [], 'rmse': [], 'mape': []}

    for iteration in range(n_iters):
        print(f"  [{dataset_name}] Iter {iteration+1}/{n_iters}...", end=" ", flush=True)
        t0 = time.time()

        models = []
        for ps in patch_sizes:
            X_train, y_train = create_time_series_data(train_scaled, time_steps=ps)
            model = Sequential([
                Dense(200, activation='relu', input_shape=(ps, n_features)),
                LSTM(200, activation='relu', dropout=0.2, recurrent_dropout=0.2),
                Dense(n_features)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
            models.append(model)

        predictions = []
        min_len = float('inf')
        for ps, mdl in zip(patch_sizes, models):
            X_test, y_test = create_time_series_data(test_scaled, time_steps=ps)
            pred = mdl.predict(X_test, verbose=0)
            predictions.append(pred)
            min_len = min(min_len, len(pred))

        predictions = [p[:min_len] for p in predictions]
        lstm_predicted = np.mean(predictions, axis=0)

        smallest_ps = min(patch_sizes)
        _, y_test_all = create_time_series_data(test_scaled, time_steps=smallest_ps)
        y_test_trimmed = y_test_all[:min_len]

        if use_rf:
            rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            rf.fit(lstm_predicted, y_test_trimmed)
            final_pred = rf.predict(lstm_predicted)
        else:
            final_pred = lstm_predicted

        final_pred_inv = scaler.inverse_transform(final_pred)
        y_test_inv = scaler.inverse_transform(y_test_trimmed)

        y_true = y_test_inv[:, target_col_idx]
        y_pred = final_pred_inv[:, target_col_idx]

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mse)
        epsilon = 1e-6
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

        all_metrics['r2'].append(r2)
        all_metrics['mse'].append(mse)
        all_metrics['mae'].append(mae)
        all_metrics['rmse'].append(rmse)
        all_metrics['mape'].append(mape)

        elapsed = time.time() - t0
        print(f"R2={r2:.4f}, MSE={mse:.6f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, t={elapsed:.0f}s")

        for mdl in models:
            del mdl
        tf.keras.backend.clear_session()

    return {k: {'mean': np.mean(v), 'std': np.std(v), 'values': v} for k, v in all_metrics.items()}

# =====================================================
# VERI SETI INDIR VE HAZIRLA
# =====================================================
print("UCI Household Electric Power Consumption veri seti hazirlaniyor...")

import urllib.request
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
zip_path = os.path.join(SAVE_DIR, "household_power.zip")
txt_path = os.path.join(SAVE_DIR, "household_power_consumption.txt")

if not os.path.exists(txt_path):
    print("  Indiriliyor...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(SAVE_DIR)
    print("  Indirildi ve cikarildi.")
else:
    print("  Veri mevcut.")

# Oku ve isle
df = pd.read_csv(txt_path, sep=';', low_memory=False)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('datetime', inplace=True)

# Numerik donusum
for col in ['Global_active_power', 'Global_reactive_power', 'Voltage',
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Gunluk ortalamaya topla
feature_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
daily = df[feature_cols].resample('D').mean().dropna()

print(f"  Gunluk veri: {len(daily)} gun ({daily.index.min().date()} - {daily.index.max().date()})")
print(f"  Sutunlar: {list(daily.columns)}")
print(f"  Haftalik dongu: Cts={daily[daily.index.dayofweek==5]['Global_active_power'].mean():.3f}, "
      f"Pzt={daily[daily.index.dayofweek==0]['Global_active_power'].mean():.3f} kW")

# =====================================================
# DENEYLER
# =====================================================
features = daily.values
n_features = features.shape[1]
target_idx = daily.columns.tolist().index('Global_active_power')  # = 0

split = int(len(features) * 0.8)
train_data = features[:split]
test_data = features[split:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

print(f"\n  Train: {len(train_data)} gun, Test: {len(test_data)} gun")
print(f"  Features: {n_features}, Target: Global_active_power (idx={target_idx})")

print("\n" + "=" * 70)
print("UCI POWER CONSUMPTION — 5 KONFIGURASYON")
print("[7,14] patch = haftalik/iki haftalik dongu")
print("=" * 70)

results = {}

# Config 1: MP-LRNet (LSTM + MP[7,14] + RF)
print("\n--- [1/5] MP-LRNet (LSTM + MP[7,14] + RF) ---")
results['MP-LRNet'] = run_experiment(
    train_scaled, test_scaled, scaler, target_idx, n_features,
    patch_sizes=[7, 14], use_rf=True, n_iters=10, dataset_name="Power-MPLRNet")

# Config 2: LSTM Only (patch=14, no RF)
print("\n--- [2/5] LSTM Only (patch=14, no RF) ---")
results['LSTM-Only'] = run_experiment(
    train_scaled, test_scaled, scaler, target_idx, n_features,
    patch_sizes=[14], use_rf=False, n_iters=10, dataset_name="Power-LSTM")

# Config 3: LSTM + RF (patch=14, no multi-patch)
print("\n--- [3/5] LSTM + RF (patch=14) ---")
results['LSTM+RF'] = run_experiment(
    train_scaled, test_scaled, scaler, target_idx, n_features,
    patch_sizes=[14], use_rf=True, n_iters=10, dataset_name="Power-LSTM+RF")

# Config 4: LSTM + Multi-Patch[7,14] (no RF)
print("\n--- [4/5] LSTM + MP[7,14] (no RF) ---")
results['LSTM+MP'] = run_experiment(
    train_scaled, test_scaled, scaler, target_idx, n_features,
    patch_sizes=[7, 14], use_rf=False, n_iters=10, dataset_name="Power-LSTM+MP")

# Config 5: MP-LRNet [7,28] (haftalik + aylik)
print("\n--- [5/5] MP-LRNet-28 (LSTM + MP[7,28] + RF) ---")
results['MP-LRNet-7,28'] = run_experiment(
    train_scaled, test_scaled, scaler, target_idx, n_features,
    patch_sizes=[7, 28], use_rf=True, n_iters=10, dataset_name="Power-MP728")

# =====================================================
# SONUCLARI KAYDET
# =====================================================
print("\n" + "=" * 70)
print("SONUC OZETI — UCI POWER CONSUMPTION")
print("=" * 70)
print(f"\n{'Model':<22} {'R2':>16} {'MSE':>12} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
print("-" * 80)
for model, m in results.items():
    r2 = f"{m['r2']['mean']:.4f}+/-{m['r2']['std']:.4f}"
    mse = f"{m['mse']['mean']:.6f}"
    mae = f"{m['mae']['mean']:.4f}"
    rmse = f"{m['rmse']['mean']:.4f}"
    mape = f"{m['mape']['mean']:.2f}%"
    print(f"{model:<22} {r2:>16} {mse:>12} {mae:>10} {rmse:>10} {mape:>10}")

# CSV
rows = []
for model, m in results.items():
    rows.append({
        'Dataset': 'UCI_Power',
        'Model': model,
        'R2_mean': round(m['r2']['mean'], 6),
        'R2_std': round(m['r2']['std'], 6),
        'MSE_mean': round(m['mse']['mean'], 6),
        'MSE_std': round(m['mse']['std'], 6),
        'MAE_mean': round(m['mae']['mean'], 6),
        'MAE_std': round(m['mae']['std'], 6),
        'RMSE_mean': round(m['rmse']['mean'], 6),
        'RMSE_std': round(m['rmse']['std'], 6),
        'MAPE_mean': round(m['mape']['mean'], 6),
        'MAPE_std': round(m['mape']['std'], 6),
    })

df_out = pd.DataFrame(rows)
csv_path = os.path.join(SAVE_DIR, "power_results.csv")
df_out.to_csv(csv_path, index=False)
print(f"\nCSV: {csv_path}")

# JSON
detail = {'UCI_Power': {}}
for model, m in results.items():
    detail['UCI_Power'][model] = {k: [round(float(x), 6) for x in v['values']] for k, v in m.items()}

json_path = os.path.join(SAVE_DIR, "power_detail.json")
with open(json_path, 'w') as f:
    json.dump(detail, f, indent=2)
print(f"JSON: {json_path}")

print(f"\n{'='*70}")
print("TAMAMLANDI! 5 konfigurasyon x 10 iterasyon = 50 deney")
print(f"{'='*70}")
