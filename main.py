# Library untuk manipulasi data
import pandas as pd
import numpy as np

# Library untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Library untuk machine learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Load data
df = pd.read_csv("data/student_habits_performance.csv")

# Melihat struktur kolom dan tipe data
df.info()

# Statistik deskriptif kolom 'parental_education_level'
df['parental_education_level'].describe()

# Mengisi missing value dengan modus (nilai terbanyak)
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])

# Verifikasi tidak ada nilai kosong
df.isna().sum()

# Menentukan kolom kategorikal (tipe objek)
cat_col = df.select_dtypes(include='object').columns.tolist()

# Menghapus kolom student_id karena tidak relevan
cat_col.remove('student_id')

# Visualisasi frekuensi masing-masing kategori
plt.figure(figsize=(9, 6))
for i in range(len(cat_col)):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[cat_col[i]], color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {cat_col[i]}')

plt.tight_layout()
plt.show()

# Menentukan kolom numerik
num_col = df.select_dtypes(exclude='object').columns.tolist()

# Visualisasi distribusi numerik
plt.figure(figsize=(12, 12))
for i in range(len(num_col)):
    plt.subplot(3, 3, i + 1)
    plt.hist(df[num_col[i]], bins=20, edgecolor='black')
    plt.title(f'Distribution of {num_col[i]}')

plt.tight_layout()
plt.show()

# Menghapus kolom student_id karena bukan fitur prediktif
df2 = df.drop('student_id', axis=1)

# Pemetaan manual berdasarkan urutan kualitas
diet_quality = {'Poor': 0, 'Fair': 1, 'Good': 2}
parental_education_level = {'High School': 0, 'Bachelor': 1, 'Master': 2}
internet_quality = {'Poor': 0, 'Average': 1, 'Good': 2}

# Menambahkan kolom baru hasil encoding
df2['dq_e'] = df2['diet_quality'].map(diet_quality)
df2['pel_e'] = df2['parental_education_level'].map(parental_education_level)
df2['iq_e'] = df2['internet_quality'].map(internet_quality)

# One-hot encoding untuk kolom kategorikal nominal
dummies = pd.get_dummies(df[['gender', 'part_time_job', 'extracurricular_participation']], drop_first=True)

# Gabungkan df2 dan hasil one-hot encoding
df3 = pd.concat([df2, dummies], axis=1)

# Drop kolom asli yang sudah di-encode
df3 = df3.drop([
    'gender',
    'part_time_job',
    'diet_quality',
    'parental_education_level',
    'internet_quality',
    'extracurricular_participation'
], axis=1)

# Menampilkan heatmap korelasi
plt.figure(figsize=(12, 10))
corr = df3.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pisahkan X (fitur) dan y (target)
X = df3.drop('exam_score', axis=1)
y = df3['exam_score']

# Normalisasi fitur dengan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi data latih dan uji (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Inisialisasi model utama dan pembanding
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Fungsi evaluasi

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)                # Training model
    y_pred = model.predict(X_test)             # Prediksi

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n{name}")
    print("-" * 40)
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.2f}")
    print(f"MAPE : {mape:.2f}%")

    return name, rmse, mae, r2, mape

# Evaluasi semua model
results = []
for name, model in models.items():
    result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results.append(result)

# Ubah hasil ke DataFrame
result_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R2_Score', 'MAPE (%)'])

# Tampilkan hasil
print("\n📊 Hasil Evaluasi Semua Model:")
print(result_df)

# Visualisasi hasil evaluasi
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Evaluation Metrics Comparison', fontsize=16, fontweight='bold')

# Barplot untuk RMSE
sns.barplot(ax=axs[0, 0], data=result_df, x='Model', y='RMSE', hue='Model', palette='crest', legend=False)
axs[0, 0].set_title('RMSE (Root Mean Squared Error)')
axs[0, 0].set_ylabel('RMSE')
axs[0, 0].set_xlabel('Model')
axs[0, 0].tick_params(axis='x', rotation=15)

# Barplot untuk MAE
sns.barplot(ax=axs[0, 1], data=result_df, x='Model', y='MAE', hue='Model', palette='magma', legend=False)
axs[0, 1].set_title('MAE (Mean Absolute Error)')
axs[0, 1].set_ylabel('MAE')
axs[0, 1].set_xlabel('Model')
axs[0, 1].tick_params(axis='x', rotation=15)

# Barplot untuk R² Score
sns.barplot(ax=axs[1, 0], data=result_df, x='Model', y='R2_Score', hue='Model', palette='viridis', legend=False)
axs[1, 0].set_title('R² Score')
axs[1, 0].set_ylabel('R²')
axs[1, 0].set_xlabel('Model')
axs[1, 0].set_ylim(0, 1)
axs[1, 0].tick_params(axis='x', rotation=15)

# Barplot untuk MAPE
sns.barplot(ax=axs[1, 1], data=result_df, x='Model', y='MAPE (%)', hue='Model', palette='rocket', legend=False)
axs[1, 1].set_title('MAPE (Mean Absolute Percentage Error)')
axs[1, 1].set_ylabel('MAPE (%)')
axs[1, 1].set_xlabel('Model')
axs[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
