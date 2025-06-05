# Laporan Proyek Machine Learning - Prediksi Performa Akademik Berdasarkan Kebiasaan Siswa

## Domain Proyek

Pendidikan merupakan salah satu faktor utama dalam pembangunan sumber daya manusia. Namun, performa akademik siswa sering kali dipengaruhi oleh berbagai faktor, termasuk kebiasaan belajar, kualitas diet, aktivitas ekstrakurikuler, pekerjaan paruh waktu, dan lingkungan keluarga. Dengan memahami faktor-faktor ini, institusi pendidikan dapat mengambil langkah preventif dan intervensi yang tepat untuk meningkatkan prestasi siswa.

Beberapa penelitian menunjukkan bahwa kebiasaan belajar yang baik, dukungan keluarga, dan pola hidup sehat berkontribusi signifikan terhadap pencapaian akademik siswa [[1]](https://www.sciencedirect.com/science/article/pii/S1877042815043227). Oleh karena itu, analisis prediktif berbasis machine learning dapat membantu mengidentifikasi siswa yang berisiko dan memberikan rekomendasi berbasis data.

---

## Business Understanding

### Problem Statements

1. **Apa saja kebiasaan siswa yang paling berpengaruh terhadap nilai ujian mereka?**  
2. **Fitur atau kebiasaan apa yang paling berkontribusi terhadap performa akademik siswa?**  
3. **Seberapa baik model machine learning dapat memprediksi nilai ujian siswa berdasarkan data kebiasaan dan latar belakang mereka?**

### Goals

- Mengidentifikasi faktor-faktor utama yang memengaruhi performa akademik siswa.
- Membangun model prediksi nilai ujian siswa berdasarkan fitur kebiasaan dan latar belakang.
- Memberikan insight bagi sekolah/orang tua untuk meningkatkan performa akademik siswa.

### Solution Statements

- Menggunakan beberapa algoritma regresi (Linear Regression, Random Forest, Gradient Boosting, XGBoost).
- Membandingkan performa model menggunakan metrik RMSE, MAE, R², dan MAPE.
- Memilih model terbaik dan memberikan interpretasi fitur penting.

---

## Data Understanding

**Dataset:** [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance/)  
**Format:** CSV, 1000 baris, 16 kolom.

### Variabel dalam Dataset

| Kolom                           | Tipe    | Deskripsi                         |
| ------------------------------- | ------- | --------------------------------- |
| `student_id`                    | Object  | ID unik tiap siswa                |
| `age`                           | Integer | Usia siswa                        |
| `gender`                        | Object  | Jenis kelamin siswa               |
| `study_hours_per_day`           | Float   | Jam belajar harian                |
| `social_media_hours`            | Float   | Jam media sosial per hari         |
| `netflix_hours`                 | Float   | Jam menonton Netflix              |
| `part_time_job`                 | Object  | Status pekerjaan paruh waktu      |
| `attendance_percentage`         | Float   | Persentase kehadiran              |
| `sleep_hours`                   | Float   | Jam tidur rata-rata               |
| `diet_quality`                  | Object  | Kualitas pola makan               |
| `exercise_frequency`            | Integer | Frekuensi olahraga per minggu     |
| `parental_education_level`      | Object  | Pendidikan orang tua              |
| `internet_quality`              | Object  | Kualitas internet di rumah        |
| `mental_health_rating`          | Integer | Penilaian kesehatan mental (1–10) |
| `extracurricular_participation` | Object  | Partisipasi ekstrakurikuler       |
| `exam_score`                    | Float   | Nilai ujian (Target)              |

### Pemeriksaan Kualitas Data

- **Missing Values**:  
  - `parental_education_level`: 91 nilai kosong  
  - Kolom lain tidak memiliki missing value

- **Data Duplikat**:  
  - Tidak ditemukan duplikasi berdasarkan `student_id`.

---

## Data Preparation

1. **Menghapus Kolom Tidak Relevan**  
   - Kolom `student_id` dihapus karena tidak memiliki nilai prediktif.

2. **Encoding Fitur Kategorikal**
   - Kolom ordinal di-*mapping* secara manual ke nilai numerik berdasarkan urutan:
     - `diet_quality`: {'Poor': 0, 'Fair': 1, 'Good': 2}
     - `internet_quality`: {'Poor': 0, 'Average': 1, 'Good': 2}
     - `parental_education_level`: {'High School': 0, 'Bachelor': 1, 'Master': 2}
   - Kolom nominal (`gender`, `part_time_job`, `extracurricular_participation`) diencoding menggunakan **One-Hot Encoding** dengan `get_dummies()`.

3. **Menghapus Kolom Asli Setelah Encoding**  
   - Kolom-kolom kategorikal asli dihapus setelah encoding dilakukan.

4. **Gabungkan Semua Fitur**  
   - Hasil encoding digabungkan dengan dataset utama menggunakan `concat`.

5. **Pemisahan Fitur dan Target**
   - Fitur (`X`) = semua kolom kecuali `exam_score`.
   - Target (`y`) = kolom `exam_score`.

6. **Feature Scaling**
   - Seluruh fitur dinormalisasi menggunakan `StandardScaler`.

7. **Split Data**
   - Data dibagi menjadi 80% data latih dan 20% data uji dengan `train_test_split(random_state=42)`.

---

## Modeling

### Model yang Digunakan

1. **Linear Regression**  
   - Pendekatan linier untuk meminimalkan kesalahan kuadrat.  
   - Parameter: `LinearRegression()` (default)

2. **Random Forest Regressor**  
   - Ensembel pohon keputusan dari subset data.  
   - Parameter: `RandomForestRegressor(random_state=42)`

3. **Gradient Boosting Regressor**  
   - Pohon dibangun secara sekuensial untuk memperbaiki kesalahan sebelumnya.  
   - Parameter: `GradientBoostingRegressor(random_state=42)`

4. **XGBoost Regressor**  
   - Versi optimasi dari Gradient Boosting dengan efisiensi tinggi.  
   - Parameter: `XGBRegressor(random_state=42, objective='reg:squarederror')`

---

## Evaluation

### Metrik Evaluasi

- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  
- **R² Score (Koefisien Determinasi)**  
- **MAPE (Mean Absolute Percentage Error)**  

### Hasil Evaluasi

| Model             | RMSE | MAE  | R² Score | MAPE (%) |
| ----------------- | ---- | ---- | -------- | -------- |
| Linear Regression | 5.09 | 4.14 | 0.90     | 6.83     |
| Random Forest     | 6.24 | 4.96 | 0.85     | 8.30     |
| Gradient Boosting | 5.57 | 4.66 | 0.88     | 7.58     |
| XGBoost           | 6.36 | 5.14 | 0.84     | 8.35     |

**Kesimpulan Evaluasi:**  
Model **Linear Regression** memberikan performa terbaik dengan nilai error terendah dan R² Score tertinggi.

---

## Jawaban atas Problem Statements

1. **Apa saja kebiasaan siswa yang paling berpengaruh terhadap nilai ujian mereka?**  
   - `study_hours_per_day`: korelasi 0.83  
   - `mental_health_rating`: korelasi 0.32  
   - `exercise_frequency`: korelasi 0.16  
   - `attendance_percentage`: korelasi 0.09  

2. **Fitur apa yang paling berkontribusi terhadap performa akademik siswa?**  
   - **Jam belajar per hari** (tertinggi)  
   - **Kesehatan mental**  
   - **Frekuensi olahraga**  
   - **Persentase kehadiran**

3. **Seberapa baik model memprediksi nilai ujian siswa?**  
   - Model **Linear Regression** memberikan prediksi yang cukup akurat (R² = 0.90).  
   - Cocok untuk deteksi dini siswa berisiko dan rekomendasi intervensi.

---

## Kesimpulan

1. **Linear Regression adalah model terbaik.**  
   - RMSE, MAE, MAPE terendah dan R² tertinggi.

2. **Fitur utama yang memengaruhi nilai ujian:**
   - Jam belajar per hari, kesehatan mental, olahraga, dan kehadiran.

3. **Model dapat digunakan untuk deteksi siswa berisiko.**  
   - Sekolah/orang tua dapat melakukan intervensi berbasis data.

---

![Heatmap Korelasi](image.png)

---

**Catatan:**  
Seluruh analisis, visualisasi, dan hasil evaluasi dapat dilihat di notebook `notebook/analysis.ipynb`.