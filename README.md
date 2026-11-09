# LAPORAN PROYEK MACHINE LEARNING

## INFORMASI PROYEK

**Judul Proyek:** Analisis dan Klasifikasi Sentimen Berbasis Aspek pada Ulasan E-Commerce

**Identitas Kelompok:**

| Nama | NIM | Program Studi |
|-------|-----------|-----------------|
| Anggi Wahyu Saputra | 244311004 | Teknologi Rekayasa Perangkat Lunak |
| Kevin Rafael Suryatmoko | 244311017 | Teknologi Rekayasa Perangkat Lunak |

**Program Studi:** Teknologi Rekayasa Perangkat Lunak (TRPL)  
**Mata Kuliah:** Machine Learning / Pembelajaran Mesin  
**Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.  
**Tahun Akademik:** 2026/2027  
**Link GitHub Repository:** [Nanti diisi URL Repository]  
**Link Video Pembahasan:** [Nanti diisi URL Video]  
**SDG yang Dipilih:** SDG 8 – Decent Work & Economic Growth  
**Target SDG yang Relevan:** Target 8.2 – Achieve higher levels of economic productivity through diversification, technological upgrading and innovation, including through a focus on high-value added and labour-intensive sectors

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti **ML Lifecycle** yang terdiri dari empat tahap berulang (iteratif):

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|-------|-----------|-----------------|----------------|
| **Learning Problem** | **L** | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization** | **O** | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P** | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. **[L]** Mengidentifikasi masalah nyata dari **ide sendiri** yang selaras dengan salah satu tujuan **SDGs** ([Agenda 2030 – sdgs.un.org/2030agenda](https://sdgs.un.org/2030agenda)), lalu merumuskan problem statement (target, objective, data) secara jelas
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. **[L]** Melakukan data preparation yang sesuai dengan karakteristik dataset
4. **[M]** Merancang tiga model machine learning dengan justifikasi hypothesis space dan inductive bias (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. **[O]** Mengoptimasi model secara sistematis (loss function, optimizer, hyperparameter tuning, regularisasi)
6. **[P]** Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML dan menganalisis hasilnya
7. Melaporkan hasil eksperimen secara ilmiah dan sistematis
8. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
9. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW  **[L]**

### 2.1 Latar Belakang
Pelaku usaha mikro, kecil, dan menengah (UMKM) di platform e-commerce sering kali kewalahan menganalisis ribuan ulasan pelanggan yang masuk setiap harinya secara manual. Ulasan tersebut umumnya bersifat majemuk; di mana pelanggan bisa memuji satu hal namun mengkritik hal lain dalam satu kalimat. Analisis sentimen tradisional yang hanya mengklasifikasikan ulasan menjadi positif atau negatif secara umum tidak cukup memberikan wawasan yang bisa dieksekusi (*actionable insight*).

Oleh karena itu, diperlukan Aspect-Based Sentiment Analysis (ABSA) untuk mengekstraksi sentimen berdasarkan aspek spesifik seperti kualitas produk, harga, pengiriman, dan pelayanan (Liu, 2012). Dengan adanya sistem otomasi ini, UMKM dapat mengetahui secara presisi area mana yang perlu diperbaiki (misalnya, mengganti jasa ekspedisi tanpa harus mengubah kualitas produk) untuk meningkatkan daya saing di pasar digital.

**Referensi:**
- Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Synthesis Lectures on Human Language Technologies.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS).

### 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Jawaban Anda |
|----------|--------------|
| **SDG yang dipilih** | Goal 8: Decent Work & Economic Growth |
| **Target SDG** | Target 8.2: Achieve higher levels of economic productivity through diversification, technological upgrading and innovation, including through a focus on high-value added and labour-intensive sectors |
| **Masalah nyata** | UMKM kesulitan memproses data ulasan dalam jumlah besar secara manual, sehingga inovasi produk dan perbaikan layanan sering salah sasaran. |
| **Peran ML** | Machine Learning (khususnya NLP) mampu membedah teks bahasa natural secara otomatis ke dalam kategori sentimen per aspek, menggantikan analisis manual. |
| **Indikator kontribusi** | Otomatisasi ini membantu UMKM mengambil keputusan berbasis data (*data-driven decision*) dengan lebih cepat, mengurangi *churn rate* pelanggan, dan meningkatkan produktivitas penjualan. |
| **Pertimbangan etika** | Terdapat potensi bias dari ulasan palsu (*fake reviews*) atau ragam bahasa gaul yang tidak baku. Privasi terjaga karena dataset menggunakan data publik yang sudah ter-anonimisasi. |

### 2.3 Ringkasan Alur Lifecycle Proyek
Proyek ini dimulai dengan mendefinisikan dataset ulasan e-commerce berbahasa Indonesia dan merumuskannya sebagai tugas klasifikasi multikelas/multilabel **[L]**. Selanjutnya, dirancang tiga model: Logistic Regression (baseline), LightGBM (advanced), dan LSTM/IndoBERT (deep learning) dengan mengekstraksi fitur teks (TF-IDF/Embedding) **[M]**. Model tersebut akan dioptimasi melalui hyperparameter tuning **[O]** dan dievaluasi kinerjanya menggunakan metrik F1-Score, Precision, dan Recall **[P]**, guna melihat model mana yang terbaik untuk sistem UMKM.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements
1. Pelaku UMKM membutuhkan waktu yang lama untuk mengkategorikan masalah secara manual dari ribuan ulasan pelanggan.
2. Analisis sentimen tingkat dokumen (general) gagal menangkap sentimen multidimensi (misal: "barang bagus tapi pengiriman lambat").
3. Dibutuhkan model machine learning yang tangguh terhadap ragam bahasa tidak baku (slang) pada ulasan e-commerce berbahasa Indonesia.

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Jawaban Anda |
|----------|--------------|
| **Target** | Klasifikasi teks (Supervised Learning) multilabel/multikelas untuk mendeteksi sentimen (Positif/Negatif/Netral) pada aspek spesifik (Produk, Harga, Pengiriman, Layanan). |
| **Objective** | Mengoptimalkan *Cross-entropy loss*. Metrik keberhasilan utama adalah F1-Score (karena potensi ketidakseimbangan kelas) dan Accuracy. |
| **Data** | Data teks ulasan publik e-commerce (Tokopedia, Shopee) berbahasa Indonesia (sekitar 3.000 - 5.000 baris) berformat CSV dari Kaggle. Asumsi data adalah independen dan representatif terhadap percakapan *online marketplace*. |

### 3.3 Goals
1. Mengotomatisasi analisis ulasan per aspek untuk membantu UMKM mengambil keputusan bisnis dengan lebih cepat (mendukung SDG 8.2).
2. Membangun model machine learning dan deep learning untuk ekstraksi aspek dan klasifikasi sentimen dengan target F1-Score > 80%.
3. Membandingkan performa pendekatan tradisional (TF-IDF) dengan algoritma *Gradient Boosting* (LightGBM) dan Deep Learning dalam menangani teks ulasan bahasa Indonesia.

### 3.4 Solution Approach  **[M]**

#### **Model 1 – Baseline Model**
**Logistic Regression / Naive Bayes + TF-IDF**
Model ini dipilih sebagai *baseline* karena kesederhanaannya, sangat efisien secara komputasi, dan sering kali menjadi tolak ukur standar yang baik dan *robust* untuk tugas klasifikasi teks berbasis *bag-of-words* atau TF-IDF.

#### **Model 2 – Advanced / ML Model**
**LightGBM Classifier + TF-IDF**
LightGBM adalah varian *Gradient Boosting* yang sangat cepat dan efisien dalam menangani fitur berdimensi tinggi (yang dihasilkan oleh TF-IDF). LightGBM dipilih sebagai model utama karena mampu memodelkan hubungan non-linear dari kata-kata yang tidak dapat ditangkap oleh *Logistic Regression*, dengan waktu training yang jauh lebih cepat dibandingkan *Random Forest* standar.

#### **Model 3 – Deep Learning Model (WAJIB)**
**LSTM (Long Short-Term Memory) / BiLSTM + Word Embedding**
Untuk dataset berupa teks (NLP), arsitektur RNN seperti LSTM dipilih karena *inductive bias*-nya mampu memahami konteks berurutan (*sequential dependency*) dari susunan kata dalam kalimat. Tidak seperti TF-IDF yang menghilangkan urutan kata, LSTM dapat mendeteksi makna kata berdasarkan kata sebelum dan sesudahnya (misal mendeteksi negasi: "tidak bagus"). *(Alternatif: menggunakan Transfer Learning dengan Pre-trained model IndoBERT jika resource memadai).*

---

## 4. DATA UNDERSTANDING  **[L]**
### 4.1 Informasi Dataset
**Sumber Dataset (Kaggle):**  
- Utama: [E-Commerce Sampled Reviews in Bahasa Indonesia](https://www.kaggle.com/datasets/satyaahb/e-commerce-sampled-reviews-in-bahasa-indonesia)
- Alternatif 1: [Shopee Fashion Reviews Indonesian](https://www.kaggle.com/datasets/jareedhnnn/shopee-fashion-reviews-indonesian/data)
- Alternatif 2: [Tokopedia Product Reviews 2025](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025)

**Deskripsi Dataset:**
- Jumlah baris (rows): ± 3.000 – 5.000 baris data
- Tipe data: Text / Tabular
- Format file: CSV

### 4.2 Deskripsi Fitur (Rencana)
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| review_text| String | Teks ulasan murni pelanggan | "Barangnya mantap tapi kurirnya lama" |
| rating | Integer | Rating bintang (1-5) | 1, 3, 5 |
| aspect_category| Categorical| Aspek yang dibicarakan (Label) | Kualitas, Pengiriman, Harga |
| sentiment | Categorical| Sentimen per aspek (Label) | Positif, Negatif, Netral |
*(Catatan: Penyesuaian kolom akan dilakukan bergantung pada dataset final yang digunakan dan proses anotasi aspek).*

### 4.3 Kondisi Data
*(Akan diisi setelah dataset diunduh dan dieksplorasi)*
- **Missing Values:** [Akan dicek, biasanya ada review kosong]
- **Noise:** Data teks e-commerce penuh dengan singkatan (slang), *typo*, dan emoji.
- **Imbalanced Data:** Kemungkinan jumlah ulasan positif jauh lebih banyak (rating 5) dibanding negatif/netral.

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)
*(Silakan tambahkan plot Wordcloud, Distribusi Label Sentimen, dan Distribusi Panjang Kalimat di sini saat tahap pengerjaan)*

---

## 5. DATA PREPARATION  **[L]**
*(Diisi saat pengerjaan. Panduan tahapan untuk teks:)*
- **Data Cleaning:** Menghapus missing values, duplikat, tag HTML, dan karakter non-alfanumerik.
- **Text Preprocessing:** *Case folding* (lowercasing), *Normalization* (memperbaiki slang/typo bahasa gaul), *Stopwords removal*, dan *Stemming* (menggunakan library Sastrawi).
- **Transformation (TF-IDF):** Mengubah teks menjadi vektor numerik untuk Baseline dan LightGBM.
- **Transformation (Embedding/Padding):** Melakukan tokenisasi dan *sequence padding* untuk model Deep Learning (LSTM).
- **Data Splitting:** Membagi data Train 80% dan Test 20% dengan *stratified split*.
- **Data Balancing:** Menggunakan SMOTE (jika klasifikasi TF-IDF) atau penyesuaian *class weight* jika sentimen tidak seimbang.

---

*(Bab 6 dan Bab 7 akan diisi dengan kode dan hasil analisis nyata setelah Anda melakukan eksperimen di Jupyter Notebook / Google Colab)*

## 6. MODELING  **[M + O]**
*(Template tetap dari dosen - akan diisi kemudian)*
### 6.1 Model 1 — Baseline Model
### 6.2 Model 2 — ML / Advanced Model (LightGBM)
### 6.3 Model 3 — Deep Learning Model (LSTM / IndoBERT)

## 7. EVALUATION  **[P]**
*(Template tetap dari dosen - akan diisi kemudian)*

## 8. CONCLUSION
*(Template tetap dari dosen - akan diisi kemudian)*

## 9. FUTURE WORK (Opsional)
*(Template tetap dari dosen - akan diisi kemudian)*

## 10. REPRODUCIBILITY (WAJIB)
*(Diisi URL GitHub dan requirements saat proyek selesai)*

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | 1. Anggi Wahyu Saputra / 244311004 <br> 2. Kevin Rafael Suryatmoko / 244311017 |
| Judul ide | Analisis dan Klasifikasi Sentimen Berbasis Aspek pada Ulasan E-Commerce |
| SDG & target yang dipilih | Goal 8: Pekerjaan Layak dan Pertumbuhan Ekonomi — Target 8.2 |
| Masalah nyata yang diangkat | UMKM di e-commerce kesulitan menganalisis ulasan pelanggan secara spesifik dan manual. Analisis konvensional tidak dapat mendeteksi masalah per komponen (produk, pengiriman, harga), sehingga UMKM lambat dalam mengevaluasi bisnis. |
| Tugas ML (Target & jenis tugas) | Klasifikasi Teks (Supervised Learning) untuk mendeteksi Aspek dan Sentimen (Positif / Negatif / Netral). |
| Objective & metrik utama | Mengotomatisasi analisis *actionable insight*. Metrik utama: F1-Score dan Accuracy. |
| Dataset (sumber, URL, ukuran, jenis data) | Kaggle – E-Commerce Sampled Reviews in Bahasa Indonesia (Teks CSV, ±3.000 - 5.000 baris) beserta 2 alternatif dataset serupa. |
| Rencana tiga model (baseline / advanced / deep learning) | 1. Baseline: Logistic Regression + TF-IDF<br>2. Advanced: LightGBM + TF-IDF<br>3. Deep Learning: LSTM / BiLSTM + Word Embedding |
| Risiko & pertimbangan etika (bias, privasi) | Bias ragam bahasa tidak baku; Aman secara privasi karena dataset publik ter-anonimisasi. |
| Referensi awal (minimal 2) | 1. Liu, B. (2012). *Sentiment Analysis and Opinion Mining*.<br>2. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting...* |
