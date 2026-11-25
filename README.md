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

**Catatan untuk mahasiswa:** Setiap bagian diberi label **[L]**, **[M]**, **[O]**, atau **[P]**. Pastikan Anda dapat menjelaskan keterkaitan antar tahap, misalnya bagaimana hasil **P** mendorong perbaikan pada **L/M/O** (Bab 7.5).

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
Pelaku usaha mikro, kecil, dan menengah (UMKM) di sektor fesyen pada platform e-commerce (seperti Shopee) sering kali kewalahan menganalisis ribuan ulasan pelanggan yang masuk setiap harinya secara manual. Ulasan produk fesyen umumnya bersifat majemuk; di mana pelanggan dapat memuji satu karakteristik produk namun mengkritik bagian lainnya dalam satu kalimat yang sama (misalnya memuji warna produk tetapi mengeluhkan bahan yang tipis atau ukuran yang kekecilan). Analisis sentimen tradisional tingkat dokumen (*document-level*) yang hanya mengklasifikasikan ulasan menjadi positif atau negatif secara umum tidak cukup memberikan wawasan yang dapat langsung dieksekusi (*actionable insight*).

Oleh karena itu, diperlukan **Aspect-Based Sentiment Analysis (ABSA)** untuk mengekstraksi sentimen secara spesifik pada **6 aspek krusial produk fesyen**, yaitu **kualitas, bahan, ukuran, desain, warna, dan kenyamanan** (Liu, 2012). Dengan adanya sistem otomasi berbasis pemrosesan bahasa alami (*Natural Language Processing*), UMKM dapat mengetahui secara presisi area mana yang perlu segera diperbaiki—misalnya memperbaiki panduan ukuran (*size chart*) atau mengganti pemasok kain tanpa harus merombak desain model—guna menekan tingkat pengembalian barang (*return rate*) dan meningkatkan daya saing di pasar digital.

**Referensi:**
- Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Synthesis Lectures on Human Language Technologies.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS).
- Wilie, B., et al. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*. Proceedings of AACL-IJCNLP.

### 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Jawaban Anda |
|----------|--------------|
| **SDG yang dipilih** | Goal 8: Decent Work & Economic Growth |
| **Target SDG** | Target 8.2: Achieve higher levels of economic productivity through diversification, technological upgrading and innovation, including through a focus on high-value added and labour-intensive sectors |
| **Masalah nyata** | UMKM fesyen kesulitan memproses dan mengelompokkan ribuan ulasan pelanggan secara manual untuk menemukan penyebab spesifik keluhan (seperti ukuran tidak pas, kain panas, atau warna pudar), sehingga perbaikan kualitas produk sering salah sasaran. |
| **Peran ML** | Machine Learning dan Deep Learning (khususnya NLP) mampu membedah teks ulasan berbahasa Indonesia secara otomatis ke dalam kategori sentimen pada 6 aspek produk secara serentak (*multi-output*), menggantikan analisis manual. |
| **Indikator kontribusi** | Otomatisasi ini membantu UMKM mengambil keputusan perbaikan produk berbasis data (*data-driven decision*) dengan lebih cepat, mengurangi *return rate* maupun *churn rate* pelanggan, dan meningkatkan produktivitas penjualan. |
| **Pertimbangan etika** | Terdapat potensi bias dari ragam bahasa gaul (*slang*), singkatan e-commerce tidak baku, serta ketimpangan distribusi kelas (*class imbalance*) di mana keluhan negatif pada aspek tertentu berjumlah sangat sedikit. Privasi pelanggan tetap terjaga karena dataset menggunakan data publik yang sudah ter-anonimisasi. |

### 2.3 Ringkasan Alur Lifecycle Proyek
Proyek ini dimulai dengan mendefinisikan dataset ulasan produk fesyen Shopee berbahasa Indonesia dan merumuskannya sebagai tugas klasifikasi *multi-output multiclass* pada 6 aspek sekaligus **[L]**. Selanjutnya, dirancang tiga tingkatan model: **Multi-Output Logistic Regression** (*baseline*), **Tuned LightGBM + ClassifierChain** (*advanced ML*), dan **Multi-Head IndoBERT / BiLSTM** (*deep learning*) dengan mengekstraksi fitur teks menggunakan TF-IDF *unigram–bigram* serta representasi *contextual embedding* **[M]**. Model tersebut dioptimasi melalui pembobotan kelas (*class weighting*) dan *hyperparameter tuning* **[O]**, kemudian dievaluasi kinerjanya menggunakan metrik **Macro F1-Score**, Precision, Recall, dan Accuracy **[P]**, guna menentukan arsitektur terbaik untuk sistem analitik UMKM.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements
1. Pelaku UMKM fesyen membutuhkan waktu yang sangat lama untuk membaca dan mengkategorikan keluhan pelanggan secara manual dari ribuan ulasan produk.
2. Analisis sentimen tingkat dokumen (*general sentiment*) gagal menangkap sentimen multidimensi yang kontras dalam satu ulasan (misal: *"warnanya cantik sesuai foto, tapi bahannya tipis menerawang dan ukuran kekecilan"*).
3. Dibutuhkan model *machine learning* yang tangguh terhadap ragam bahasa tidak baku (*slang* Shopee), struktur negasi, serta ketimpangan kelas ekstrem (*severe class imbalance*) pada aspek dengan ulasan negatif minoritas.

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Jawaban Anda |
|----------|--------------|
| **Target** | Klasifikasi teks (*Supervised Learning*) berbentuk **Multi-Output Multiclass Classification** untuk mendeteksi 3 polaritas sentimen (`Negatif`, `Netral`, `Positif`) secara serentak pada **6 aspek produk fesyen** (`aspek_kualitas`, `aspek_bahan`, `aspek_ukuran`, `aspek_desain`, `aspek_warna`, dan `aspek_kenyamanan`). |
| **Objective** | Mengoptimalkan *Weighted Cross-Entropy Loss* dan pembobotan kelas seimbang (`class_weight='balanced'`). Metrik keberhasilan utama adalah **Macro F1-Score** (untuk mengukur performa deteksi secara adil di tengah dominasi kelas `Netral`) serta *Accuracy* dan *Weighted F1*. |
| **Data** | Data teks ulasan publik produk fesyen Shopee berbahasa Indonesia dari Kaggle (`jareedhnnn/shopee-fashion-reviews-indonesian`, berkas `dataset_labeled_v6.xlsx`) berjumlah **6.158 baris** (serta pengayaan data ulasan negatif bintang 1–3). Asumsi data adalah independen antar-ulasan dan representatif terhadap percakapan ulasan fesyen di *online marketplace*. |

### 3.3 Goals
1. Mengotomatisasi analisis sentimen pada 6 aspek produk fesyen untuk membantu UMKM mengambil keputusan bisnis dan perbaikan produk dengan lebih cepat (mendukung SDG 8.2).
2. Membangun pipeline model *machine learning* dan *deep learning* untuk klasifikasi sentimen berbasis aspek dengan target performa keseluruhan (**Overall Macro F1-Score**) **> 80%**.
3. Membandingkan performa pendekatan linear (*Logistic Regression* + TF-IDF), algoritma *Gradient Boosting* berantai (*LightGBM + ClassifierChain* + TF-IDF *Unigram–Bigram*), dan *Deep Learning* kontekstual (*Multi-Head IndoBERT / BiLSTM*) dalam menangani teks ulasan fesyen bahasa Indonesia.

### 3.4 Solution Approach  **[M]**

#### **Model 1 – Baseline Model**
**Multi-Output Logistic Regression (`class_weight='balanced'`) + TF-IDF**
Model ini dipilih sebagai *baseline* karena kesederhanaannya, sangat efisien secara komputasi, deterministik, serta mudah diinterpretasi melalui koefisien bobot kata per aspek. Penambahan parameter `class_weight='balanced'` diterapkan sebagai tolok ukur awal untuk menekan bias prediksi terhadap kelas mayoritas (`Netral`).

#### **Model 2 – Advanced / ML Model**
**Tuned LightGBM (`ClassifierChain`) + TF-IDF Unigram–Bigram `(1, 2)`**
LightGBM adalah varian *Gradient Boosting Decision Tree* yang sangat cepat dan efisien dalam menangani matriks fitur *sparse* berdimensi tinggi dari TF-IDF. Penggunaan fitur *unigram–bigram* membantu menangkap frasa negasi dua kata (seperti *"tidak luntur"* atau *"nggak kekecilan"*), sementara arsitektur **`ClassifierChain`** memodelkan korelasi berantai antar-aspek (seperti hubungan antara `bahan`, `kenyamanan`, dan `kualitas`) serta mampu menangkap interaksi non-linear dengan penyesuaian `min_child_samples=3` untuk kelas minoritas.

#### **Model 3 – Deep Learning Model (WAJIB)**
**Multi-Head IndoBERT (`indobert-base-p1`) / BiLSTM + Word Embedding**
Untuk dataset berupa teks (NLP), arsitektur berbasis *Transformer* (**Multi-Head IndoBERT** dengan 6 *classification head* paralel) maupun **BiLSTM** dipilih karena kemampuannya memahami konteks kalimat dan ketergantungan urutan kata (*contextual & sequential dependency*). Tidak seperti TF-IDF yang menghilangkan urutan kata dan mencampur sentimen pada kalimat majemuk, arsitektur *Deep Learning* mampu memisahkan sentimen positif dan negatif pada klausa yang berbeda dalam satu ulasan (misal: *"desain bagus tapi bahan panas"*) serta mengenali kedekatan makna kosakata keluhan yang jarang muncul.

---

## 4. DATA UNDERSTANDING  **[L]**
### 4.1 Informasi Dataset
**Sumber Dataset:**  
[Sebutkan sumber: Kaggle – Shopee Fashion Reviews (Indonesian) - gold_anotasi_manual.csv]

**Deskripsi Dataset:**
- Jumlah baris (rows): [349 baris data ulasan yang telah dianotasi secara manual (gold standard)]
- Jumlah kolom (columns/features): [10 kolom (row_id, urut, rating, ulasan, gold_kualitas, gold_bahan, gold_ukuran, gold_desain, gold_warna, gold_kenyamanan).]
- Tipe data: [Text & Tabular]
- Ukuran dataset: [54.9 kB (untuk file gold_anotasi_manual.csv; total keseluruhan repositori mencapai 2.11 MB yang terdiri dari 4 file).]
- Format file: [CSV]

### 4.2 Deskripsi Fitur
Jelaskan setiap fitur/kolom yang ada dalam dataset.
**Contoh tabel:**
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| row_id | Integer |Nomor indeks/ID unik baris dari dataset induk (rentang 22 hingga 6148). | 22, 55, 65 |
| urut |  Integer | Nomor urut sampel anotasi manual (1 hingga 350). | 163, 158, 278 |
| rating | Float | Skor penilaian bintang yang diberikan pembeli pada skala 1.0 hingga 5.0. | 5.0, 4.0, 1.0 |
| ulasan | String (Text) | Teks ulasan mentah pembeli terkait produk fashion Shopee (mencakup atribut terstruktur bawaan seperti Warna, Desain, Kenyamanan, dan komentar bebas). | "Warna:maroon Konsistensi:kualitas nya ga main2 Kenyamanan:sangat nyaman..." |
| gold_kualitas | Categorical | Label sentimen hasil anotasi manual untuk aspek kualitas produk (positif, negatif, netral, atau kosong jika aspek tidak dibahas). | positif, negatif, null |
| gold_bahan | Categorical | Label sentimen hasil anotasi manual untuk aspek bahan/material produk. | positif, negatif, null |
| gold_ukuran | Categorical | Label sentimen hasil anotasi manual untuk aspek ukuran (sizing) produk. |positif, negatif, null |
| gold_desain | Categorical | Label sentimen hasil anotasi manual untuk aspek desain/model produk. | positif, negatif, null |
| gold_warna | Categorical | Label sentimen hasil anotasi manual untuk aspek warna produk. | positif, negatif, null |
| gold_kenyamanan | Categorical | Label sentimen hasil anotasi manual untuk aspek kenyamanan saat dipakai.| positif, negatif, null |


**[Buat tabel deskripsi fitur Anda di sini]**

### 4.3 Kondisi Data

Jelaskan kondisi dan permasalahan data:

- **Missing Values:** [Pada kolom fitur utama (row_id, urut, rating, ulasan) tidak ditemukan nilai kosong (0%). Namun pada enam kolom label aspek (gold_kualitas, gold_bahan, gold_ukuran, gold_desain, gold_warna, gold_kenyamanan), terdapat proporsi nilai kosong (null) yang tinggi berkisar antara 51.86% hingga 79.08%. Kondisi ini bukan kerusakan data, melainkan karakteristik alami Aspect-Based Sentiment Analysis karena tidak semua pembeli membahas keenam aspek sekaligus dalam satu ulasan (nilai null menandakan aspek tersebut tidak disebutkan dalam ulasan).
- **Duplicate Data:** [Ditemukan sebanyak 0 baris (0.00%) duplikasi teks pada kolom ulasan (dan 0 baris duplikat identik di seluruh kolom). Seluruh 349 baris data memiliki teks ulasan yang unik sehingga tidak memicu bias duplikasi pada pelatihan model.]
- **Outliers:** [Pada fitur numerik rating, nilai berada dalam rentang skala tetap 1.0 hingga 5.0 tanpa outlier numerik. Pada fitur teks ulasan, panjang ulasan berkisar antara 15 hingga 1030 karakter, dengan terdeteksi sebanyak 20 baris (5.73%) outlier panjang karakter berdasarkan metode IQR (ulasan yang sangat panjang karena merinci banyak atribut produk sekaligus).]
- **Imbalanced Data:** [Terjadi ketidakseimbangan kelas (class imbalance) yang signifikan baik pada fitur rating maupun pada setiap label aspek. Distribusi rating didominasi oleh skor tinggi (bintang 5.0), dan pada setiap kolom aspek (gold_*), label sentimen positif jauh lebih dominan dibandingkan negatif maupun netral.]
- **Noise:** [Ditemukan elemen noise khas ulasan e-commerce Indonesia, meliputi format teks terstruktur bawaan template Shopee (terdeteksi pada 174 baris atau 49.86% ulasan, seperti "Warna:maroon Desain:baguams..."), singkatan tidak baku ("ga main2", "standart"), salah ketik/typo ("baguams", "siller"), serta tanda baca berulang.]
- **Data Quality Issues:** [Selain tingginya nilai null pada aspek yang tidak dibahas, ditemukan variasi penulisan/kategori pada kolom label aspek di luar kelas utama (positif, negatif, netral) serta ditemukan 0 ulasan yang sangat pendek (< 5 karakter), sehingga diperlukan tahapan standarisasi dan pembersihan label (label cleaning / mapping menjadi positif, negatif, netral, dan tidak_disebutkan) sebelum masuk ke tahap pemodelan.]
- **Asumsi Data (i.i.d., distribusi train-test, potensi data leakage):** [Data diasumsikan bersifat independent and identically distributed (i.i.d.). Untuk mencegah potensi data leakage, pembagian data dilakukan menggunakan metode Stratified Train-Test Split (rasio 80:20) pada target aspek sehingga proporsi kelas pada train set (279 baris) dan test set (70 baris) tetap terjaga seimbang dengan irisan teks duplikat sebesar 0 baris, serta proses fitting ekstraksi fitur (TF-IDF Vectorizer) hanya dilakukan pada data latih (training set).]

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

**Requirement:** Minimal 3 visualisasi yang bermakna dan insight-nya.
**Contoh jenis visualisasi yang dapat digunakan:**
- Histogram (distribusi data)
- Boxplot (deteksi outliers)
- Heatmap korelasi (hubungan antar fitur)
- Bar plot (distribusi kategori)
- Scatter plot (hubungan 2 variabel)
- Wordcloud (untuk text data)
- Sample images (untuk image data)
- Time series plot (untuk temporal data)
- Confusion matrix heatmap
- Class distribution plot


#### Visualisasi 1: [Judul Visualisasi]
[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 2: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 3: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

---

## 5. DATA PREPARATION [L]

Bagian ini menjelaskan **semua** proses transformasi dan preprocessing data yang dilakukan pada dataset ulasan produk fesyen e-commerce (`jareedhnnn/shopee-fashion-reviews-indonesian`, berkas `dataset_labeled_v6.xlsx`) sebelum masuk ke tahapan pemodelan *Multi-Output Aspect-Based Sentiment Analysis* (ABSA) untuk 6 aspek sekaligus.

### 5.1 Data Cleaning

**Aktivitas:**
- Selecting main labeled dataset & dropping unused columns
- Handling missing values
- Removing duplicates
- Handling outliers & Noise

**Langkah-langkah data cleaning yang dilakukan:**
- **Dataset Selection & Dropping Unused Columns:** 
  - Repositori dataset Kaggle memiliki beberapa berkas terpisah (`DATASET_dengan_Kategori (1).xlsx` untuk kategori umum dan `gold_anotasi_manual.csv` yang hanya berisi 349 sampel anotasi parsial), sehingga dipilih berkas utama berlabel lengkap yakni **`dataset_labeled_v6.xlsx`** (6.158 baris).
  - **Strategi:** Mempertahankan hanya kolom teks ulasan (`ulasan`) dan 6 kolom target sentimen aspek (`aspek_kualitas`, `aspek_bahan`, `aspek_ukuran`, `aspek_desain`, `aspek_warna`, `aspek_kenyamanan`), serta membuang kolom lain yang tidak relevan.
  - **Alasan:** Memastikan proses pelatihan berfokus sepenuhnya pada hubungan teks ulasan terhadap keenam aspek produk fesyen tanpa membebani komputasi dengan fitur yang tidak digunakan.
- **Handling Missing Values:**
  - Memeriksa keberadaan nilai kosong (`NaN`) pada kolom teks ulasan maupun keenam kolom target aspek.
  - **Strategi:** Menghapus baris yang memiliki nilai kosong pada salah satu kolom teks atau target aspek menggunakan fungsi `dropna(subset=[text_col] + ASPECT_COLS)`.
  - **Alasan:** Pemodelan klasifikasi *multi-output* mewajibkan setiap sampel ulasan memiliki label lengkap pada seluruh 6 aspek agar perhitungan *loss* berjalan valid.
- **Removing Duplicates:**
  - Menghapus ulasan ganda yang identik pada kolom teks ulasan.
  - **Strategi:** Menghapus baris duplikat secara tegas menggunakan fungsi `drop_duplicates(subset=[text_col])` *sebelum* tahap pembagian data.
  - **Alasan:** Sangat krusial untuk mencegah model menghafal ulasan *spam* secara berulang dan menghindari kebocoran data (*data leakage*) antara *train set*, *validation set*, dan *test set*.
- **Handling Noise (Short Texts & Empty Strings):**
  - Menyaring ulasan yang terlalu singkat maupun yang menjadi string kosong setelah pembersihan simbol.
  - **Strategi:** Memfilter dan membuang baris yang panjang karakternya kurang dari 5 huruf (`char_length < 5`) serta teks kosong pasca-pembersihan.
  - **Alasan:** Komentar yang terlalu singkat umumnya tidak memuat informasi aspek maupun polaritas sentimen yang bermakna dan dapat memicu *noise* pada bobot kosakata.

### 5.2 Feature Engineering

**Aktivitas:**
- Multi-target aspect formulation (6 aspect sentiment columns)
- Feature extraction

**Langkah-langkah feature engineering yang dilakukan:**
- **Multi-Target Aspect Formulation (6 Aspek Produk Fesyen):** 
  - Berbeda dengan *rule-based pseudo-labeling*, dataset ini telah dilengkapi anotasi sentimen eksplisit (`positif`, `negatif`, `netral`) untuk **6 aspek spesifik**:
    - `aspek_kualitas`: 1.068 Positif | 180 Negatif | 4.910 Netral
    - `aspek_bahan`: 1.823 Positif | 175 Negatif | 4.160 Netral
    - `aspek_ukuran`: 571 Positif | 276 Negatif | 5.311 Netral
    - `aspek_desain`: 2.010 Positif | 63 Negatif | 4.085 Netral
    - `aspek_warna`: 2.316 Positif | 156 Negatif | 3.686 Netral
    - `aspek_kenyamanan`: 2.353 Positif | 78 Negatif | 3.727 Netral
  - Keenam kolom ini diformulasikan sebagai target *Multi-Output Classification* sehingga satu ulasan diprediksi polaritasnya pada seluruh aspek secara serentak.
- **Feature Extraction (TF-IDF Unigram & Bigram):** 
  - Mengekstrak teks ulasan menjadi matriks bobot numerik menggunakan `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))`. Penggunaan *unigram* dan *bigram* `(1, 2)` bertujuan menangkap frasa negasi dua kata (seperti *"tidak luntur"*, *"tidak panas"*, atau *"kurang rapi"*) sebagai *input* utama bagi Baseline Logistic Regression dan LightGBM (`ClassifierChain`).
- **Feature Extraction (Contextual Subword Embedding):** 
  - Khusus untuk arsitektur Deep Learning (**Multi-Head IndoBERT** `indobert-base-p1`), ekstraksi fitur menggunakan representasi *dense vector* kontekstual berdimensi 768. Tujuannya agar urutan kata dan struktur kalimat kontras antar-aspek (misalnya *"warna cantik tapi bahan tipis"*) dapat dipertahankan.

### 5.3 Data Transformation

**Untuk Data Text:**
- **Lowercasing:** Mengonversi seluruh teks ulasan menjadi huruf kecil agar variasi kapitalisasi (seperti "Bagus", "BAGUS", "bagus") dianggap sebagai satu entitas token yang identik.
- **Removing punctuation/numbers:** Menghapus angka, tanda baca, dan emoji menggunakan Regular Expression (`re.sub(r'[^a-z\s]', ' ', teks)`) serta menormalisasi spasi berlebih untuk mengurangi variasi *noise* pada token kata.
- **Stopwords & Stemming (Terkendali):** Disediakan melalui pustaka `Sastrawi`, namun pada konfigurasi utama dipertahankan tanpa penghapusan *stopword* maupun *stemming* agresif agar kata penentu negasi dan polaritas (seperti *"tidak"*, *"nggak"*, *"kurang"*, *"bukan"*) tetap terjaga utuh.
- **Tokenization & Padding sequences:** Khusus untuk persiapan data masuk ke model Deep Learning (IndoBERT / BiLSTM), untaian kalimat diubah menjadi token indeks numerik (`input_ids`) beserta `attention_mask` dan diseragamkan panjang maksimumnya (*padding*).

**Untuk Data Tabular:**
- **Encoding (Uniform Target Mapping):** Mengubah nilai kategori pada seluruh 6 kolom target aspek (`positif`, `negatif`, `netral`) menjadi indeks bilangan bulat seragam (`negatif -> 0`, `netral -> 1`, `positif -> 2`) ke dalam matriks target berdimensi $(N \times 6)$. Transformasi ini wajib dilakukan agar fungsi *loss* pada algoritma *Machine Learning* dan *Deep Learning* dapat memproses target secara konsisten antar-aspek.

### 5.4 Data Splitting

**Strategi pembagian data:**
Menggunakan metode *Stratified Split* dengan rasio pembagian berskala **80:10:10** dari total **± 6.158 sampel** bersih:
- **Training set:** 80% (± 4.926 samples)
- **Validation set:** 10% (± 616 samples)
- **Test set:** 10% (616 samples)
- **Random state:** `42` untuk reproducibility

**Alasan strategi splitting:**
- Metode **Stratified Split** diaplikasikan menggunakan stratifikasi gabungan multi-aspek dengan mekanisme *fallback* berdasarkan aspek kelas minoritas terkecil (`aspek_desain_encoded`, yang hanya memiliki 63 sampel negatif atau ± 1,0%). Strategi ini memaksa keterwakilan proporsional kelas minoritas tetap terjaga di partisi Latih, Validasi, dan Uji agar evaluasi model tidak bias.
- **Pencegahan Data Leakage (Sangat Penting):** Inisiasi `.fit()` pada algoritma `TfidfVectorizer` **secara ketat hanya diterapkan pada Training Set (`X_train`)**. Validation set (`X_val`) dan Test set (`X_test`) murni hanya melalui tahap `.transform()`. Hal ini menjamin bahwa model tidak pernah melihat satupun kosakata (*vocabulary*) maupun bobot IDF yang bersumber dari data validasi maupun data uji.

### 5.5 Data Balancing (jika diperlukan)

**Teknik yang digunakan:**
- Class weights (`class_weight='balanced'`) & Weighted Cross-Entropy Loss

**Penjelasan penggunaan data balancing:**
Berdasarkan hasil eksplorasi distribusi pada ke-6 aspek, data mengalami **ketimpangan kelas yang ekstrem (*severe class imbalance*)**, di mana kelas `Netral` mendominasi **59,9% hingga 86,2%** data pada setiap aspek, sedangkan kelas `Negatif` hanya berkisar antara **1,0% hingga 4,5%** (seperti pada `aspek_desain` dan `aspek_kenyamanan`). Oleh karena itu, pendekatan *oversampling* sintetik menggunakan algoritma seperti SMOTE sengaja tidak dilakukan karena pembuatan sampel vektor TF-IDF sintetik secara bersamaan pada **6 target aspek** berisiko merusak korelasi semantik antar-aspek dan memicu *overfitting*. Sebagai kompensasi atas kelas minoritas, penyeimbangan dilakukan secara internal dari dalam arsitektur algoritma melalui aktivasi parameter `class_weight='balanced'` pada model Logistic Regression dan LightGBM, serta penerapan *Weighted Cross-Entropy Loss* pada model Multi-Head IndoBERT.

### 5.6 Ringkasan Data Preparation

**Per langkah, jelaskan:**
1. **Apa yang dilakukan**
   Memilih berkas dataset utama berlabel 6 aspek (`dataset_labeled_v6.xlsx`), menghapus nilai kosong (`NaN`), membuang baris duplikat ulasan serta teks *noise* (< 5 karakter), memformulasikan 6 kolom target aspek produk fesyen (`kualitas`, `bahan`, `ukuran`, `desain`, `warna`, `kenyamanan`), melakukan pra-pemrosesan teks (*lowercasing* dan pembersihan simbol/angka via Regex dengan mempertahankan kata negasi), mengonversi label target ke format numerik seragam (`0: negatif`, `1: netral`, `2: positif`), membelah dataset melalui metode *Stratified Split* (80:10:10), mengekstrak teks ke matriks matematis dengan TF-IDF *unigram-bigram* (khusus di-*fit* pada *training set*), serta menerapkan pembobotan kelas (`class_weight='balanced'`).
2. **Mengapa penting**
   Pembersihan duplikasi ganda sebelum pembagian data sangat krusial agar tidak terjadi kebocoran data (*data leakage*) ke fase evaluasi (*test set*). Penggunaan 6 target aspek eksplisit menggantikan *pseudo-labeling* manual sehingga analisis sentimen berbasis aspek lebih akurat dan spesifik pada domain fesyen. Pembersihan simbol/angka mengurangi *noise* kosakata, sementara pembagian terstratifikasi dan pembobotan kelas mencegah model menjadi bias yang hanya menebak kelas mayoritas (`Netral`) dan mengabaikan keluhan pelanggan (`Negatif`).
3. **Bagaimana implementasinya**
   Pemuatan berkas dari Kaggle dan prosedur pembersihan data dikerjakan menggunakan `kagglehub` dan *method* bawaan `pandas` (`dropna`, `drop_duplicates`, dan `drop`). Manipulasi *string* dilakukan melalui modul `re` (Regular Expression) bersama opsi pustaka NLP lokal `Sastrawi`. Pengkodean target dikerjakan via pemetaan kamus *dictionary* pada `pandas`, strategi *splitting* diinstruksikan lewat fungsi `train_test_split` dari `scikit-learn` dengan pengunci `random_state=42` dan `stratify`, sedangkan representasi fitur dituntaskan menggunakan `TfidfVectorizer` (`scikit-learn`) dan *tokenizer* `transformers` untuk IndoBERT.

---

## 6. MODELING  **[M + O]**

> Setiap model (6.1, 6.2, 6.3) dijelaskan dalam dua fase: **Model Design [M]** (sub-bagian .1–.3) dan **Optimization [O]** (sub-bagian .4–.6).

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model  **[M]**

**Nama Model:** Multi-Output Logistic Regression (`MultiOutputClassifier` + `LogisticRegression`)  
**Teori Singkat:**  
Logistic Regression merupakan model klasifikasi linear probabilistik yang menghitung kombinasi linear antara vektor fitur masukan (bobot TF-IDF) dengan matriks parameter bobot ($W$), lalu memetakannya ke dalam distribusi probabilitas tiap kelas sentimen (`0: Negatif`, `1: Netral`, `2: Positif`) menggunakan fungsi aktivasi *Softmax* (multinomial). Karena permasalahan memiliki 6 target aspek sekaligus, algoritma dibungkus ke dalam arsitektur `MultiOutputClassifier` yang melatih 6 estimator Logistic Regression independen secara paralel—masing-masing bertanggung jawab memprediksi polaritas sentimen pada satu aspek spesifik.

**Alasan Pemilihan:**  
Model ini dipilih sebagai *baseline* karena arsitekturnya sederhana, sangat efisien secara komputasi (waktu pelatihan hanya **10,53 detik** pada 4.926 sampel latih), bersifat deterministik, serta memiliki interpretabilitas tinggi melalui nilai koefisien bobot kata untuk memvalidasi kata kunci indikator sentimen pada setiap aspek produk fesyen.

#### 6.1.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** Generalized Linear Model (Model Klasifikasi Linear Multikelas *Multi-Output*).
- **Hypothesis space:** Himpunan fungsi pemisah berupa *hyperplane* linear dalam ruang vektor fitur TF-IDF berdimensi $d = 5.000$ untuk masing-masing dari 6 aspek target secara terpisah.
- **Inductive bias / asumsi:** 
  1. Hubungan antara kemunculan kata (bobot TF-IDF) dan log-odds kelas sentimen bersifat aditif dan linear.
  2. Fitur kata bersifat independen secara kondisional (*bag-of-words* unigram) tanpa memperhitungkan urutan kata maupun interaksi non-linear antar-frasa.
  3. Keenam target aspek diprediksi secara independen (*conditional independence* antar-target) tanpa berbagi sinyal prediksi lintas aspek.
- **Implikasi terhadap dataset Anda:** Asumsi linear ini bekerja dengan baik pada aspek yang memiliki kata kunci eksplisit dan konsisten, seperti `aspek_ukuran` (*"kekecilan"*, *"kebesaran"*, *"pas"*) dan `aspek_bahan` (*"adem"*, *"lembut"*, *"tipis"*). Namun, keterbatasan representasi *unigram* linear membuat model kesulitan memisahkan polaritas pada ulasan majemuk yang mengandung klausa kontras (misal *"desain kece tapi bahan tipis"*), sehingga kata milik aspek lain seperti `tipis` ikut terbaca sebagai indikator negatif pada `aspek_desain`.

#### 6.1.3 Implementasi (Ringkas)  **[M]**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# 1. Ekstraksi fitur TF-IDF dasar (fit hanya pada training set)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# 2. Inisialisasi & pelatihan Multi-Output Logistic Regression
base_lr = LogisticRegression(
    class_weight="balanced",
    C=1.0,
    max_iter=1000,
    random_state=42,
)
baseline_model = MultiOutputClassifier(base_lr, n_jobs=-1)
baseline_model.fit(X_train_tfidf, Y_train)
Y_val_pred = baseline_model.predict(X_val_tfidf)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi  **[O]**

**Objective/loss yang dioptimalkan:**  
*Weighted Multinomial Cross-Entropy Loss* (Log-Loss) yang dilengkapi dengan penalti regularisasi **L2 (Ridge)** untuk mengendalikan besaran bobot dan mencegah *overfitting* pada matriks fitur TF-IDF yang berdimensi tinggi dan jarang (*sparse*):
$$\mathcal{L}(W) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{2} w_c \, y_{i,c} \log(\hat{p}_{i,c}) + \frac{1}{2C} \Vert{}W\Vert{}_2^2$$
dengan $w_c = \frac{N}{3 \cdot N_c}$ sebagai faktor pembobot kelas otomatis (`class_weight='balanced'`) agar kesalahan klasifikasi pada kelas minoritas (`Negatif`) memperoleh penalti yang seimbang terhadap kelas mayoritas (`Netral`).

**Parameter yang digunakan:**
```text
- max_features (TF-IDF)     : 5000 (unigram)
- C (inverse regularization): 1.0 (regularisasi L2 standar)
- penalty                   : 'l2'
- solver                    : 'lbfgs'
- class_weight              : 'balanced'
- max_iter                  : 1000
- random_state              : 42
```

#### 6.1.5 Hasil Awal  **[O → P]**

Evaluasi awal pada **Data Validation (616 sampel)** menghasilkan **Overall Accuracy sebesar 0.9332**, **Weighted F1 sebesar 0.9358**, dan **Overall Macro F1 sebesar 0.7666**:

| Aspek | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | F1 Negatif (0) | F1 Netral (1) | F1 Positif (2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`aspek_kualitas`** | 0.8994 | 0.7309 | 0.8009 | 0.7608 | 0.9032 | 0.5385 | 0.9423 | 0.8018 |
| **`aspek_bahan`** | 0.9156 | 0.7960 | 0.8377 | 0.8150 | 0.9170 | 0.6286 | 0.9502 | 0.8663 |
| **`aspek_ukuran`** | 0.9578 | 0.8214 | 0.8716 | 0.8418 | 0.9590 | 0.7059 | 0.9848 | 0.8346 |
| **`aspek_desain`** | 0.9594 | 0.7142 | 0.7522 | 0.7276 | 0.9617 | 0.2500 | 0.9731 | 0.9598 |
| **`aspek_warna`** | 0.9464 | 0.7641 | 0.8711 | 0.7973 | 0.9505 | 0.4848 | 0.9728 | 0.9343 |
| **`aspek_kenyamanan`** | 0.9205 | 0.6549 | 0.6683 | 0.6574 | 0.9233 | 0.1176 | 0.9456 | 0.9091 |
| **RATA-RATA (OVERALL)** | **0.9332** | **0.7469** | **0.8003** | **0.7666** | **0.9358** | **0.4542** | **0.9615** | **0.8843** |

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model  **[M]**

**Nama Model:** [Tuned LightGBM (LGBMClassifier) dengan ClassifierChain + TF-IDF Unigram–Bigram (1, 2)]  
**Teori Singkat:**  
[LightGBM (Light Gradient Boosting Machine) membangun ensemble pohon keputusan secara boosting (memperbaiki residual/gradient kesalahan prediksi secara bertahap) dengan strategi pertumbuhan leaf-wise (memecah daun dengan gain terbesar) dan histogram-based splitting. Melalui pendekatan rantai klasifikasi (ClassifierChain), prediksi pada aspek sebelumnya dimanfaatkan sebagai informasi tambahan untuk memprediksi aspek berikutnya sehingga korelasi antar-aspek produk fashion dapat diperhitungkan.]

**Alasan Pemilihan:**  
[Cepat diproses dan memakai fitur TF-IDF unigram–bigram (1, 2) sehingga frasa dua kata seperti "tidak luntur" atau "nggak kekecilan" bisa dikenali, sekaligus menangkap pola non-linear.]

**Keunggulan:**
- [Cepat dilatih dan efisien untuk data berdimensi besar.]
- [Menangkap hubungan non-linear serta frasa dua kata (bigram).]
- [Bisa memanfaatkan kaitan antar-aspek lewat ClassifierChain dan mengatur min_child_samples untuk kelas minoritas.]

**Kelemahan:**
- [Cenderung mudah overfitting pada data latih bila kedalaman/jumlah daun dan learning rate tidak dikontrol dengan baik.]
- [Urutan rantai aspek dapat memengaruhi hasil prediksi aspek berikutnya.]
- [Membutuhkan tuning hiperparameter agar performa kelas minoritas (Negatif) tetap optimal.]

#### 6.2.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [Gradient Boosted Decision Trees (leaf-wise) dengan skema rantai (ClassifierChain) untuk 6 aspek.]
- **Hypothesis space:** [Gabungan aturan keputusan bertingkat (piecewise) dari banyak pohon pada fitur TF-IDF (1, 2) dan prediksi aspek sebelumnya.]
- **Inductive bias / asumsi:** [Kesalahan dapat ditekan bertahap lewat aturan ambang (split) fitur, frasa dua kata memberi makna lebih jelas dibanding satu kata saja, dan urutan aspek membawa informasi tambahan.]
- **Perbandingan dengan baseline:** [Lebih fleksibel dari Logistic Regression unigram independen karena mampu membaca frasa dua kata, pola non-linear, dan hubungan antar-aspek.]

#### 6.2.3 Implementasi (Ringkas)  **[M]**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

# 1. Ekstraksi fitur TF-IDF unigram-bigram (fit hanya pada training set)
tfidf_lgb = TfidfVectorizer(ngram_range=(1, 2), max_features=10000) 
X_train_lgb = tfidf_lgb.fit_transform(X_train) 
X_val_lgb = tfidf_lgb.transform(X_val) 

# 2. Inisialisasi model dasar LightGBM
base_lgb = LGBMClassifier( 
    n_estimators=200, 
    learning_rate=0.05, 
    min_child_samples=3, 
    class_weight="balanced", 
    random_state=42, 
    n_jobs=-1, 
    verbose=-1, 
)

# 3. Bungkus ke ClassifierChain & latih pada data train
advanced_model = ClassifierChain(base_lgb, order="random", random_state=42) 
advanced_model.fit(X_train_lgb, Y_train) 
Y_val_pred_lgb = advanced_model.predict(X_val_lgb) 
```

#### 6.2.4 Optimization: Hyperparameter Tuning  **[O]**

**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- n_estimators: 200 (model awal) / 200 (hasil tuning)
- max_depth: 0.05 (model awal) / 0.1 (hasil tuning)
- num_leaves: 15 (hasil tuning)
- min_child_samples: 3 (model awal) / 5 (hasil tuning)
- class_weight: 'balanced'
- random_state: 42
```

**Hyperparameter Tuning (WAJIB minimal 1 metode sederhana):**
- Metode: [RandomizedSearchCV (n_iter=4, scoring='f1_macro')]
- Search space: [n_estimators: [100, 200], learning_rate: [0.05, 0.1], num_leaves: [15, 31], min_child_samples: [3, 5]]
- Skema validasi: [cv=2 saat search, lalu dievaluasi pada Validation Set (616 sampel)]
- Best parameters: [num_leaves=15, n_estimators=200, min_child_samples=5, learning_rate=0.1]
- Skor validasi sebelum vs sesudah tuning:[Overall Macro F1: 0.7929 → 0.7945, Overall Accuracy: 0.9629 → 0.9619, Weighted F1: 0.9614 → 0.9603]

#### 6.2.5 Kontrol Overfitting  **[O]**
[Overfitting ditekan dengan membatasi jumlah daun pohon (num_leaves=15 hasil tuning), mengatur jumlah minimal sampel di tiap daun (min_child_samples=5 hasil tuning / 3 awal), memakai learning_rate yang terkendali (0.1 / 0.05), serta melakukan fit TF-IDF (max_features=10000, ngram_range=(1, 2)) hanya pada data latih (4.926 sampel).]

#### 6.2.6 Hasil Model  **[O → P]**
Pada Validation Set (616 sampel), perbandingan hasil Sebelum Tuning (waktu latih 70,77 detik) dan Sesudah Tuning (total waktu tuning & evaluasi 955,83 detik) adalah sebagai berikut:
| Metrik | Sebelum Tuning | Sesudah Tuning |
| :--- | :---: | :---: |
| **`Overall Accuracy`** | `0.9629` | `0.9619` |
| **`Overall Macro F1`** | `0.7929` | `0.7945` |
| **`Weighted F1`** | `0.9614` | `0.9603` |
| **`F1 Negatif (0)`** | `0.4739` | `0.4795` |
| **`F1 Netral (1)`** | `0.9826` | `0.9816` |
| **`F1 Positif (2)`** | `0.9223` | `0.9224` |

**[Tuliskan hasil pada validation set; evaluasi akhir dijelaskan di Bab 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model  **[M]**

**Nama Model:** [Multi-Head IndoBERT (indobenchmark/indobert-base-p1)]

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [x] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Model NLP berbasis Transformer (indobert-base-p1) memahami konteks kata berdasarkan perhatian (self-attention) pada seluruh kalimat bahasa Indonesia, sehingga lebih akurat membaca frasa sehari-hari, negasi, maupun sentimen campuran antar-aspek pada ulasan fashion.]

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias  **[M]**

**Inductive bias arsitektur:** [IndoBERT (12 lapisan Transformer encoder, 12 attention heads, ukuran tersembunyi 768) menangkap hubungan kontekstual antar-token dalam kalimat bahasa Indonesia, lalu dirangkum melalui attention-masked mean pooling dan diproyeksikan ke 6 classification head (Linear(768, 3) per aspek).]  
**Hypothesis space:** [Ruang representasi deep transformer berlapis yang memetakan urutan token ulasan ke logit 3 kelas (Negatif, Netral, Positif) untuk masing-masing dari 6 aspek.]

**Deskripsi Layer:**
| Urutan | Layer / Komponen | Konfigurasi / Output Shape | Keterangan |
| :--- | :--- | :--- | :--- |
| `1` | **Input Layer** | `(batch_size, 128)` | `input_ids` dan `attention_mask` (`max_len=128`) |
| `2` | **Backbone Teks** | `BertModel` (`indobert-base-p1`, hidden `768`, 12 layer) | Ekstraksi representasi kontekstual bahasa Indonesia |
| `3` | **Pooling** | *Attention-masked Mean Pooling* `(batch_size, 768)` | Merangkum vektor token aktif menjadi 1 vektor kalimat |
| `4` | **Dropout** | `Dropout(p=0.3)` | Mengurangi *overfitting* sebelum lapisan klasifikasi |
| `5` | **Output Head (6 Aspek)** | `ModuleList` (`6 x Linear(768, 3)`) → `(batch_size, 6, 3)` | Logit 3 kelas sentimen untuk 6 aspek |

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [124,455,186]
Trainable parameters: [124,455,186]
```

#### 6.3.3 Input & Preprocessing Khusus  **[M]**

**Input shape:** [(N, 128) untuk input_ids dan attention_mask (diseragamkan dengan padding dan truncation hingga max_len=128).]  
**Preprocessing khusus untuk DL:**
- [Tokenisasi teks menggunakan AutoTokenizer dari indobenchmark/indobert-base-p1 pada 4.926 data latih dan 616 data validasi, dengan label kelas 0, 1, 2 pada 6 aspek.]

#### 6.3.4 Optimization: Konfigurasi Training  **[O]**

**Training Configuration:**
```
- Optimizer: AdamW (weight_decay=0.01)
- Learning rate: [2e-5 (Exp-1) dan 3e-5 (Exp-2) dengan linear warmup (10% langkah) + linear decay]
- Loss function: [Rata-rata Cross-Entropy Loss (nn.CrossEntropyLoss()) di atas 6 aspek]
- Metrics: [accuracy, macro_f1]
- Batch size: [16 (Exp-1) dan 32 (Exp-2)]
- Epochs: [3]
- Validation split: Validation Set (616 sampel)
- Callbacks: [Penyimpanan best checkpoint berdasarkan Val Macro F1 tertinggi]
- Regularisasi: [Dropout(p=0.3), weight_decay=0.01, dan gradient clipping (max_norm=1.0)]
```

**Eksperimen optimasi (minimal 2 konfigurasi dibandingkan):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Metric |
|------------|---------------|------------|--------------|----------|------------|
| Exp-1 | [ 2e-5 ] | [16] | [Dropout(0.3), weight_decay=0.01, grad clip 1.0] | [0.0942] | [Macro F1: 0.8347 | Acc: 0.9713] |
| Exp-2 | [ 3e-5 ] | [32] | [Dropout(0.3), weight_decay=0.01, grad clip 1.0] | [0.0966] | [Macro F1: 0.8418 | Acc: 0.9697] |

#### 6.3.5 Implementasi (Ringkas)  **[M + O]**

**Framework:** TensorFlow/Keras / PyTorch
```python
import torch
import torch.nn as nn
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Definisi Multi-Head IndoBERT untuk 6 aspek x 3 kelas sentimen
class MultiHeadIndoBERT(nn.Module):
    def __init__(self, model_name="indobenchmark/indobert-base-p1", num_aspects=6, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        dropped = self.dropout(pooled)
        return torch.stack([head(dropped) for head in self.heads], dim=1)

# 2. Pindahkan ke GPU/CPU & latih per epoch dengan AdamW + CrossEntropyLoss
model = MultiHeadIndoBERT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

#### 6.3.6 Training Process & Diagnosis Optimasi  **[O]**

**Training Time:**  
[347,59 detik (Exp-1, 3 epoch) dan 313,91 detik (Exp-2, 3 epoch)]

**Computational Resource:**  
[GPU (Tesla T4), Google Colab]

**Training History Visualization:**

[Exp-1 (lr=2e-5, batch=16): Epoch 1 (Train Loss: 0.3246, Val Loss: 0.1285, Val Acc: 0.9564, Val Macro F1: 0.7696) → Epoch 2 (Train Loss: 0.1026, Val Loss: 0.1076, Val Acc: 0.9659, Val Macro F1: 0.7968) → Epoch 3 (Train Loss: 0.0624, Val Loss: 0.0942, Val Acc: 0.9713, Val Macro F1: 0.8347).]

[Exp-2 (lr=3e-5, batch=32): Epoch 1 (Train Loss: 0.3339, Val Loss: 0.1578, Val Acc: 0.9478, Val Macro F1: 0.7769) → Epoch 2 (Train Loss: 0.1067, Val Loss: 0.1047, Val Acc: 0.9640, Val Macro F1: 0.8233) → Epoch 3 (Train Loss: 0.0635, Val Loss: 0.0966, Val Acc: 0.9697, Val Macro F1: 0.8418).]
**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Tidak terlihat overfitting yang parah hingga epoch ke-3 karena Val Loss terus turun dari epoch 1 ke epoch 3 (0.1285 → 0.0942 pada Exp-1 dan 0.1578 → 0.0966 pada Exp-2) seiring Val Macro F1 yang terus meningkat.]
- Apakah model sudah converge? [Mendekati konvergen pada epoch ke-3, meskipun kenaikan Val Macro F1 masih aktif hingga epoch terakhir sehingga penambahan epoch sedikit (dengan pemantauan validation loss) masih berpotensi memberi perbaikan kecil.]
- Apakah perlu lebih banyak epoch? [Opsional/tidak harus banyak—3 epoch sudah memberi lonjakan performa besar (0.8347 dan 0.8418), dan jika ditambah epoch sebaiknya dipantau agar Val Loss tidak mulai naik.]
- Apakah learning rate sesuai (loss stabil/osilasi/lambat)? [Sesuai—penurunan loss berjalan mulus dan stabil tanpa osilasi ekstrem pada 2e-5 (Exp-1) maupun 3e-5 (Exp-2).]

#### 6.3.7 Model Summary
```
[Total parameters    : 124,455,186
Trainable parameters: 124,455,186
MultiHeadIndoBERT(
  (encoder): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(50000, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    ...
  )
  (dropout): Dropout(p=0.3, inplace=False)
  (heads): ModuleList(
    (0-5): 6 x Linear(in_features=768, out_features=3, bias=True)
  )
)]
```
---

## 7. EVALUATION  **[P]**

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**Justifikasi pemilihan metrik:** [Mengapa metrik ini selaras dengan Objective di Bab 3.2? Mengapa accuracy saja cukup/tidak cukup?]

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Semua prediksi dievaluasi pada **test set** yang tidak pernah dipakai saat training maupun tuning.

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Tampilkan minimal 5 contoh prediksi benar dan salah]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting & Generalization Gap:**  
   [Bandingkan performa train vs validation vs test; analisis apakah model overfitting atau underfitting]

6. **Kesesuaian Inductive Bias:**  
   [Apakah asumsi tiap model (Bab 6.x.2) terbukti sesuai dengan data? Jelaskan dengan bukti hasil]

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

Jelaskan minimal **satu** iterasi perbaikan berdasarkan hasil evaluasi:

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
|--------------------------|-----------------------|----------|--------------------|
| [misal: recall kelas minoritas rendah] | [L] data | [misal: class weight / SMOTE] | [ ... ] |
| [misal: gap train-val besar] | [O] optimasi | [misal: tambah dropout, early stopping] | [ ... ] |
| [misal: model terlalu sederhana] | [M] desain | [misal: tambah layer / ganti arsitektur] | [ ... ] |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.3 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
|-------|-----------------|-----------|
| **L** – Learning Problem | [ ... ] | [ ... ] |
| **M** – Model Design | [ ... ] | [ ... ] |
| **O** – Optimization | [ ... ] | [ ... ] |
| **P** – Predict & Evaluate | [ ... ] | [ ... ] |

### 8.4 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Kontribusi terhadap SDGs:**  
[Jelaskan secara jujur dan terukur bagaimana hasil proyek mendukung target SDG yang dipilih (Bab 2.2), serta keterbatasan kontribusinya]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

**Saran struktur folder (mengikuti lifecycle):**
```
project/
├── data/                 # [L] dataset (jangan upload jika besar)
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb      # [L]
│   ├── 02_data_preparation.ipynb          # [L]
│   ├── 03_model_design_baseline_ml.ipynb  # [M]
│   ├── 04_deep_learning_optimization.ipynb# [M+O]
│   └── 05_evaluation_comparison.ipynb     # [P]
├── src/                  # script modular
├── models/               # model tersimpan
├── reports/figures/
├── requirements.txt
└── README.md
```

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll
```

### 10.3 Reproducibility Checklist
- [ ] Random seed ditetapkan (misal 42) di semua library
- [ ] Notebook dapat dijalankan ulang dari awal sampai akhir
- [ ] Test set tidak dipakai saat training/tuning
- [ ] Versi library tercatat

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

Pengerjaan mengikuti urutan lifecycle L → M → O → P dengan satu iterasi perbaikan. Rincian tugas per minggu ada di file panduan terpisah: **Panduan_Tahapan_6_Minggu.md**.

| Minggu | Fokus Lifecycle | Aktivitas | Deliverable / Checkpoint |
|--------|-----------------|-----------|--------------------------|
| **1** | **L** – Learning Problem | Mahasiswa mengajukan ide sendiri selaras dengan satu SDG, pilih dataset, studi literatur, rumuskan problem statement, target, objective, data (Bab 1–3) | Formulir ide + proposal singkat 1–2 halaman; ide & dataset disetujui dosen. **Checkpoint 1** |
| **2** | **L** – Data | Data understanding, EDA (opsional), cleaning, transformasi, splitting (Bab 4–5) | Notebook data preparation + repo GitHub dibuat |
| **3** | **M** – Model Design | Rancang & latih Model 1 (baseline) dan Model 2 (advanced); jelaskan hypothesis space & inductive bias (Bab 6.1–6.2) | Hasil baseline & advanced pada validation set. **Checkpoint 2** |
| **4** | **M → O** | Tuning Model 2; desain arsitektur Model 3 (deep learning) dan training pertama (Bab 6.2.4, 6.3.1–6.3.5) | Model DL berjalan end-to-end (training awal) |
| **5** | **O** – Optimization | Optimasi Model 3: eksperimen learning rate/batch size/regularisasi, plot loss/metric, diagnosis konvergensi (Bab 6.3.4–6.3.7) | Model DL terlatih (≥10 epoch) + tabel eksperimen. **Checkpoint 3** |
| **6** | **P** – Predict & Evaluate | Evaluasi test set, perbandingan 3 model, error analysis, iterasi perbaikan, kesimpulan, video, finalisasi repo (Bab 7–10) | Laporan final + GitHub + video + presentasi |

**Catatan untuk dosen:**
- Enam minggu memberi ruang tambahan untuk tahap Optimization, tempat mahasiswa biasanya paling banyak mengalami kendala (training deep learning, tuning, overfitting).
- Tiga checkpoint (akhir minggu 1, 3, dan 5) membantu mendeteksi keterlambatan lebih awal.

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|----------|-------|-------|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2) | L | 10% |
| Problem formulation (target, objective, data), latar belakang & referensi | L | 10% |
| Data understanding & data preparation (tanpa data leakage) | L | 10% |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias | M | 15% |
| Model deep learning: arsitektur, training ≥10 epoch, plot loss/metric | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P | 20% |
| Reproducibility: GitHub, README, requirements, struktur folder | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| **Total** | | **100%** |

> Bobot dapat disesuaikan dengan kebijakan mata kuliah.

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

Referensi resmi: [Transforming our world: the 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda)

**Aturan ide:**
- Ide **harus berasal dari mahasiswa**, bukan dari daftar contoh di bawah. Contoh hanya inspirasi.
- Pilih **satu** SDG utama dan sebutkan target (nomor target) yang relevan.
- Masalah harus nyata, spesifik, dan dapat dirumuskan sebagai tugas ML (klasifikasi, regresi, forecasting, dll.).
- Dataset harus tersedia (publik atau dikumpulkan sendiri secara etis) dan cukup untuk melatih tiga model, termasuk deep learning.
- Disarankan setiap mahasiswa/kelompok memiliki ide yang berbeda; dosen memeriksa di Checkpoint 1.

| SDG | Nama (Indonesia) | Contoh arah ide ML (inspirasi) |
|-----|------------------|--------------------------------|
| 1 | Tanpa Kemiskinan | Prediksi tingkat kemiskinan wilayah dari data sosial-ekonomi (tabular) |
| 2 | Tanpa Kelaparan | Klasifikasi penyakit daun tanaman (image); prediksi hasil panen (tabular) |
| 3 | Kehidupan Sehat dan Sejahtera | Deteksi penyakit dari data klinis atau citra medis; prediksi risiko diabetes |
| 4 | Pendidikan Berkualitas | Prediksi risiko putus studi/performa siswa; analisis ulasan pembelajaran (teks) |
| 5 | Kesetaraan Gender | Deteksi ujaran kebencian atau bias gender pada teks |
| 6 | Air Bersih dan Sanitasi Layak | Prediksi kelayakan/kualitas air minum (tabular) |
| 7 | Energi Bersih dan Terjangkau | Forecasting konsumsi atau produksi energi (time series) |
| 8 | Pekerjaan Layak dan Pertumbuhan Ekonomi | Klasifikasi lowongan kerja atau prediksi pengangguran |
| 9 | Industri, Inovasi, dan Infrastruktur | Deteksi kerusakan jalan (image); predictive maintenance |
| 10 | Berkurangnya Kesenjangan | Pengenalan bahasa isyarat (image); analisis akses layanan publik |
| 11 | Kota dan Permukiman yang Berkelanjutan | Prediksi kualitas udara kota; prediksi kepadatan lalu lintas |
| 12 | Konsumsi dan Produksi yang Bertanggung Jawab | Klasifikasi jenis sampah (image); prediksi limbah makanan |
| 13 | Penanganan Perubahan Iklim | Prediksi curah hujan/suhu; deteksi titik api atau kebakaran hutan |
| 14 | Ekosistem Lautan | Klasifikasi spesies ikan; deteksi sampah laut atau kesehatan terumbu karang |
| 15 | Ekosistem Daratan | Klasifikasi spesies flora/fauna; deteksi deforestasi dari citra |
| 16 | Perdamaian, Keadilan, dan Kelembagaan yang Tangguh | Deteksi hoaks/berita palsu (teks); deteksi fraud |
| 17 | Kemitraan untuk Mencapai Tujuan | Sistem rekomendasi kolaborasi/mitra; analisis data terbuka pemerintah |

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | [ ... ] |
| Judul ide | [ ... ] |
| SDG & target yang dipilih | [ ... ] |
| Masalah nyata yang diangkat | [ ... ] |
| Tugas ML (Target & jenis tugas) | [ ... ] |
| Objective & metrik utama | [ ... ] |
| Dataset (sumber, URL, ukuran, jenis data) | [ ... ] |
| Rencana tiga model (baseline / advanced / deep learning) | [ ... ] |
| Risiko & pertimbangan etika (bias, privasi) | [ ... ] |
| Referensi awal (minimal 2) | [ ... ] |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
