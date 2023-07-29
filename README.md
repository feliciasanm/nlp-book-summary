# Proyek Book Summary Similarity _with_ NLP

Hi, _repository_ ini memuat _**capstone project**_ yang kukerjakan untuk kelas **_Certified Senior Data Scientist_ Narasio Data** (Maret 2022), yakni **membuat pengelompokkan buku-buku yang dianggap mirip berdasarkan rangkuman bukunya, menggunakan NLP**. Dalam prosesnya, digunakan beberapa teknik menarik seperti **Doc2Vec, NER, dan K-Means**. Cukup _excited_ dengan proyek ini karena merasa menggunakan teknik yang _adventurous_ kali ini!

**Berikut ini link untuk [PPT cantik hasil presentasi _capstone_](https://drive.google.com/file/d/1CHHD8UVev5pfxwq21x-i34LvCXhaXwF4/view?usp=sharing) (sama dengan PDF _slide_ yang ada di _repository_!)** ❤️

_**Note:** Proyek sudah selesai sampai tahap modelling (bagian NLP dan K-Means selesai), namun bagian yang mengevaluasi isi cluster satu per satu belum dipublikasi karena ingin ada _reworking__ 🚧

## Problem Setting
**Seorang pemilik toko buku** ingin mengadakan **pameran buku berskala besar**, dan dia memerlukan bantuan untuk **menyusun letak buku saat pameran** agar tepat dan buku-buku dengan **alur yang serupa** berada **di bagian yang sama**.

**Data yang diberikan** untuk tujuan tersebut memuat **beberapa metadata**, termasuk **summary bukunya**.

### Problem Translation
| Question    | Problem                                                                                 | How to Solve |
| ----------- | --------------------------------------------------------------------------------------- | ------------ |
| _What?_     | Event pameran buku yang segera diadakan seorang pemilik toko buku perlu disukseskan     | Untuk membantu mendukung acaranya, maka buku jualannya akan dianalisis. |
| _How Much?_ | Skala jumlah buku yang terlibat sangat banyak untuk dipersiapkan secara manual          | Agar dapat menangani skala buku yang banyak, digunakan _machine learning_ untuk membantu mengatur bukunya. |
| _How?_      | Topik buku-buku yang ada perlu diatur dengan cara yang menarik calon pengunjung/pembeli | Agar layout pameran menarik calon pengunjung/pembeli, pengaturan buku dibantu dengan NLP, terutama berdasarkan _book summary_/alur bukunya. |

### Data
Data diberikan dalam satu file, yakni **`nlp.csv`**, dan **memiliki 33,722 baris serta 8 kolom**. 
_Secara total, terdapat **~16,600 buku dan ~4,700 penulis unik** di dalam dataset ini!_

| Kolom              | Tipe Data | Deskripsi Singkat |
| ------------------ | :-------: | :---------------- |
| `wikipedia_ID`     | text      | ID dari Wikipedia. |
| `freebase_ID`      | text      | ID buku tersebut di [Freebase](https://en.wikipedia.org/wiki/Freebase_(database)) (sekarang telah dipindahkan ke [Wikidata](https://www.wikidata.org)). |
| `title`            | text      | Judul buku. |
| `author`           | text      | Nama penulis. |
| `publication_date` | text      | Tanggal publikasi (format bervariasi, bisa `YYYY`, `YYYY-MM`, maupun `YYYY-MM-DD`, sehingga tipe data bukan `date`). |
| `genre_ID`         | text      | ID genre buku tersebut di Freebase (lihat catatan di kolom `freebase_ID`). |
| `genre`            | text      | Genre buku (setiap baris memuat hanya satu genre, dan _semua kolom selain_ `genre` _dan_ `genre_ID` _diulang untuk setiap genre yang dimiliki buku tersebut_). |
| `summary`          | text      | Rangkuman alur cerita dari buku tersebut, dengan panjang dan tingkat detail yang bervariasi. |

**Berdasarkan pencarian di internet**, nampaknya data ini merupakan versi **modifikasi dari [CMU Book Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html)**.

## Metode
Secara teknis, **setiap _summary_ buku dapat dihitung sebagai dokumen terpisah**, sehingga problem ini bisa dikatakan adalah **problem _document similarity_**. Karena **sebagian besar data berupa teks** (kecuali tanggal nantinya), maka **digunakan teknik NLP** untuk menanganinya. Hasil penanganan data kemudian akan **dikelompokkan dengan _clustering_ menggunakan K-Means**.

### I. Preprocessing
Dibagi menjadi **_preprocessing_ umum** untuk data dan **_preprocessing_** yang lebih spesifik ke **NLP** (elaborasi tambahan juga disertakan untuk _notable preprocessing_).

#### Preprocessing Umum
1. Pengecekan **tipe data** (`publication_date` dikoreksi dan dipotong menjadi YYYY saja, sehingga seragam dan dapat dipakai sebagai integer)
2. Pengecekan **duplikat** (genre buku dipisahkan mmenjadi tabel terpisah agar tidak banyak duplikasi baris karena genre)
3. Pengecekan **_missing values_** (`author` dan `publication_date` banyak diisi dengan _query_ SPARQL ke Wikidata, `summary` juga ada yang diisi manual)

#### Preprocessing NLP
1. **_Lowercasing_**
2. Pembersihan **_noise_** (simbol-simbol atau teks-teks aneh yang tidak diinginkan)
3. **_Tokenization_** (bahasa Inggris, menggunakan `NLTK`)
4. Penghilangan **_stopwords_** (korpus bahasa Inggris `NLTK` + temuan hasil EDA)
5. **_Lemmatization_** (menggunakan `WordNetLemmatizer` dari `NLTK`) & **_Stemming_** (menggunakan `SnowballStemmer` dari `NLTK`)

### II. Feature Engineering
Terdapat 3 sumber _feature_ yang akan dipakai untuk _clustering_ K-Means:

#### 1. Doc2Vec
**Teknik utama yang dipakai untuk _summary_ buku**, Doc2Vec menggunakan _library_ `Gensim` dipilih alih-alih rata-rata Word2Vec maupun TF-IDF karena:
- Dapat **menghasilkan vektor langsung** untuk _clustering_
- **Lebih _specialized_** untuk menghasilkan vektor **dari dokumen**
- Dirasa **lebih dapat menangkap hubungan** dalam **alur cerita** dibandingkan kedua metode lainnya
(selama pembuatan proyek juga sebenarnya sudah dicoba-coba, namun yang paling cocok tetap Doc2Vec)

**Untuk proyek ini**, metode yang lebih _advanced_ seperti BERT tidak dipilih karena **keterbatasan waktu dan juga _hardware_** 😆 yang melakukan pemrosesan. Di akhir proyek, jumlah _feature_ yang akhirnya dipakai cukup _light but mean_!

#### 2. Named Entity Recognition (NER)
Terkadang, kita **mengklasifikasikan buku berdasarkan _setting_ cerita**. Misalnya, mungkin kita mengklasifikasikan buku-buku _spy flick_ yang ada CIA dan MI6nya bersama, atau kita bisa meletakkan buku-buku yang berlatar _historical_ Eropa berdekatan. Oleh karena itu, teknik NER dirasa dapat berkontribusi untuk _feature engineering_ proyek kali ini. 

Detail yang cukup menarik kali ini adalah **penggunaan dua _library_ untuk NER**, yakni `spaCy` dengan opsi `en_core_web_sm` dan `Stanza` dari Stanford `CoreNLP`. Berdasarkan riset saat pembuatan proyek, **opsi terbaik adalah memakai `Stanza`** di Python, namun waktu pemrosesan diperkirakan **akan memakan 11,5 jam!** Oleh karena itu, ada **teknik menarik untuk berkompromi** dengan cara **menggunakan `spaCy` terlebih dahulu** untuk _first pass_, lalu **diikuti dengan menggunakan `Stanza`** pada **hasil `spaCy`** untuk menghilangkan entitas-entitas tidak diinginkan yang masih tercampur. 

_Entitas yang digunakan: 'NORP', 'ORG', 'GPE', 'LOC', 'EVENT'_

#### 3. _Feature_ Hasil EDA Lainnya
**Dua _feature_ tambahan** digunakan setelah dievaluasi pada tahap EDA dan mempertimbangkan _problem statement_ proyek.
- **Periode publikasi juga dapat mempengaruhi gaya penulisan**, sehingga periode publikasi ditambahkan setelah dilakukan _cut_ manual
- **Panjang _summary_ merupakan _proxy_** yang cukup berguna untuk mengukur **kompleksitas plot buku** (memang kompleks atau sebaliknya tidak bisa dideskripsikan)


### III. Pengelompokkan _with_ Clustering
**Metode K-Means** digunakan setelah **semua _feature_ dilakukan _scaling_** (karena K-Means "peka dengan jarak"). 

K-Means dipilih karena **cukup sering dipakai untuk problem serupa di NLP**, sehingga menjadi pilihan yang masuk akal dan cukup aman (walaupun ada potensi alternatif lain seperti DBSCAN). Karena K-Means merupakan teknik _unsupervised learning_ yang **memerlukan jumlah _cluster_** , maka dilakukan baik **_elbow method_ maupun _silhoutte score_** untuk menentukan jumlah _cluster_. Di luar dari _code_ yang ditampilkan di sini, sebenarnya **beberapa variasi angka _cluster_** juga telah **dicoba manual** dan dicek angka mana yang secara **_cluster_ tidak terlalu _sparse_** isinya untuk pengelompokkan buku. 

_Akhirnya, dipilihlah enam sebagai jumlah _cluster_ dan dilakukanlah proses _clustering_._

## Kata Penutup
Seperti biasa, **detail lengkap pembuatannya tentu ada pada _notebook code_ yang telah tersedia di _repository_ ini!** 

Untuk proyek kali ini, aku **ingin membuat sesuatu yang** agak **inovatif** dan **melangkah cukup jauh di luar** dari **teknik-teknik yang disediakan di kelas** (TF-IDF dan Word2Vec). **[Proyek sebelumnya](https://github.com/feliciasanm/machine-learning-npl)**, aku membatasi diri memaksimalkan teknik yang dibahas di kelas (saat itu **_classification_ dengan _logistic regression_**), yang menjadi motivasiku di proyek kali ini untuk **tidak mengenal kompromi belajar** meskipun tekniknya mesti belajar otodidak ataupun cara penggunaannya _out of the box_ 🤣

_Siapa yang tahu kapan aku akan menggunakan NLP lagi? Secara personal, _next time_ aku juga ingin mencoba _network analysis_..._