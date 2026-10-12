# ⚽ Data Manager: Satu Musim

**Playbook pelatih · edisi data science**

| | |
|---|---|
| **Genre** | Simulasi manajer sepak bola, bergaya *Football Manager* mini |
| **Sasaran** | Siswa SMA, tanpa syarat pengalaman coding |
| **Durasi** | Satu sesi = satu musim penuh |
| **Teknologi** | Python, Streamlit, pandas, scikit-learn |

> Kamu manajer tim sekolah. Sebelum peluit pertama berbunyi, kamu harus membersihkan laporan scout,
> menyusun taktik, melatih pemain, lalu melatih model machine learning yang memprediksi hasil
> pertandingan. **Satu pertandingan. Satu kesempatan.**

---

## 📋 Susunan dokumen

`Instalasi` → `Aturan main` → `Empat babak` → `Skor & peringkat` → `Peta data science` → `Catatan desain`

---

## 1 · Instalasi

### Persiapan

- Python **3.10 atau lebih baru**
- Satu folder berisi `season_manager.py`
- Koneksi internet untuk memasang pustaka

### Perintah

| Langkah | Windows | macOS / Linux |
|---------|---------|---------------|
| Buat lingkungan | `python -m venv .venv` | `python3 -m venv .venv` |
| Aktifkan | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| Pasang pustaka | `pip install streamlit pandas numpy scikit-learn` | *(sama)* |
| Jalankan | `streamlit run season_manager.py` | *(sama)* |

Game terbuka otomatis di `http://localhost:8501`.

### Kalau ada kendala

| Gejala | Penyebab / solusi |
|--------|-------------------|
| `streamlit` tidak dikenali | Pakai `python -m streamlit run season_manager.py` |
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| Port 8501 terpakai | Tambahkan `--server.port 8502` |

*Diuji pada Python 3.13, Streamlit 1.65, pandas 3.0, numpy 2.5, scikit-learn 1.9.*

---

## 2 · Aturan main

### Tujuan

Kalahkan **tim lawan** (nama dan rating tampil sejak awal, berbeda tiap musim) sambil mengumpulkan
sampai **4 ⭐**, satu bintang per babak.

### Tiga tier di setiap babak

Setiap babak bisa dikerjakan dengan tiga cara. Semuanya **dinilai sama** untuk bintang dan skor.

| Tier | Gaya | Kamu mengerjakan |
|------|------|------------------|
| 🟢 **1** | Klik saja | Memilih opsi, kode dijalankan otomatis |
| 🟡 **2** | Isi kekosongan | Melengkapi `____` pada kode |
| 🔴 **3** | Kode bebas | Menulis Python sendiri |

### Alur dan kunci

Babak dikerjakan **berurutan**. Babak berikutnya terbuka setelah yang sebelumnya selesai.
Kamu boleh **mengulang babak sebelumnya** selama belum bertanding, tetapi hasil babak setelahnya akan
**di-reset** karena datanya sudah berubah.

> ⚠️ **Pertandingan hanya dimainkan sekali.** Setelah itu semua babak terkunci (hanya bisa dibaca),
> dan satu-satunya jalan adalah **Mulai Musim Baru** dengan data yang baru.

---

## 3 · Empat babak

### 🔍 Babak 1 · Talent Scouting: *bersihkan data*

Laporan scout masih kotor: **24 baris** untuk **22 pemain unik**.

| Masalah di data | Contoh |
|-----------------|--------|
| Nilai kosong | `pace`, `technique`, atau `mental` bernilai `NaN` |
| Baris duplikat | Pemain yang sama dilaporkan dua kali |
| Format usia campur | `17` bercampur dengan `'18 th'` |

**⭐ Target:** data bersih **tanpa kehilangan satu pun** dari 22 pemain unik.

| Tier | Pilihan |
|------|---------|
| 1 | Pilih "Isi dengan rata-rata" atau "Hapus baris kosong", plus dua *checkbox* |
| 2 | Isi tipe data usia (`int`) dan nama method statistik (`mean`) |
| 3 | Tulis pembersihan sendiri; simpan hasilnya di `clean_df` |

> 💡 **Jebakan yang disengaja:** "Hapus baris kosong" membuang 3 pemain, jadi ⭐ tidak didapat.
> Mengisi nilai kosong (dengan rata-rata atau median) mempertahankan semuanya.

### 📊 Babak 2 · Squad & Tactics: *analisis skuad*

1. Pilih **formasi**: `4-4-2`, `4-3-3`, atau `3-5-2`.
2. Pilih **11 pemain** yang jumlah posisinya pas (GK, DEF, MID, FWD).
3. Dapatkan **Tactic Score**.

```
Tactic Score  =  rata-rata rating Starting XI  +  5  (jika formasi sesuai)
```

**⭐ Target:** Tactic Score mencapai angka target yang tertera di layar. Target dihitung dari
datamu sendiri (skor terbaik yang mungkin dibentuk dikurangi 3), jadi **selalu bisa dicapai**.

| Tier | Pilihan |
|------|---------|
| 1 | Klik satu tombol; muncul grafik rata-rata rating per posisi |
| 2 | Isi `groupby(____)[____].mean()` |
| 3 | Bebas (misal korelasi antar atribut); simpan skor di `tactic_score` |

> 💡 Kode default di Tier 3 menghitung `tactic_score = avg_rating` **tanpa bonus +5**,
> sehingga bintang tidak didapat. Siswa perlu membaca petunjuk di dalam kode untuk menyadarinya.

### 🏋️ Babak 3 · Training: *rekayasa fitur*

Setiap pemain diberi **satu fokus latihan** (fisik, teknik, atau mental) yang menambah **+5** pada atribut itu.
Lalu dibuat fitur baru:

```
fitness_index  =  rata-rata (pace, technique, mental)  setiap pemain
Team Fitness   =  rata-rata fitness_index seluruh Starting XI
```

**⭐ Target:** Team Fitness mencapai angka target yang tertera.

| Tier | Pilihan |
|------|---------|
| 1 | Satu klik menerapkan semua pilihan latihan |
| 2 | Isi besar peningkatan per sesi (`+ ____`) |
| 3 | Terapkan latihan sendiri dan **buat kolom `fitness_index`** |

Game memeriksa hasil latihanmu: atribut tidak boleh naik lebih dari +5, tidak boleh turun, dan tidak boleh
melebihi 100.

### 🏟️ Babak 4 · Match: *machine learning*

Model dilatih dari **250 pertandingan lama** (data sintetis) dengan empat fitur:

| Fitur | Arti |
|-------|------|
| `team_rating` | Rata-rata rating timmu |
| `tactic_score` | Dari Babak 2 |
| `team_fitness` | Dari Babak 3 |
| `rating_diff` | Rating timmu dikurangi rating lawan |

Label yang diprediksi: `win` (1 = menang, 0 = tidak).

**⭐ Target:** akurasi model di **data uji** minimal **65%**.

| Tier | Pilihan |
|------|---------|
| 1 | Pilih *Logistic Regression* atau *Random Forest* |
| 2 | Isi `model = ____` |
| 3 | Bagi data sendiri (`train_test_split`), latih model apa pun, simpan `model` dan `accuracy` |

Setelah model dilatih, game menghitung **peluang menang** dan memainkan pertandingan:
**Menang / Seri / Kalah**. Hasilnya mengandung unsur peluang.
Kerja data yang baik menaikkan peluang, tetapi tidak menjaminnya.

---

## 4 · Skor dan peringkat

```
Skor musim  =  jumlah bintang (maks 4)  +  poin hasil
Poin hasil  :  Menang = 2   Seri = 1   Kalah = 0
Maksimum    :  6
```

| Skor | Peringkat |
|:----:|-----------|
| 6 | 🏆 Juara Liga |
| 5 | 🥈 Runner-up |
| 3 – 4 | 🥉 Papan Tengah |
| 0 – 2 | ⚠️ Zona Degradasi |

Tab **🏆 Season Report** merangkum formasi, semua skor, status tiap bintang, dan pertanyaan refleksi.

---

## 5 · Peta data science: dari lapangan ke notebook

Seluruh game adalah satu **pipeline data science** yang dibungkus cerita sepak bola:

```
 SCOUT           SKUAD             LATIHAN             PERTANDINGAN
 ─────           ─────             ───────             ────────────
 Data mentah ──▶ Eksplorasi ─────▶ Fitur baru ───────▶ Model prediksi
 (kotor)         (statistik)       (fitness_index)     (klasifikasi)

 Data cleaning   EDA               Feature             Machine learning
                                   engineering         + evaluasi
```

| Babak | Konsep | Keterampilan yang dilatih | Contoh kode |
|-------|--------|---------------------------|-------------|
| 1 | **Data cleaning** | Menangani *missing values*, duplikat, tipe data | `fillna(df[k].mean())`, `drop_duplicates(...)`, `.str.extract(r"(\d+)")` |
| 2 | **EDA & statistik** | Agregasi per kelompok, korelasi | `groupby("position")["rating"].mean()`, `.corr()` |
| 3 | **Feature engineering** | Membuat variabel baru dari variabel lama | `df[[...]].mean(axis=1)` |
| 4 | **Klasifikasi** | Melatih, menguji, membandingkan model | `LogisticRegression()`, `train_test_split`, `accuracy_score` |

### Pelajaran tersembunyi

- **Keputusan cleaning punya harga.** `dropna` terasa praktis, tapi membuang 3 dari 22 pemain, yaitu 14% talenta.
- **Data bersih di awal menentukan segalanya di akhir.** Tactic Score dan Team Fitness berasal dari
  pemain yang lolos Babak 1.
- **Akurasi tergantung cara kita menguji.** Pada data yang sama, split bawaan memberi
  Logistic Regression ±72%, sedangkan split default Tier 3 (`test_size=0.25`, `random_state=7`) memberi ±60%.
  Itu bahan diskusi tentang *variasi evaluasi*.
- **Model yang "lebih canggih" belum tentu lebih baik.** Random Forest (±66%) kalah dari Logistic Regression
  (±72%) pada split bawaan, karena hubungan di data ini memang cenderung linear.
- **Probabilitas bukan kepastian.** Peluang menang 60% berarti tidak menang di sekitar 4 dari 10 musim.

---

## 6 · Catatan desain dan batasan

| Topik | Catatan |
|-------|---------|
| **Peluang** | Dengan permainan sangat baik (4 ⭐ di 14 percobaan), hasil pertandingan: 6 menang, 5 seri, 3 kalah. Peluang menang rata-rata ±50%. Panduan di dalam game sudah menjelaskan unsur peluang ini. |
| **Pilihan fokus latihan** | Belum memengaruhi Team Fitness, karena `fitness_index` adalah rata-rata tiga atribut dan +5 pada atribut mana pun hasilnya sama. Teks "latihan tepat sasaran" belum sesuai dengan mekanik ini. |
| **Data historis** | Sintetis dan dibangkitkan dengan seed tetap (42), sehingga sama setiap musim. Data pemain dan lawan berganti tiap musim. |
| **Progres** | Hilang jika browser di-refresh. Satu sesi = satu musim. |
| **Keamanan** | Kode siswa dijalankan dengan `exec()` tanpa sandbox penuh. Pakai di komputer sendiri atau jaringan kelas; **jangan dipublikasikan** tanpa isolasi tambahan. |

### Ide diskusi akhir musim

1. Bagian mana yang paling menentukan hasil: cleaning, taktik, latihan, atau model? Bagaimana kamu tahu?
2. Kalau kamu menang walau semua bintang kosong, apakah itu bukti kerja datamu bagus?
3. Kenapa akurasi 65% sudah dianggap cukup baik di game ini?
4. Apa yang akan kamu ubah di Babak 1 kalau diberi satu kesempatan lagi?
