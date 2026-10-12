"""
Data Manager: Satu Musim
=========================
Game edukasi data science bertema manajer sepak bola (ala Football Manager)
untuk siswa SMA. Satu playthrough = satu "musim mini" yang terdiri dari
4 section berurutan:

    1. Talent Scouting   -> Data Cleaning
    2. Squad & Tactics   -> EDA / Statistik
    3. Training          -> Feature Engineering
    4. Match             -> Machine Learning (klasifikasi)

TUJUAN: kalahkan tim lawan di pertandingan akhir, sambil mengumpulkan
sampai 4 bintang (satu target per section). Skor musim = bintang + poin
hasil pertandingan (Menang 2, Seri 1, Kalah 0), maksimal 6.

Semua state disimpan di st.session_state (reset saat browser di-refresh).
Tidak ada database eksternal -> sesuai desain "tidak session-heavy":
satu sesi = satu musim penuh, lalu selesai.

Jalankan dengan:
    streamlit run season_manager.py
"""

import random

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Data Manager: Satu Musim", layout="wide", page_icon="⚽")

# =============================================================================
# KONSTANTA GAME
# =============================================================================

ATTR = ["pace", "technique", "mental"]
BOOST_LATIHAN = 5          # tambahan atribut per sesi latihan
BONUS_FORMASI = 5          # bonus Tactic Score jika formasi sesuai
MARGIN_TARGET = 3          # target = skor terbaik yang mungkin dikurangi margin ini
TARGET_AKURASI = 0.65      # target akurasi model di data uji
POIN_HASIL = {"Menang": 2, "Seri": 1, "Kalah": 0}
MAKS_BINTANG = 4
MAKS_SKOR = MAKS_BINTANG + max(POIN_HASIL.values())

FORMATIONS = {
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
}

# =============================================================================
# UTILITIES
# =============================================================================

def tampil_df(df, **kw):
    """st.dataframe selebar kolom, kompatibel dengan versi Streamlit lama dan baru."""
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def friendly_error(e: Exception) -> str:
    """Terjemahkan exception Python jadi pesan ramah siswa."""
    msg = str(e)
    if isinstance(e, KeyError):
        return f"Kolom atau key {msg} tidak ditemukan. Cek lagi nama kolom dataframe kamu."
    if isinstance(e, NameError):
        return f"Ada variabel yang belum didefinisikan: {msg}."
    if isinstance(e, SyntaxError):
        return f"Ada kesalahan penulisan kode (syntax error): {msg}."
    if isinstance(e, TypeError):
        return f"Tipe data tidak cocok untuk operasi ini: {msg}."
    if isinstance(e, AttributeError):
        return f"Method/atribut tidak ditemukan: {msg}. Cek lagi nama fungsinya."
    if isinstance(e, ValueError):
        return f"Nilai yang dipakai tidak valid: {msg}."
    return f"Terjadi error: {msg}"


def run_student_code(code: str, local_vars: dict):
    """Jalankan kode siswa. Nama yang tersedia: pd, np, beberapa kelas sklearn,
    ditambah variabel yang diberikan lewat local_vars.

    Catatan: ini BUKAN sandbox keamanan penuh, cukup untuk penggunaan kelas/lokal."""
    ns = {
        "pd": pd,
        "np": np,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
        **local_vars,
    }
    try:
        exec(code, ns)  # satu namespace, supaya loop/comprehension bisa melihat semua variabel
        return ns, None
    except Exception as e:  # noqa: BLE001 - sengaja tangkap semua untuk pesan ramah
        return ns, friendly_error(e)


def init_state():
    defaults = {
        "intro_seen": False,
        "seed_initialized": False,
        "raw_scout_df": None,
        "n_unik": None,
        "opponent": None,
        "clean_df": None,
        "section1_done": False,
        "squad_df": None,
        "formation": None,
        "tactic_score": None,
        "target_tactic": None,
        "section2_done": False,
        "trained_squad_df": None,
        "team_fitness": None,
        "target_fitness": None,
        "section3_done": False,
        "model_accuracy": None,
        "match_result": None,
        "win_prob": None,
        "section4_done": False,
        "hasil": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_season():
    """Mulai musim baru (dipakai sebagai on_click callback). Panduan tidak ditampilkan ulang."""
    for k in list(st.session_state.keys()):
        if k != "intro_seen":
            del st.session_state[k]


def mulai():
    st.session_state.intro_seen = True


def terkunci() -> bool:
    """Setelah pertandingan dimainkan, musim dianggap selesai dan section 1-4 hanya bisa dibaca."""
    return bool(st.session_state.section4_done)


# --- hasil per section disimpan di state supaya tidak hilang saat halaman dirender ulang ---

def simpan_hasil(kunci, pesan=None, metrics=None, tabel=None, chart=None):
    st.session_state.hasil[kunci] = {
        "pesan": pesan or [],
        "metrics": metrics or {},
        "tabel": tabel or [],
        "chart": chart,
    }


def tampilkan_hasil(kunci):
    h = st.session_state.hasil.get(kunci)
    if not h:
        return
    st.divider()
    st.markdown("**Hasil terakhir**")
    for tipe, teks in h["pesan"]:
        getattr(st, tipe)(teks)
    if h["metrics"]:
        cols = st.columns(len(h["metrics"]))
        for col, (nama, nilai) in zip(cols, h["metrics"].items()):
            col.metric(nama, nilai)
    if h["chart"] is not None:
        st.bar_chart(h["chart"])
    for judul, df in h["tabel"]:
        st.caption(judul)
        tampil_df(df)


def reset_downstream(dari: int):
    """Kalau section sebelumnya dikerjakan ulang, hasil section sesudahnya tidak berlaku lagi."""
    s = st.session_state
    if dari <= 2:
        s.section2_done, s.squad_df, s.formation, s.tactic_score = False, None, None, None
        s.hasil.pop("s2", None)
    if dari <= 3:
        s.section3_done, s.trained_squad_df, s.team_fitness = False, None, None
        s.hasil.pop("s3", None)


# =============================================================================
# DATA GENERATION (musim baru = data baru)
# =============================================================================

FIRST_NAMES = ["Andi", "Budi", "Dedi", "Rizky", "Fajar", "Agus", "Dimas", "Eko",
               "Farhan", "Galih", "Hendra", "Irfan", "Joko", "Kevin", "Lukman",
               "Made", "Nanda", "Oscar", "Putra", "Rian", "Satria", "Taufik",
               "Umar", "Wahyu", "Yoga", "Zaki"]
LAST_NAMES = ["Pratama", "Saputra", "Wijaya", "Santoso", "Kurniawan",
              "Firmansyah", "Hidayat", "Permana", "Nugraha", "Setiawan"]

POSITION_POOL = ["GK"] * 3 + ["DEF"] * 7 + ["MID"] * 7 + ["FWD"] * 5


def generate_scout_data(n=22, seed=None):
    """Generate dataset scout mentah & kotor: missing values, duplikat, format usia campur."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    names = []
    used = set()
    while len(names) < n:
        nm = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if nm not in used:
            used.add(nm)
            names.append(nm)

    positions = POSITION_POOL[:n]
    rng.shuffle(positions)

    rows = []
    for i in range(n):
        pace = int(np_rng.randint(45, 96))
        technique = int(np_rng.randint(45, 96))
        mental = int(np_rng.randint(45, 96))
        overall = round((pace + technique + mental) / 3 + np_rng.normal(0, 2), 1)
        age = rng.randint(16, 19)
        rows.append({
            "name": names[i],
            "position": positions[i],
            "age": age,
            "pace": pace,
            "technique": technique,
            "mental": mental,
            "rating": overall,
        })

    df = pd.DataFrame(rows)

    # --- kotori data secara sengaja ---
    # 1) missing values di beberapa atribut
    for col in ATTR:
        missing_idx = df.sample(frac=0.15, random_state=seed).index
        df.loc[missing_idx, col] = np.nan

    # 2) format usia campur (beberapa jadi string kotor)
    #    kolom dibuat bertipe object dulu: pandas versi baru menolak teks di kolom int
    df["age"] = df["age"].astype(object)
    messy_age_idx = df.sample(n=4, random_state=seed).index
    for idx in messy_age_idx:
        df.loc[idx, "age"] = f"{int(df.loc[idx, 'age'])} th"

    # 3) duplikat data (scout lain melaporkan pemain yang sama persis)
    dup_rows = df.sample(n=2, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # acak urutan baris biar tidak terlihat "rapi"
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_opponent(seed=None, difficulty=1.0):
    """Generate ringkasan tim lawan (tidak perlu detail per pemain)."""
    rng = np.random.RandomState(seed)
    base_rating = rng.uniform(55, 80) * difficulty
    return {
        "name": str(rng.choice(["SMA Garuda", "SMA Elang Muda", "SMA Cendana",
                                "SMA Nusantara", "SMA Harapan Bangsa"])),
        "rating": round(float(base_rating), 1),
    }


def generate_historical_matches(n=250, seed=42):
    """Data pertandingan musim-musim sebelumnya (sintetis) untuk melatih model."""
    rng = np.random.RandomState(seed)
    team_rating = rng.uniform(50, 90, n)
    tactic_score = rng.uniform(50, 90, n)
    team_fitness = rng.uniform(50, 90, n)
    opp_rating = rng.uniform(50, 90, n)
    rating_diff = team_rating - opp_rating

    # probabilitas menang dipengaruhi rating_diff, tactic_score, fitness + noise
    logit = 0.08 * rating_diff + 0.03 * (tactic_score - 70) + 0.03 * (team_fitness - 70)
    prob_win = 1 / (1 + np.exp(-logit))
    win = (rng.rand(n) < prob_win).astype(int)

    return pd.DataFrame({
        "team_rating": team_rating,
        "tactic_score": tactic_score,
        "team_fitness": team_fitness,
        "rating_diff": rating_diff,
        "win": win,
    })


# =============================================================================
# TARGET, VALIDASI, DAN SKOR
# =============================================================================

def potensi_skuad(clean_df):
    """Skor terbaik yang secara teori bisa dicapai dari data bersih (untuk menentukan target).

    Dihitung per musim dari data siswa sendiri, jadi target selalu bisa dicapai.
    Mengembalikan None kalau tidak ada formasi yang bisa dibentuk."""
    attr_mean = clean_df[ATTR].mean(axis=1)
    best_tac, best_fit = None, None
    for need in FORMATIONS.values():
        idx_rating, idx_fit, layak = [], [], True
        for pos, n in need.items():
            sub = clean_df[clean_df["position"] == pos]
            if len(sub) < n:
                layak = False
                break
            idx_rating += sub.nlargest(n, "rating").index.tolist()
            idx_fit += attr_mean[sub.index].nlargest(n).index.tolist()
        if not layak:
            continue
        tac = clean_df.loc[idx_rating, "rating"].mean() + BONUS_FORMASI
        fit = attr_mean[idx_fit].mean() + BOOST_LATIHAN / 3
        best_tac = tac if best_tac is None else max(best_tac, tac)
        best_fit = fit if best_fit is None else max(best_fit, fit)
    if best_tac is None:
        return None
    return {"tactic": float(best_tac), "fitness": float(best_fit)}


def validate_clean(df):
    needed = ["name", "position", "age", "pace", "technique", "mental", "rating"]
    if not isinstance(df, pd.DataFrame):
        return ["`clean_df` harus berupa DataFrame."]
    if len(df) == 0:
        return ["Dataframe hasil pembersihan kosong."]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        return [f"Kolom berikut hilang dari `clean_df`: {', '.join(missing_cols)}."]
    problems = []
    if df[ATTR].isna().sum().sum() > 0:
        problems.append("Masih ada nilai kosong di pace/technique/mental.")
    if df.duplicated(subset=["name", "position"]).sum() > 0:
        problems.append("Masih ada baris duplikat.")
    try:
        pd.to_numeric(df["age"])
    except Exception:
        problems.append("Kolom 'age' belum semuanya berupa angka.")
    return problems


def validasi_training(squad_df, trained_df):
    """Pastikan hasil latihan mengikuti aturan game. Mengembalikan pesan error atau None."""
    if not isinstance(trained_df, pd.DataFrame) or "fitness_index" not in trained_df.columns:
        return "Pastikan `trained_df` ada dan punya kolom `fitness_index`."
    try:
        sebelum = squad_df.set_index("name")[ATTR]
        sesudah = trained_df.set_index("name")[ATTR]
        selisih = sesudah - sebelum
    except Exception:
        return "`trained_df` harus tetap memuat kolom name, pace, technique, dan mental seperti `squad_df`."
    if (selisih > BOOST_LATIHAN + 0.001).any().any():
        return f"Satu sesi latihan hanya menambah maksimal +{BOOST_LATIHAN} per atribut. Cek lagi kodemu."
    if (selisih < -0.001).any().any():
        return "Latihan tidak boleh mengurangi atribut pemain."
    if (sesudah > 100.001).any().any():
        return "Atribut pemain maksimal 100."
    return None


def hitung_bintang():
    """Bintang dihitung dari hasil terbaru yang tersimpan, hanya untuk section yang sudah selesai."""
    s = st.session_state
    return {
        "data": bool(s.section1_done and s.clean_df is not None and len(s.clean_df) >= (s.n_unik or 0)),
        "taktik": bool(s.section2_done and s.tactic_score is not None
                       and s.target_tactic is not None and s.tactic_score >= s.target_tactic),
        "fitness": bool(s.section3_done and s.team_fitness is not None
                        and s.target_fitness is not None and s.team_fitness >= s.target_fitness),
        "model": bool(s.section4_done and s.model_accuracy is not None
                      and s.model_accuracy >= TARGET_AKURASI),
    }


def peringkat(total: int) -> str:
    if total >= 6:
        return "🏆 Juara Liga"
    if total >= 5:
        return "🥈 Runner-up"
    if total >= 3:
        return "🥉 Papan Tengah"
    return "⚠️ Zona Degradasi"


# =============================================================================
# APP HEADER, TUJUAN, CARA BERMAIN
# =============================================================================

init_state()

if not st.session_state.seed_initialized:
    _seed = random.randint(1, 999999)
    st.session_state.season_seed = _seed
    _raw = generate_scout_data(seed=_seed)
    st.session_state.raw_scout_df = _raw
    st.session_state.n_unik = int(_raw.drop_duplicates(subset=["name", "position"]).shape[0])
    st.session_state.opponent = generate_opponent(seed=_seed)
    st.session_state.seed_initialized = True

opponent = st.session_state.opponent

st.title("⚽ Data Manager: Satu Musim")
st.caption(
    "Kelola tim sekolahmu selama satu musim: scouting pemain, susun taktik, "
    "latih tim, lalu bertanding — semua lewat data science."
)

st.info(
    f"🎯 **Tujuan musim ini:** kalahkan **{opponent['name']}** (rating tim {opponent['rating']}) di "
    f"pertandingan akhir, sambil mengumpulkan sampai **{MAKS_BINTANG} ⭐**, satu target per section. "
    "Pertandingan hanya dimainkan **satu kali** dan hasilnya mengandung unsur peluang: "
    "kerja data yang baik menaikkan peluang menang, tapi tidak menjaminnya."
)

with st.expander("📖 Cara bermain", expanded=not st.session_state.intro_seen):
    st.markdown(
        f"""
**Alur satu musim** (4 section, dikerjakan berurutan; section berikutnya terbuka setelah yang sebelumnya selesai)
1. 🔍 **Talent Scouting:** bersihkan data laporan scout yang berantakan (nilai kosong, duplikat, format usia campur).
2. 📊 **Squad & Tactics:** pilih formasi dan 11 pemain Starting XI, lalu analisis skuadmu untuk mendapat **Tactic Score**.
3. 🏋️ **Training:** tentukan fokus latihan tiap pemain dan buat fitur baru `fitness_index` untuk mendapat **Team Fitness**.
4. 🏟️ **Match:** latih model machine learning dari data pertandingan lama, lalu mainkan pertandingan melawan lawanmu.

**Tiga tier di setiap section**
- 🟢 **Tier 1:** tinggal pilih dan klik. Kodenya sudah disiapkan.
- 🟡 **Tier 2:** lengkapi bagian kosong (`____`) pada kode.
- 🔴 **Tier 3:** tulis kode Python sendiri.

Pilih tier mana pun yang nyaman, dan boleh berpindah kapan saja. Semua tier dihitung sama untuk bintang dan skor.

**Target ⭐ per section** (target angka dihitung dari datamu sendiri, jadi selalu bisa dicapai)
- ⭐ **Data:** bersihkan data **tanpa kehilangan satu pun pemain unik**.
- ⭐ **Taktik:** Tactic Score mencapai target yang tertera di section 2.
- ⭐ **Fitness:** Team Fitness mencapai target yang tertera di section 3.
- ⭐ **Model:** akurasi model di data uji minimal **{TARGET_AKURASI:.0%}**.

**Skor musim** = jumlah bintang (maks {MAKS_BINTANG}) + poin hasil pertandingan (Menang {POIN_HASIL['Menang']}, Seri {POIN_HASIL['Seri']}, Kalah {POIN_HASIL['Kalah']}), maksimal **{MAKS_SKOR}**.
🏆 Juara Liga (6) • 🥈 Runner-up (5) • 🥉 Papan Tengah (3-4) • ⚠️ Zona Degradasi (0-2)

**Tips**
- Kerjakan section lebih awal dengan teliti: kualitas data di awal menentukan hasil di akhir.
- Kamu boleh mengulang section yang sudah selesai sebelum bertanding. Section sesudahnya akan ikut di-reset.
- Setelah pertandingan dimainkan, musim selesai. Klik **Mulai Musim Baru** untuk bermain lagi dengan data baru.
"""
    )
    st.button("Saya mengerti, mulai musim!", on_click=mulai, type="primary", key="btn_mulai")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 1. Talent Scouting",
    "📊 2. Squad & Tactics",
    "🏋️ 3. Training",
    "🏟️ 4. Match",
    "🏆 Season Report",
])

# =============================================================================
# SECTION 1 — TALENT SCOUTING (DATA CLEANING)
# =============================================================================

def selesai_s1(df):
    """Validasi hasil cleaning, simpan, lalu render ulang supaya section 2 langsung terbuka."""
    masalah = validate_clean(df)
    if masalah:
        for p in masalah:
            st.error(p)
        return
    s = st.session_state
    reset_downstream(2)
    s.clean_df = df
    s.section1_done = True
    hilang = s.n_unik - len(df)
    pesan = [("success", "Data berhasil dibersihkan! Lanjut ke tab Squad & Tactics.")]
    if hilang <= 0:
        pesan.append(("success", f"⭐ Target tercapai: semua {s.n_unik} pemain unik dipertahankan."))
    else:
        pesan.append((
            "warning",
            f"Target ⭐ belum tercapai: {hilang} dari {s.n_unik} pemain unik terbuang. "
            "Adakah cara membersihkan nilai kosong tanpa kehilangan pemain?",
        ))
    simpan_hasil("s1", pesan=pesan, tabel=[(f"Data bersih ({len(df)} pemain)", df)])
    st.rerun()


with tab1:
    st.header("🔍 Talent Scouting — Data Cleaning")
    st.markdown(
        "Scout kamu baru saja mengirim laporan pemain, tapi datanya masih **kotor**: "
        "ada nilai kosong, data usia dengan format campur, dan beberapa baris duplikat. "
        "Bersihkan dulu sebelum kamu bisa merekrut pemain!"
    )
    st.info(
        f"🎯 **Target ⭐ section ini:** data bersih (tanpa nilai kosong, duplikat, dan usia berupa angka) "
        f"**tanpa kehilangan satu pun dari {st.session_state.n_unik} pemain unik**."
    )

    raw_df = st.session_state.raw_scout_df

    if terkunci():
        st.caption("Musim sudah selesai. Bagian ini hanya bisa dibaca.")
        tampilkan_hasil("s1")
    else:
        st.subheader("Data Mentah dari Scout")
        tampil_df(raw_df.astype({"age": str}))  # age campuran angka/teks, ditampilkan sebagai teks

        n_missing = raw_df[ATTR].isna().sum().sum()
        n_dupes = raw_df.duplicated().sum()
        st.info(
            f"📋 Ditemukan **{n_missing}** nilai kosong, **{n_dupes}** baris duplikat, "
            f"dan beberapa format usia yang tidak konsisten (contoh: `'18 th'`)."
        )

        t1, t2, t3 = st.tabs(["🟢 Tier 1: Pilih Metode", "🟡 Tier 2: Lengkapi Kode", "🔴 Tier 3: Kode Bebas"])

        with t1:
            st.markdown("Pilih strategi pembersihan, lalu klik **Terapkan**.")
            missing_strategy = st.selectbox(
                "Cara menangani nilai kosong:",
                ["Isi dengan rata-rata (mean)", "Hapus baris yang datanya kosong"],
                key="s1_t1_strategy",
            )
            remove_dupes = st.checkbox("Hapus baris duplikat", value=True, key="s1_t1_dupes")
            fix_age = st.checkbox("Standarkan kolom usia menjadi angka", value=True, key="s1_t1_age")

            if st.button("✅ Terapkan Pembersihan", key="tier1_clean_btn"):
                df = raw_df.copy()
                if remove_dupes:
                    df = df.drop_duplicates(subset=["name", "position"]).reset_index(drop=True)
                if fix_age:
                    df["age"] = df["age"].astype(str).str.extract(r"(\d+)")[0].astype(int)
                if missing_strategy.startswith("Isi"):
                    for col in ATTR:
                        df[col] = df[col].fillna(df[col].mean())
                else:
                    df = df.dropna(subset=ATTR).reset_index(drop=True)
                selesai_s1(df)

        with t2:
            st.markdown("Lengkapi bagian yang kosong (`____`) pada kode di bawah ini.")
            st.code(
                "df = raw_df.copy()\n"
                "df = df.drop_duplicates(subset=['name', 'position'])\n"
                "df['age'] = df['age'].astype(str).str.extract(r'(\\d+)').astype(____BLANK1____)\n"
                "kolom = ['pace', 'technique', 'mental']\n"
                "df[kolom] = df[kolom].fillna(df[kolom].____BLANK2____())\n"
                "clean_df = df",
                language="python",
            )
            blank1 = st.text_input("BLANK1 (tipe data untuk kolom usia):", placeholder="contoh: int", key="s1_t2_b1")
            blank2 = st.text_input(
                "BLANK2 (nama method statistik untuk mengisi nilai kosong):",
                placeholder="contoh: mean",
                key="s1_t2_b2",
            )
            if st.button("▶️ Jalankan Kode", key="tier2_clean_btn"):
                if not blank1 or not blank2:
                    st.warning("Isi dulu semua bagian yang kosong.")
                else:
                    code = (
                        "df = raw_df.copy()\n"
                        "df = df.drop_duplicates(subset=['name', 'position'])\n"
                        f"df['age'] = df['age'].astype(str).str.extract(r'(\\d+)').astype({blank1})\n"
                        "kolom = ['pace', 'technique', 'mental']\n"
                        f"df[kolom] = df[kolom].fillna(df[kolom].{blank2}())\n"
                        "clean_df = df"
                    )
                    result, err = run_student_code(code, {"raw_df": raw_df})
                    if err:
                        st.error(err)
                    else:
                        selesai_s1(result.get("clean_df"))

        with t3:
            st.markdown(
                "Tulis kode Python sendiri untuk membersihkan `raw_df`. "
                "Simpan hasil akhirmu ke variabel **`clean_df`**."
            )
            default_code = (
                "# raw_df sudah tersedia sebagai variabel\n"
                "df = raw_df.copy()\n"
                "# TODO: hapus duplikat\n"
                "# TODO: standarkan kolom age jadi int\n"
                "# TODO: tangani nilai kosong di pace/technique/mental\n"
                "clean_df = df\n"
            )
            student_code = st.text_area("Kode kamu:", value=default_code, height=220, key="tier3_clean_code")
            if st.button("▶️ Jalankan Kode", key="tier3_clean_btn"):
                result, err = run_student_code(student_code, {"raw_df": raw_df})
                if err:
                    st.error(err)
                elif "clean_df" not in result:
                    st.error("Variabel `clean_df` belum didefinisikan di kodemu.")
                else:
                    selesai_s1(result["clean_df"])

        tampilkan_hasil("s1")

# =============================================================================
# SECTION 2 — SQUAD & TACTICS (EDA / STATISTIK)
# =============================================================================

def compute_tactic_score(sq_df, formation_ok):
    bonus = BONUS_FORMASI if formation_ok else 0
    return round(sq_df["rating"].mean() + bonus, 1)


def selesai_s2(squad_df, formation, tactic_score, tabel=None, chart=None):
    try:
        tactic_score = round(float(tactic_score), 1)
    except (TypeError, ValueError):
        st.error("`tactic_score` harus berupa angka.")
        return
    if not (0 <= tactic_score <= 100):
        st.error("`tactic_score` harus berada di antara 0 dan 100.")
        return

    s = st.session_state
    reset_downstream(3)
    s.squad_df = squad_df
    s.formation = formation
    s.tactic_score = tactic_score
    s.section2_done = True

    pesan = [("success", "Analisis squad selesai! Lanjut ke tab Training.")]
    if s.target_tactic is not None:
        if tactic_score >= s.target_tactic:
            pesan.append(("success", f"⭐ Target tercapai: Tactic Score {tactic_score} ≥ {s.target_tactic}."))
        else:
            pesan.append((
                "warning",
                f"Target ⭐ belum tercapai: Tactic Score {tactic_score}, target ≥ {s.target_tactic}. "
                "Coba susun ulang Starting XI, atau periksa lagi rumus skor taktikmu.",
            ))
    simpan_hasil("s2", pesan=pesan, metrics={"Tactic Score": tactic_score}, tabel=tabel, chart=chart)
    st.rerun()


with tab2:
    st.header("📊 Squad & Tactics — EDA & Statistik")

    if not st.session_state.section1_done:
        st.warning("Selesaikan dulu section **Talent Scouting** untuk membuka bagian ini.")
    else:
        clean_df = st.session_state.clean_df
        pot = potensi_skuad(clean_df)
        st.session_state.target_tactic = round(pot["tactic"] - MARGIN_TARGET, 1) if pot else None
        st.session_state.target_fitness = round(pot["fitness"] - MARGIN_TARGET, 1) if pot else None

        if pot is None:
            st.error(
                "Pemain yang tersisa tidak cukup untuk membentuk formasi apa pun. "
                "Kembali ke **Talent Scouting** dan bersihkan data tanpa membuang terlalu banyak pemain."
            )
        else:
            st.info(
                f"🎯 **Target ⭐ section ini:** Tactic Score **≥ {st.session_state.target_tactic}**. "
                f"(Di Tier 1 dan 2, Tactic Score = rata-rata rating Starting XI + {BONUS_FORMASI} jika formasi sesuai.)"
            )

        if terkunci():
            st.caption("Musim sudah selesai. Bagian ini hanya bisa dibaca.")
            tampilkan_hasil("s2")
        elif pot is not None:
            st.subheader("1) Pilih Formasi")
            formation = st.selectbox("Formasi:", list(FORMATIONS.keys()), key="s2_formation")
            req = FORMATIONS[formation]
            st.write(f"Kebutuhan pemain: {req}")

            st.subheader("2) Pilih 11 Pemain Starting XI")
            chosen_names = st.multiselect(
                "Pilih pemain (harus sesuai jumlah kebutuhan formasi):",
                clean_df["name"].tolist(),
                key="s2_squad",
            )
            squad_df = clean_df[clean_df["name"].isin(chosen_names)].copy()

            counts = squad_df["position"].value_counts().to_dict()
            ok = all(counts.get(pos, 0) == need for pos, need in req.items()) and len(squad_df) == 11
            if chosen_names:
                st.write(f"Komposisi saat ini: {counts} (total {len(squad_df)} pemain)")
                if ok:
                    st.success("Susunan pemain sudah sesuai formasi! ✅")
                else:
                    st.warning("Susunan pemain belum sesuai kebutuhan formasi di atas.")

            st.divider()
            st.subheader("3) Analisis Squad (EDA)")

            e1, e2, e3 = st.tabs(["🟢 Tier 1: Statistik Cepat", "🟡 Tier 2: Lengkapi Kode", "🔴 Tier 3: Kode Bebas"])

            with e1:
                if not ok:
                    st.info("Lengkapi susunan 11 pemain yang sesuai formasi dulu di atas.")
                else:
                    if st.button("📈 Hitung Rata-rata Rating per Posisi", key="tier1_eda_btn"):
                        stats = squad_df.groupby("position")["rating"].mean().round(1)
                        selesai_s2(
                            squad_df, formation, compute_tactic_score(squad_df, ok),
                            tabel=[("Starting XI", squad_df[["name", "position", *ATTR, "rating"]])],
                            chart=stats,
                        )

            with e2:
                st.markdown("Lengkapi kode untuk menghitung rata-rata rating per posisi.")
                st.code(
                    "stats = squad_df.groupby(____BLANK1____)[____BLANK2____].mean()\n"
                    "tactic_score = squad_df['rating'].mean()",
                    language="python",
                )
                b1 = st.text_input("BLANK1 (kolom untuk mengelompokkan):", placeholder="contoh: 'position'", key="s2_t2_b1")
                b2 = st.text_input("BLANK2 (kolom yang dirata-ratakan):", placeholder="contoh: 'rating'", key="s2_t2_b2")
                if st.button("▶️ Jalankan Kode", key="tier2_eda_btn"):
                    if not ok:
                        st.warning("Susunan pemain belum sesuai formasi.")
                    elif not b1 or not b2:
                        st.warning("Isi dulu semua blank.")
                    else:
                        code = (
                            f"stats = squad_df.groupby({b1})[{b2}].mean()\n"
                            "tactic_score = squad_df['rating'].mean()"
                        )
                        result, err = run_student_code(code, {"squad_df": squad_df})
                        if err:
                            st.error(err)
                        else:
                            selesai_s2(
                                squad_df, formation, compute_tactic_score(squad_df, ok),
                                tabel=[("Hasil kodemu (rata-rata per kelompok)", result["stats"].round(1))],
                            )

            with e3:
                st.markdown(
                    "Tulis kode Python bebas untuk menganalisis `squad_df` "
                    "(misalnya korelasi antar atribut). Simpan skor taktik akhir "
                    "ke variabel **`tactic_score`** (semakin tinggi semakin bagus, maksimal 100)."
                )
                default_code = (
                    "# squad_df sudah tersedia\n"
                    "corr = squad_df[['pace', 'technique', 'mental', 'rating']].corr()\n"
                    "avg_rating = squad_df['rating'].mean()\n"
                    f"# Petunjuk: di Tier 1 dan 2, skor taktik = avg_rating + {BONUS_FORMASI} kalau formasi sesuai.\n"
                    "# TODO: kombinasikan avg_rating dengan analisis lain jadi tactic_score\n"
                    "tactic_score = avg_rating\n"
                )
                student_code = st.text_area("Kode kamu:", value=default_code, height=220, key="tier3_eda_code")
                if st.button("▶️ Jalankan Kode", key="tier3_eda_btn"):
                    if not ok:
                        st.warning("Susunan pemain belum sesuai formasi.")
                    else:
                        result, err = run_student_code(student_code, {"squad_df": squad_df})
                        if err:
                            st.error(err)
                        elif "tactic_score" not in result:
                            st.error("Variabel `tactic_score` belum didefinisikan di kodemu.")
                        else:
                            tabel = []
                            if "corr" in result:
                                tabel.append(("Korelasi antar atribut", result["corr"].round(2)))
                            selesai_s2(squad_df, formation, result["tactic_score"], tabel=tabel)

            tampilkan_hasil("s2")

# =============================================================================
# SECTION 3 — TRAINING (FEATURE ENGINEERING)
# =============================================================================

def apply_training(sq_df, choices, boost=BOOST_LATIHAN):
    df = sq_df.copy()
    for idx, row in df.iterrows():
        col = choices[row["name"]]
        df.loc[idx, col] = df.loc[idx, col] + boost
    df["fitness_index"] = df[ATTR].mean(axis=1).round(1)
    return df


def selesai_s3(squad_df, trained_df):
    err = validasi_training(squad_df, trained_df)
    if err:
        st.error(err)
        return
    s = st.session_state
    team_fitness = round(float(trained_df["fitness_index"].mean()), 1)
    s.trained_squad_df = trained_df
    s.team_fitness = team_fitness
    s.section3_done = True

    pesan = [("success", "Latihan selesai! Lanjut ke tab Match.")]
    if s.target_fitness is not None:
        if team_fitness >= s.target_fitness:
            pesan.append(("success", f"⭐ Target tercapai: Team Fitness {team_fitness} ≥ {s.target_fitness}."))
        else:
            pesan.append((
                "warning",
                f"Target ⭐ belum tercapai: Team Fitness {team_fitness}, target ≥ {s.target_fitness}. "
                "Kebugaran tim dipengaruhi atribut pemain yang kamu pilih di Starting XI.",
            ))
    simpan_hasil("s3", pesan=pesan, metrics={"Team Fitness": team_fitness}, tabel=[("Skuad setelah latihan", trained_df)])
    st.rerun()


with tab3:
    st.header("🏋️ Training — Feature Engineering")

    if not st.session_state.section2_done:
        st.warning("Selesaikan dulu section **Squad & Tactics** untuk membuka bagian ini.")
    else:
        st.info(
            f"🎯 **Target ⭐ section ini:** Team Fitness **≥ {st.session_state.target_fitness}**. "
            "`fitness_index` = rata-rata pace, technique, dan mental seorang pemain; "
            "Team Fitness = rata-rata `fitness_index` seluruh Starting XI."
        )

        if terkunci():
            st.caption("Musim sudah selesai. Bagian ini hanya bisa dibaca.")
            tampilkan_hasil("s3")
        else:
            squad_df = st.session_state.squad_df.copy().reset_index(drop=True)
            st.markdown(
                f"Pilih fokus latihan untuk tiap pemain. Setiap sesi latihan menambah "
                f"**+{BOOST_LATIHAN}** pada atribut terkait. Latihan yang tepat sasaran akan meningkatkan performa tim!"
            )

            focus_map_col = {"Fisik (pace)": "pace", "Teknik (technique)": "technique", "Mental (mental)": "mental"}
            training_choices = {}
            cols = st.columns(2)
            for i, row in squad_df.iterrows():
                with cols[i % 2]:
                    choice = st.selectbox(
                        f"{row['name']} ({row['position']})",
                        list(focus_map_col.keys()),
                        key=f"train_{row['name']}",
                    )
                    training_choices[row["name"]] = focus_map_col[choice]

            f1, f2, f3 = st.tabs(["🟢 Tier 1: Terapkan Langsung", "🟡 Tier 2: Lengkapi Kode", "🔴 Tier 3: Kode Bebas"])

            with f1:
                if st.button("💪 Jalankan Sesi Latihan", key="tier1_train_btn"):
                    selesai_s3(squad_df, apply_training(squad_df, training_choices))

            with f2:
                st.markdown("Lengkapi kode untuk menerapkan efek latihan ke tiap pemain.")
                st.code(
                    "df = squad_df.copy()\n"
                    "for idx, row in df.iterrows():\n"
                    "    col = training_choices[row['name']]\n"
                    "    df.loc[idx, col] = df.loc[idx, col] + ____BLANK1____\n"
                    "df['fitness_index'] = df[['pace', 'technique', 'mental']].mean(axis=1)\n"
                    "trained_df = df",
                    language="python",
                )
                b1 = st.text_input(
                    "BLANK1 (besar peningkatan tiap sesi latihan):", placeholder="contoh: 5", key="s3_t2_b1"
                )
                if st.button("▶️ Jalankan Kode", key="tier2_train_btn"):
                    if not b1:
                        st.warning("Isi dulu blank-nya.")
                    else:
                        code = (
                            "df = squad_df.copy()\n"
                            "for idx, row in df.iterrows():\n"
                            "    col = training_choices[row['name']]\n"
                            f"    df.loc[idx, col] = df.loc[idx, col] + {b1}\n"
                            "df['fitness_index'] = df[['pace', 'technique', 'mental']].mean(axis=1)\n"
                            "trained_df = df"
                        )
                        result, err = run_student_code(
                            code, {"squad_df": squad_df, "training_choices": training_choices}
                        )
                        if err:
                            st.error(err)
                        else:
                            selesai_s3(squad_df, result.get("trained_df"))

            with f3:
                st.markdown(
                    "Tulis kode Python bebas untuk menerapkan `training_choices` ke `squad_df`, "
                    "lalu buat fitur baru **`fitness_index`**. Simpan hasil akhir ke variabel "
                    "**`trained_df`**."
                )
                default_code = (
                    "# squad_df dan training_choices sudah tersedia\n"
                    "df = squad_df.copy()\n"
                    "for idx, row in df.iterrows():\n"
                    "    col = training_choices[row['name']]\n"
                    f"    df.loc[idx, col] = df.loc[idx, col] + {BOOST_LATIHAN}\n"
                    "# TODO: buat kolom fitness_index dari pace, technique, mental\n"
                    "trained_df = df\n"
                )
                student_code = st.text_area("Kode kamu:", value=default_code, height=240, key="tier3_train_code")
                if st.button("▶️ Jalankan Kode", key="tier3_train_btn"):
                    result, err = run_student_code(
                        student_code, {"squad_df": squad_df, "training_choices": training_choices}
                    )
                    if err:
                        st.error(err)
                    else:
                        selesai_s3(squad_df, result.get("trained_df"))

            tampilkan_hasil("s3")

# =============================================================================
# SECTION 4 — MATCH (MODELING)
# =============================================================================

def play_match(model, team_rating, tactic_score, team_fitness, opp_rating):
    rating_diff = team_rating - opp_rating
    feat = pd.DataFrame([{
        "team_rating": team_rating,
        "tactic_score": tactic_score,
        "team_fitness": team_fitness,
        "rating_diff": rating_diff,
    }])
    win_prob = model.predict_proba(feat)[0][1]
    roll = random.random()
    if win_prob >= 0.55:
        outcome = "Menang" if roll < win_prob else ("Seri" if roll < win_prob + 0.15 else "Kalah")
    elif win_prob <= 0.45:
        outcome = "Kalah" if roll < (1 - win_prob) else ("Seri" if roll < (1 - win_prob) + 0.15 else "Menang")
    else:
        outcome = "Seri" if roll < 0.4 else ("Menang" if roll < 0.7 else "Kalah")
    return win_prob, outcome


def selesai_s4(model, acc):
    """Mainkan pertandingan resmi dengan model siswa. Hanya berlaku sekali per musim."""
    s = st.session_state
    try:
        acc = float(acc)
    except (TypeError, ValueError):
        st.error("`accuracy` harus berupa angka.")
        return
    if not (0 <= acc <= 1):
        st.error("`accuracy` harus berupa angka antara 0 dan 1.")
        return
    if not hasattr(model, "predict_proba"):
        st.error("`model` harus punya method `predict_proba` (contoh: LogisticRegression atau RandomForestClassifier).")
        return

    team_rating = s.trained_squad_df["rating"].mean()
    try:
        win_prob, outcome = play_match(model, team_rating, s.tactic_score, s.team_fitness, s.opponent["rating"])
    except Exception:
        st.error(
            "Model tidak cocok dengan fitur pertandingan. Latih modelmu dengan kolom "
            "`team_rating`, `tactic_score`, `team_fitness`, dan `rating_diff`."
        )
        return

    s.model_accuracy = round(acc, 2)
    s.win_prob = float(win_prob)
    s.match_result = outcome
    s.section4_done = True

    pesan = []
    if s.model_accuracy >= TARGET_AKURASI:
        pesan.append(("success", f"⭐ Target tercapai: akurasi model {acc:.0%} ≥ {TARGET_AKURASI:.0%}."))
    else:
        pesan.append((
            "warning",
            f"Target ⭐ belum tercapai: akurasi model {acc:.0%}, target ≥ {TARGET_AKURASI:.0%}. "
            "Musim berikutnya, coba model atau pembagian data yang berbeda.",
        ))
    pesan.append(("markdown", f"### Hasil Pertandingan: **{outcome}**"))
    simpan_hasil(
        "s4", pesan=pesan,
        metrics={"Akurasi Model (data uji)": f"{acc:.0%}", "Probabilitas Menang": f"{win_prob:.0%}"},
    )
    st.rerun()


with tab4:
    st.header("🏟️ Match — Prediksi dengan Machine Learning")

    if not st.session_state.section3_done:
        st.warning("Selesaikan dulu section **Training** untuk membuka bagian ini.")
    else:
        st.info(
            f"🎯 **Tujuan:** kalahkan **{opponent['name']}** (rating {opponent['rating']}). "
            f"**Target ⭐ section ini:** akurasi model di data uji **≥ {TARGET_AKURASI:.0%}**. "
            "Pertandingan hanya dimainkan **sekali**, jadi pastikan modelmu sudah baik sebelum menekan tombol."
        )

        if terkunci():
            st.success("Pertandingan musim ini sudah dimainkan. Lihat tab **Season Report**, atau mulai musim baru.")
            tampilkan_hasil("s4")
        else:
            st.subheader("Data Historis (untuk melatih model)")
            hist_df = generate_historical_matches()
            st.caption("Data pertandingan musim-musim sebelumnya, dipakai untuk melatih model prediksi menang/kalah.")
            tampil_df(hist_df.head(8))

            feature_cols = ["team_rating", "tactic_score", "team_fitness", "rating_diff"]
            X = hist_df[feature_cols]
            y = hist_df["win"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            m1, m2, m3 = st.tabs(["🟢 Tier 1: Pilih Model", "🟡 Tier 2: Lengkapi Kode", "🔴 Tier 3: Kode Bebas"])

            with m1:
                model_choice = st.selectbox(
                    "Pilih algoritma model:", ["Logistic Regression", "Random Forest"], key="s4_t1_model"
                )
                if st.button("🤖 Latih Model & Mainkan Pertandingan", key="tier1_match_btn"):
                    model = LogisticRegression() if model_choice == "Logistic Regression" else RandomForestClassifier(
                        n_estimators=100, random_state=1
                    )
                    model.fit(X_train, y_train)
                    acc = accuracy_score(y_test, model.predict(X_test))
                    selesai_s4(model, acc)

            with m2:
                st.markdown("Lengkapi kode untuk melatih model dan mengevaluasi akurasinya.")
                st.code(
                    "model = ____BLANK1____\n"
                    "model.fit(X_train, y_train)\n"
                    "preds = model.predict(X_test)\n"
                    "accuracy = accuracy_score(y_test, preds)",
                    language="python",
                )
                b1 = st.text_input(
                    "BLANK1 (buat objek modelnya):",
                    placeholder="contoh: LogisticRegression() atau RandomForestClassifier()",
                    key="s4_t2_b1",
                )
                if st.button("▶️ Jalankan Kode", key="tier2_match_btn"):
                    if not b1:
                        st.warning("Isi dulu blank-nya.")
                    else:
                        code = (
                            f"model = {b1}\n"
                            "model.fit(X_train, y_train)\n"
                            "preds = model.predict(X_test)\n"
                            "accuracy = accuracy_score(y_test, preds)"
                        )
                        result, err = run_student_code(
                            code, {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
                        )
                        if err:
                            st.error(err)
                        else:
                            selesai_s4(result["model"], result["accuracy"])

            with m3:
                st.markdown(
                    "Tulis kode Python bebas: lakukan train/test split sendiri (atau pakai `X_train`/`X_test` "
                    "yang sudah tersedia), latih model apa pun dari scikit-learn, lalu simpan objeknya ke "
                    "variabel **`model`** dan akurasinya ke **`accuracy`**."
                )
                default_code = (
                    "# X (fitur) dan y (label 'win') dari data historis tersedia,\n"
                    "# begitu juga X_train, X_test, y_train, y_test (split bawaan game).\n"
                    "# Petunjuk: akurasi bisa berubah-ubah tergantung cara membagi data\n"
                    "# (test_size, random_state). Menurutmu kenapa?\n"
                    "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7)\n"
                    "model = LogisticRegression()\n"
                    "model.fit(X_tr, y_tr)\n"
                    "preds = model.predict(X_te)\n"
                    "accuracy = accuracy_score(y_te, preds)\n"
                )
                student_code = st.text_area("Kode kamu:", value=default_code, height=260, key="tier3_match_code")
                if st.button("▶️ Jalankan Kode", key="tier3_match_btn"):
                    result, err = run_student_code(
                        student_code,
                        {"X": X, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
                    )
                    if err:
                        st.error(err)
                    elif "model" not in result or "accuracy" not in result:
                        st.error("Pastikan variabel `model` dan `accuracy` ada di kodemu.")
                    else:
                        selesai_s4(result["model"], result["accuracy"])

# =============================================================================
# SEASON REPORT
# =============================================================================
with tab5:
    st.header("🏆 Season Report")

    if not st.session_state.section4_done:
        st.info("Selesaikan keempat section dulu untuk melihat laporan musim lengkap.")
    else:
        s = st.session_state
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Formasi", s.formation)
        c2.metric("Tactic Score", s.tactic_score)
        c3.metric("Team Fitness", s.team_fitness)
        c4.metric("Akurasi Model", f"{s.model_accuracy:.0%}")

        st.subheader(f"Hasil Akhir vs {opponent['name']}: {s.match_result}")
        st.caption(f"Peluang menang menurut modelmu: {s.win_prob:.0%}")

        result_msg = {
            "Menang": "🎉 Selamat! Kerja data science-mu membuahkan kemenangan.",
            "Seri": "🤝 Hasil imbang — masih ada ruang untuk perbaikan di section sebelumnya.",
            "Kalah": "😔 Kalah kali ini. Coba tinjau ulang keputusan cleaning, taktik, atau training kamu. "
                     "Ingat juga: pertandingan mengandung unsur peluang, jadi hasil yang bagus tidak selalu menang.",
        }
        st.write(result_msg.get(s.match_result, ""))

        st.divider()
        st.markdown("#### ⭐ Target musim")
        bintang = hitung_bintang()
        st.markdown(
            f"- {'⭐' if bintang['data'] else '⬜'} **Data:** semua {s.n_unik} pemain unik dipertahankan "
            f"(tersisa {len(s.clean_df)})\n"
            f"- {'⭐' if bintang['taktik'] else '⬜'} **Taktik:** Tactic Score {s.tactic_score} (target ≥ {s.target_tactic})\n"
            f"- {'⭐' if bintang['fitness'] else '⬜'} **Fitness:** Team Fitness {s.team_fitness} (target ≥ {s.target_fitness})\n"
            f"- {'⭐' if bintang['model'] else '⬜'} **Model:** akurasi {s.model_accuracy:.0%} (target ≥ {TARGET_AKURASI:.0%})"
        )

        n_bintang = sum(bintang.values())
        poin_hasil = POIN_HASIL.get(s.match_result, 0)
        total = n_bintang + poin_hasil
        k1, k2, k3 = st.columns(3)
        k1.metric("Bintang", f"{n_bintang} / {MAKS_BINTANG}")
        k2.metric("Poin hasil", f"{poin_hasil} / {max(POIN_HASIL.values())}")
        k3.metric("Skor musim", f"{total} / {MAKS_SKOR}")
        st.success(f"Peringkat musim ini: **{peringkat(total)}**")

        st.divider()
        st.markdown(
            "**Refleksi:** Bagian mana yang menurutmu paling menentukan hasil akhir — "
            "data cleaning yang bersih, pilihan taktik, latihan yang tepat sasaran, "
            "atau pilihan model machine learning? Diskusikan dengan temanmu!"
        )

        st.button("🔄 Main Lagi (Musim Baru)", on_click=reset_season, key="btn_main_lagi")

# =============================================================================
# SIDEBAR (di akhir script supaya selalu menampilkan progres terbaru)
# =============================================================================
with st.sidebar:
    st.header("Progres Musim")
    _bintang = hitung_bintang()
    for _label, _done, _star in [
        ("1. Talent Scouting", st.session_state.section1_done, _bintang["data"]),
        ("2. Squad & Tactics", st.session_state.section2_done, _bintang["taktik"]),
        ("3. Training", st.session_state.section3_done, _bintang["fitness"]),
        ("4. Match", st.session_state.section4_done, _bintang["model"]),
    ]:
        st.write(("✅ " if _done else "⬜ ") + _label + (" ⭐" if _star else ""))
    st.caption(f"⭐ {sum(_bintang.values())} / {MAKS_BINTANG} bintang")
    st.caption(f"Lawan: {opponent['name']} (rating {opponent['rating']})")

    st.divider()
    st.button("🔄 Mulai Musim Baru", on_click=reset_season, key="btn_reset_sidebar")
