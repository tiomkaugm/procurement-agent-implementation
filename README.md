# MARL Procurement

Implementasi sederhana dari laporan "Penerapan Multi-Agent Reinforcement Learning (MARL) untuk Proses Pengadaan Barang Perusahaan secara Otonom" (Kelompok 2, Agen Cerdas Enterprise, Magister Kecerdasan Artifisial UGM, 2026).

Sistem dimodelkan sebagai Dec-POMDP dengan empat agen (IRE, VMI, DA, SLM) dan arsitektur CTDE. Semua algoritma ditulis sendiri (tanpa RLlib atau sejenisnya) supaya bisa dijelaskan saat presentasi.

Kode ini melakukan empat hal:

1. Mereproduksi angka perhitungan Bab 6 laporan (dibuktikan oleh unit test).
2. Menyediakan environment PettingZoo AEC berbasis giliran.
3. Melatih dan membandingkan lima kebijakan: random, rule-based, independent Q-learning (IQL), CTDE actor-critic, dan rencana optimal satu putaran.
4. Menampilkan hasilnya di dashboard Streamlit.

## Install

Proyek memakai Python 3.14 (versi yang dipakai dan diuji, dengan `venv`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # atau: pip install -r requirements.txt
```

Dengan `uv`: `uv sync`.

Semua perintah di bawah dijalankan dari root repositori (folder yang berisi `pyproject.toml`).

## Cara menjalankan

| Perintah | Fungsi |
|---|---|
| `pytest` | Semua test (54 test, sekitar 30 detik) |
| `python scripts/run_report_trace.py` | Cetak jejak rule-based pada skenario laporan, dalam bahasa Indonesia |
| `python scripts/check_scenarios.py 500` | Cek generator skenario acak: persen layak, rencana naif gagal, konsensus rule-based dan random |
| `python scripts/train.py --algo iql` | Latih IQL (50.000 episode, sekitar 2 menit) ke `runs/iql/` |
| `python scripts/train.py --algo ctde` | Latih CTDE (50.000 episode, sekitar 2,5 menit) ke `runs/ctde/` |
| `python scripts/evaluate.py` | Tabel perbandingan lima kebijakan ke `runs/comparison.csv` dan `.md` (pencarian rencana optimal pertama kali sekitar 2 menit, lalu di-cache) |
| `python scripts/sensitivity.py` | Latih ulang IQL dan CTDE untuk 4 nilai peluang penerimaan vendor (sekitar 20 menit) ke `runs/sensitivity/` |
| `streamlit run app/streamlit_app.py` | Dashboard |

Opsi training: `--episodes N`, `--seed S`, `--out DIR`, `--override P` (p_accept dan p_termin dibuat tetap).

Tiap training menyimpan `curve.csv` (kurva), `checkpoint.*`, dan `config.json`.

## Ringkasan hasil

Angka acuan Bab 6 (biaya, diskon, K1 = 10.203.000, K2 = -20.397.000, total 100.397.000, skor komposit 6,60 / 7,19 / 5,41, dan J(pi) = 150 / -10) semuanya cocok, dan rule-based pada fixture menghasilkan jejak laporan lalu berakhir tanpa konsensus setelah 6 putaran.

Perbandingan kebijakan pada 500 skenario acak (seed 0-499, kebijakan greedy):

| Kebijakan | Return tim | Konsensus | Pel. anggaran | Pel. kas | Unit mendesak | Putaran |
|---|---|---|---|---|---|---|
| Random | -0,09 | 77,0% | 17,8% | 3,2% | 90,2% | 2,87 |
| Rule-based | -1,08 | 64,0% | 14,2% | 9,8% | 84,0% | 2,88 |
| IQL | 1,19 | 79,6% | 18,6% | 2,4% | 99,4% | 2,17 |
| CTDE actor-critic | 1,71 | 80,4% | 19,2% | 0,4% | 100,0% | 2,04 |
| Rencana optimal satu putaran | 0,47 | 86,2% | 11,8% | 0,6% | 98,2% | 1,71 |

Tabel lengkap (biaya dan rasio) ada di `runs/comparison.md`. Untuk seed lain, return IQL 1,32 dan 1,15, CTDE 1,80 dan 1,74.

Cara membaca hasil:

- CTDE terbaik di antara kebijakan hasil latihan pada return, pelanggaran kas, dan pemenuhan unit mendesak. Tingkat konsensusnya hanya sedikit di atas IQL dan random.
- **Rasio terhadap rencana optimal satu putaran boleh > 1 dan bukan bukti kebijakan lebih baik.** Reward per agen diberikan di setiap putaran, sehingga episode gagal (6 putaran) mengumpulkan reward positif yang menutup sebagian penalti tim. Return episode gagal: CTDE -0,62, IQL -2,52, random -6,84. Rencana optimal menilai kegagalan berhenti di satu putaran. Tingkat konsensus lebih jujur: rencana optimal 86,2% dibanding CTDE 80,4%. Ini batasan desain reward yang diketahui.
- Rencana optimal satu putaran bukan batas atas. Ia hanya merencanakan satu putaran, sehingga kebijakan multi-putaran bisa melampauinya.
- Sensitivitas terhadap peluang penerimaan vendor (0,3 / 0,5 / 0,7 / 0,9): urutan kebijakan tidak berubah. CTDE tidak pernah memilih penawaran_balik, dan pemakaian penawaran_balik serta revisi_termin tidak naik seiring peluang. Kesimpulan tidak bergantung pada asumsi peluang, tetapi kedua aksi negosiasi hampir tidak memberi sinyal belajar (keuntungan penawaran balik hanya sekitar 0,6% harga).
- Hiperparameter tidak dituning, dan tiap konfigurasi hanya dilatih dengan satu sampai tiga seed.

## Pemetaan laporan ke kode

| Bagian laporan | Isi | File |
|---|---|---|
| 5.3 | Tiga aksi per agen | `env.py` (`IRE_ACTIONS`, `DA_ACTIONS`, `SLM_ACTIONS`, mask vendor) |
| 5.4.1 | Observasi diskret 4 variabel x 5 level | `env.py` (`discrete_obs`), `agents/iql.py` |
| 5.4.2 | Giliran, satu agen aktif per langkah | `env.py` (PettingZoo AEC) |
| 6.1 | Data skenario | `config/report_scenario.yaml`, `scenario.py` |
| 6.2 | Skor komposit vendor | `scenario.py` (`composite_scores`, `eligible_vendors`) |
| 6.3.1 | Vendor cadangan berbeda untuk batch kedua | `agents/rule_based.py` |
| 6.4 | Negosiasi DA dan diskon bayar cepat | `env.py` (`_step_da`, `_step_slm`), `costs.py` |
| 6.5 | Biaya, kas, dan jejak putaran koordinasi | `costs.py`, `scripts/run_report_trace.py`, `tests/test_rule_based_trace.py` |
| 6.6 | Objektif kolektif J(pi) | `rewards.py` (`collective_objective`) |
| Bab 5 (CTDE) | Actor lokal dan critic terpusat | `agents/ctde_ac.py` |

Bab 6 lain tidak punya kode: baseline klasifikasi dan NLP kontrak (lihat Future work).

## Struktur folder

```
config/        report_scenario.yaml (fixture), env_default.yaml (randomisasi, peluang, reward)
src/procurement_marl/
  scenario.py  costs.py  rewards.py  env.py  oracle.py  evaluate.py
  agents/      random_agent.py  rule_based.py  iql.py  ctde_ac.py
scripts/       train.py  evaluate.py  sensitivity.py  check_scenarios.py  run_report_trace.py
app/           streamlit_app.py
tests/         test_costs_report.py  test_env_api.py  test_rule_based_trace.py
               test_iql.py  test_ctde.py  test_dashboard.py
runs/          kurva training, checkpoint, tabel perbandingan, sensitivitas
DECISIONS.md   setiap interpretasi atas laporan dan asumsi kami
```

## Asumsi dan keputusan

Semua interpretasi atas laporan dan angka yang tidak ada di laporan (harga lantai A dan C, penawaran awal C, reputasi vendor hasil hitung balik, peluang penerimaan vendor, rentang skenario acak, desain reward) ada di [`DECISIONS.md`](DECISIONS.md), dan nilainya ada di file YAML dengan komentar asumsi. Nilai reputasi vendor perlu dikonfirmasi ke tim.

## Future work

- Baseline klasifikasi ML (Logistic Regression, SVM, Random Forest, XGBoost) dari awal Bab 6, karena tidak ada datanya.
- Pemahaman dokumen kontrak dan komunikasi antar-agen berbasis bahasa alami.
- Integrasi LLM untuk membaca permintaan teks bebas dan menjelaskan hasil episode (dibatalkan, tidak dikerjakan).
- Memperbaiki desain reward supaya episode gagal tidak bisa menutup penalti tim lewat reward per putaran, lalu mengulang training dan evaluasi.
- Rencana optimal multi-putaran yang benar-benar batas atas.
