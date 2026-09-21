# DECISIONS.md

Catatan setiap interpretasi atas laporan dan asumsi kami sendiri.

## Inkonsistensi laporan
1. Nama agen: hanya empat, `IRE`, `VMI`, `DA`, `SLM`. Di laporan "PRA" = IRE, "SSA" = VMI, "PMI" = VMI.
2. Skor komposit 6.2 (A 6,60; B 7,19; C 5,41): reputasi tidak ada di laporan. Nilai dihitung balik: A 7,00; B 8,45; C 2,55 (`report_scenario.yaml`, konfirmasi ke tim).
3. 6.6: teks menyebut IRE mendapat -100, tabel menunjukkan SLM. Kami ikuti tabel.
4. Data kas (kas awal 150 jt, kebutuhan lain 40 jt + 30 jt, batas minimum 60 jt) muncul di 6.5 tanpa disebut di 6.1. Dimasukkan ke skenario di YAML.
5. Tabel reward 6.6 ilustratif. Hanya dipakai untuk menguji `collective_objective`; reward environment didesain sendiri.

## Asumsi angka yang tidak ada di laporan
- Harga lantai A (95.000) dan C (108.000), serta harga penawaran awal C (110.000).
- Horizon kas 4 bulan; kebutuhan lain bulan 2-4 = 0; arus kas masuk = 0 (fixture).
- Batas minimum kas hanya dilaporkan pada fixture (`enforce_min_cash: false`).
- Rentang randomisasi training (anggaran per unit, kas awal, kebutuhan lain, arus masuk) di `env_default.yaml`.

## Skor komposit
- Sub-skor 0-10. Waktu = `10 * (1 - lead_time / tenggat)` di-clip [0,10]. Harga = min-max terhadap harga list semua vendor.
- Skor dihitung untuk semua vendor sebelum masking kapasitas, jadi nilainya tetap 6,60 / 7,19 / 5,41.
- Vendor gugur jika skor < 6,0. Jika semua gugur, satu vendor terbaik dipertahankan.

## Parameter stokastik (tidak ada di laporan)
- Peluang diterima bergantung pada skor relasi vendor (petunjuk 6.4: harga B tidak ditekan ke 89.000 demi hubungan jangka panjang).
- `p_accept = clip(0,2 + 0,6 x relasi)`, `p_termin = clip(0,3 + 0,6 x relasi)`. Relasi fixture 0,6.
- Penolakan menurunkan relasi (0,2 untuk penawaran balik, 0,1 untuk termin); di bawah 0,3 vendor mundur.
- Override tetap tersedia (`p_accept_override`, `p_termin_override`) untuk test dan analisis sensitivitas.

## Penyederhanaan
- Revisi termin: di laporan diteruskan ke DA; di sini menjadi keputusan SLM dengan peluang diterima.
- Pembayaran: transport dan risiko dibayar di bulan batch; hanya harga barang yang digeser oleh cara bayar. Revisi termin: 50% bulan berikutnya, 50% dua bulan setelah batch (ganjil dibulatkan ke bawah untuk termin pertama).
- Uang disimpan sebagai `int` rupiah; diskon disimpan sebagai persen bulat dan dibulatkan ke rupiah terdekat (setengah naik) dengan aritmetika integer.

## Reward (desain sendiri)
- Dibuat sederhana, per agen mengukur hal yang menjadi tugasnya, dengan rentang kira-kira -2 sampai +1 agar seimbang antar agen.
- IRE: unit mendesak tepat waktu; VMI: kualitas dan biaya relatif; DA: penghematan dan kegagalan negosiasi; SLM: diskon dan kas; komponen tim: anggaran dan konsensus.
- Bobot `w_i` dan semua koefisien ada di `env_default.yaml`.

## Lingkungan kerja
- Tanpa git atas instruksi pengguna, jadi tidak ada commit.
- Versi Python diganti ke 3.14 (keputusan pengguna, menggantikan 3.11 di spesifikasi awal), karena hanya versi itu yang terpasang dan seluruh pengembangan, torch, dan test berjalan di sana. `pyproject.toml`: `requires-python = ">=3.14"`. `uv` tidak terpasang di mesin ini, jadi instalasi lewat `pip` di venv.
- Istilah "putaran": di environment satu putaran = satu lintasan penuh IRE sampai SLM; "6 putaran" di laporan 6.5 adalah langkah koordinasi.

## Keputusan Fase 2
- Estimasi VMI: satu aturan, yaitu jumlah unit x (harga terakhir yang disepakati dengan vendor itu di episode ini, atau harga list jika belum ada, + transport + risiko). Aturan ini menghasilkan tiga angka laporan (102 jt, 71,05 jt, 32,1 jt) dan dipakai juga oleh rule-based untuk memilih vendor. Alasannya: laporan memakai dasar harga yang berbeda-beda untuk estimasi, dan aturan tunggal lebih mudah dijelaskan. Reward VMI tetap memakai biaya pada harga list sebagai acuan tetap.
- Rencana optimal satu putaran (`oracle.py`): enumerasi penuh 6 putaran meledak (2^12 cabang per rencana), jadi rencana hanya untuk satu putaran, dan rencana yang berakhir konflik dinilai sebagai episode yang berhenti di situ dengan penalti tim. Kebijakan multi-putaran bisa melampauinya (misalnya pulih setelah penawaran ditolak), sehingga ini bukan batas atas, dan rasio > 1 diizinkan di evaluasi. Nama di kode, metrik, dan dashboard: "rencana optimal satu putaran".
- Penawaran balik maksimal sekali per batch (di-mask setelah dipakai, termasuk setelah vendor mundur atau minta alternatif). Vendor hanya mundur jika masih ada vendor lain yang bisa mengambil batch (mencegah loop tanpa akhir). Pengaman tambahan: `max_steps` 300 memicu truncation; 10.000 episode acak seed tetap selalu selesai di bawah batas.
- Batasan lead time hanya untuk batch bulan 1, karena hanya batch itu yang memuat unit mendesak (batch bulan 2 tidak punya tenggat 10 hari).
- Warning `api_test` "observasi berbeda ukuran antar agen" diterima: ukuran vektor observasi memang berbeda per agen karena observasi parsial.
- Rentang generator (asumsi kami, mencakup nilai fixture): anggaran per unit 95-125 rb, kas awal 125-230 jt, kebutuhan lain bulan 1 0-80 jt, arus masuk bulan 2-4 0-80 jt. Hasil 500 seed (0-499): layak 83,8%; rencana naif gagal di 43,4%; konsensus rule-based 64,0%; konsensus random 77,4%. Random mengungguli rule-based karena rule-based mengulang keputusan yang sama di semua putaran, sedangkan random berpindah rencana.

## Keputusan Fase 3 (IQL)
- Transisi per agen: karena agen bergiliran, transisi sebuah agen berjalan dari gilirannya ke gilirannya berikutnya. Reward = jumlah reward bersama yang diterima di antaranya (tanpa diskon di dalam jeda), state berikut = observasi diskret pada giliran berikutnya. Update terminal tanpa bootstrap.
- Reward bersama = jumlah terbobot reward semua agen pada langkah itu (`w_i` dari YAML).
- Bootstrap hanya melewati aksi valid (max Q di atas aksi yang tidak di-mask).
- Hiperparameter (asumsi kami, tanpa tuning): alpha 0,1, gamma 0,99, epsilon 1,0 turun linear ke 0,05 pada 70% episode, 50.000 episode (sekitar 2 menit di CPU).
- Seed: skenario training memakai offset 1.000.000 supaya tidak beririsan dengan skenario evaluasi (seed 0-499).
- Kurva training di `runs/iql/curve.csv` adalah rata-rata blok 500 episode saat berlatih (dengan eksplorasi), bukan evaluasi greedy.
- Hasil (evaluasi greedy pada 500 skenario, seed 0-499): IQL return tim 1,19 (seed 0), 1,32 (seed 1), 1,15 (seed 2), konsensus 79,6% / 81,4% / 80,6%. Pembanding: random -0,09 dan 77,0%; rule-based -1,08 dan 64,0%. IQL unggul jelas pada return tetapi hanya sedikit pada konsensus dibanding random.

## Keputusan Fase 4 (CTDE dan evaluasi)
- CTDE: 4 actor MLP terpisah (parameter tidak dibagi, 64 unit x 2 lapis, tanh), satu critic V(s) MLP (128 x 2 lapis) pada state penuh, hanya dipakai saat training. Action masking di logit. Advantage GAE (gamma 0,99, lambda 0,95) atas urutan giliran gabungan, dengan reward langkah = jumlah terbobot reward semua agen (sama dengan IQL). Adam lr 1e-3, entropy 0,01, clip gradien 0,5, satu update per 32 episode, 50.000 episode (sekitar 2,5 menit CPU). Hiperparameter tanpa tuning.
- Evaluasi: 500 skenario `sample_scenario(seed)` seed 0-499 untuk semua kebijakan, kebijakan greedy. Pelanggaran dihitung dari cek batasan putaran terakhir. Baris rencana optimal satu putaran: return = nilai harapan satu putaran, kolom lain dari menjalankan rencana itu di environment. Rasio = return tim / return rencana optimal (rata-rata), boleh > 1.
- Hasil (seed 0): return tim Random -0,09; Rule-based -1,08; IQL 1,19; CTDE 1,71; rencana optimal satu putaran 0,47. Konsensus 77,0 / 64,0 / 79,6 / 80,4 / 86,2%. Seed lain: IQL 1,32 dan 1,15; CTDE 1,80 dan 1,74 (konsensus CTDE 80,4% dan 77,6%).
- Temuan penting: rasio > 1 sebagian besar artefak reward. Reward per agen diberikan di setiap putaran, sehingga episode gagal (enam putaran) menerima sekitar enam kali reward positif per putaran untuk menutup penalti tim -3 x 4 agen. Return episode gagal: CTDE -0,62, IQL -2,52, random -6,84, rule-based -6,71. Rencana optimal satu putaran menilai kegagalan berhenti di putaran itu (sekitar -10), jadi tidak sebanding. Dengan konsensus, kebijakan hasil latihan hanya 80% dibanding 86% untuk rencana optimal.
- Sensitivitas (`scripts/sensitivity.py`, p_accept = p_termin tetap pada 0,3 / 0,5 / 0,7 / 0,9, seed 0): CTDE tidak pernah memilih penawaran_balik (0%) dan return-nya stabil (1,70-1,84); pemakaian revisi_termin CTDE 42-100% tanpa pola monoton. IQL memakai penawaran_balik 26-58% dan revisi_termin 45-61% tanpa pola monoton terhadap p. Urutan return IQL < CTDE dan keduanya di atas random dan rule-based tetap di semua nilai p. Kesimpulan: hasil tidak bergantung pada asumsi peluang, tetapi kedua aksi negosiasi nyaris tidak memberi sinyal yang bisa dipelajari (keuntungan penawaran_balik hanya sekitar 0,6% harga).

## Keputusan Fase 5 (dashboard) dan desain reward
- Reward per putaran dipertahankan (opsi 3, keputusan pengguna). Batasan yang diketahui: episode gagal mengumpulkan reward positif per putaran yang menutup sebagian penalti tim, sehingga return CTDE/IQL tidak sebanding langsung dengan rencana optimal satu putaran (rasio > 1 bukan bukti kebijakan lebih baik). Tingkat konsensus adalah ukuran yang lebih jujur. Batasan ini ditampilkan di dashboard (tab Perbandingan) dan wajib disebut di README.
- Dashboard (`app/streamlit_app.py`): episode dijalankan penuh lalu dijelajahi langkah demi langkah (slider dan tombol) dari log terstruktur environment. Kas per bulan dihitung dari batch yang sudah dibayar sampai langkah terpilih. Warna: biru dan oranye (slot 1 dan 2 palet kategorikal) untuk IQL dan CTDE. Belum dicek visual di browser; hanya dites lewat `streamlit.testing`.

## Fase 7 dibatalkan
- Integrasi Gemini (intake permintaan dan penjelas episode) tidak dikerjakan, atas keputusan pengguna. Spesifikasi, README, `pyproject.toml`, dan `requirements.txt` sudah dibersihkan dari rujukannya. `.env.example` dihapus; `.gitignore` dipertahankan.

## Perbaikan validasi (butir 1, 3, 4 di bug.md)
- Skor komposit adalah syarat wajib. Sebelumnya, ketika satu-satunya vendor lolos-skor kapasitasnya kurang, mask VMI dibuka penuh dan rencana bisa memilih vendor yang gugur skor (mis. C) dan tetap konsensus. Sekarang mask hanya membuka satu vendor (kapasitas terbesar), putaran berakhir konflik dengan alasan `tanpa_pemasok_layak`. Fallback "pertahankan satu vendor jika semua di bawah ambang" tetap sesuai spesifikasi dan kini tercatat di log VMI (`score_fallback`).
- Akibatnya porsi skenario layak turun (83,4% menjadi 71,6% pada rentang lama), karena angka lama ikut dihitung dari rencana yang meloloskan vendor gugur skor. Rentang disetel ulang: `deadline_days` [6, 14] (sebelumnya [5, 14]) dan `urgent_share` [0,5, 0,8] (sebelumnya [0,5, 0,9]). Hasil di 500 seed: layak 83,0%, rencana naif gagal 49,6%. Nilai fixture (10 hari, 70%) tetap tercakup.
- Diagnosis kas (`cash_diagnosis`): batas bawah biaya = seluruh permintaan di vendor termurah pada harga lantai dengan diskon bayar cepat, tanpa memperhitungkan kapasitas dan lead time, sehingga benar-benar batas bawah. Semua pembayaran jatuh dalam horizon 4 bulan, jadi jika kas horizon lebih kecil dari batas bawah, skenario pasti tanpa konsensus. Jika tidak terlampaui, hasilnya "solusi belum ditemukan", bukan bukti ketidaklayakan.
- `minta_klarifikasi` hanya diganti labelnya di tampilan ("bagi pesanan"); nama aksi internal tidak berubah.
- Hasil training di `runs/` dan angka di README dihasilkan sebelum perubahan ini dan belum dilatih ulang.
- Hasil setelah perbaikan ini (seed 0, 500 skenario): return tim Random -0,18; Rule-based 0,11; IQL 1,06; CTDE 1,93; rencana optimal satu putaran 0,44. Konsensus 78,2 / 76,2 / 81,0 / 78,0 / 84,6%. Return episode gagal: CTDE -4,95, IQL -6,64, random -8,75, rule-based -7,03 (tidak lagi positif seperti sebelum skor menjadi syarat wajib). Sensitivitas: CTDE tidak konsisten di atas IQL (return 0,84 / 2,14 / 2,11 / 2,36 lawan 1,03 / 1,42 / 0,94 / 1,54). Kesimpulan yang lebih hati-hati daripada hasil sebelumnya: kebijakan hasil latihan hanya sedikit lebih baik dari random pada konsensus, dan keunggulan CTDE atas IQL tidak terbukti.
- Tiga seed training (0, 1, 2; evaluasi pada 500 skenario yang sama): return IQL 1,06 / 1,39 / 1,01 dan CTDE 1,93 / 2,23 / 0,81; konsensus IQL 81,0 / 81,2 / 77,8% dan CTDE 78,0 / 78,0 / 82,2%. Variasi antar seed sebesar selisih IQL lawan CTDE, jadi tidak ada klaim bahwa salah satu lebih baik. Checkpoint seed 1 dan 2 di `runs/seeds/`.
