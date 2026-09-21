### Perbedaan alur environment dan kebijakan agen

Environment menyediakan pilihan tindakan, sedangkan kebijakan menentukan pilihan yang diambil. Kebijakan `Rule-based (meniru laporan)` sengaja menjalankan aturan tetap: IRE membagi pesanan setelah putaran pertama, VMI mengutamakan pemasok berbeda untuk batch berikutnya, DA selalu memilih penawaran awal, dan SLM selalu membayar cepat. Kebijakan ini tidak mencari penyesuaian baru berdasarkan jenis konflik.

Itulah penyebab putaran 3–6 pada ekspor fixture berulang dengan biaya Rp100.397.000, kelebihan anggaran Rp397.000, dan kas negatif. Pengulangan tersebut bahkan dipersyaratkan oleh [`test_rounds_3_to_6_repeat_and_end_without_consensus`](tests/test_rule_based_trace.py). Perilaku ini tidak boleh digeneralisasi sebagai perilaku seluruh kebijakan IQL atau CTDE. Random memilih aksi valid secara acak, sedangkan IQL dan CTDE memakai kebijakan hasil latihan.

## Bug dan rencana perbaikan

**Status: daftar berikut adalah hasil pemeriksaan dan rencana perubahan, belum merupakan perbaikan yang sudah diterapkan.** Bug yang berhasil direproduksi dibedakan dari keterbatasan desain, asumsi data, dan peningkatan penjelasan. Mode reproduksi laporan perlu tetap tersedia agar perubahan kebijakan tidak menghilangkan jejak acuan yang sudah diuji.

### 1. Prioritas tinggi: konsensus dapat lolos meskipun kapasitas pemasok tidak cukup

- **Status:** bug validasi terkonfirmasi melalui pengujian tambahan.
- **Penyebab:** ketika seluruh pemasok tersaring, `_mask("VMI")` membuka kembali semua pilihan dengan `[True] * 3`. Pemeriksaan akhir `_end_round` tidak memeriksa ulang kapasitas dan kelayakan skor pemasok.
- **Reproduksi:** gunakan fixture dengan kapasitas semua pemasok 100 unit dan kas awal Rp250 juta. Pilih `teruskan`, pemasok B, `penawaran_awal`, lalu `bayar_cepat`. Pesanan 1.000 unit dinyatakan konsensus tanpa pelanggaran meskipun kapasitas B hanya 100 unit.
- **Rencana:** tangani kondisi tanpa pemasok layak secara eksplisit; jangan membuka semua pilihan secara otomatis. Validasi batasan pemasok kembali sebelum menyatakan konsensus. Tinjau juga fallback `eligible_vendors` yang mempertahankan satu pemasok ketika semua skor di bawah ambang, agar aturan wajib dan pengecualiannya konsisten.
- **Kriteria selesai:** kasus kapasitas tidak cukup tidak menghasilkan konsensus; kondisi semua pemasok gugur selesai dengan alasan yang jelas tanpa loop atau kegagalan pemilihan aksi. Tambahkan tes regresi untuk kedua kondisi tersebut.

### 2. Prioritas menengah: putaran berulang tanpa perubahan rencana

- **Status:** keterbatasan desain kebijakan rule-based, bukan kesalahan reproduksi laporan. Informasi konflik sudah diteruskan oleh ENV, tetapi kebijakan ini tidak memakainya untuk mengubah strategi.
- **Rencana:** pertahankan baseline reproduksi laporan dan sediakan kebijakan adaptif terpisah yang menanggapi konflik dengan alternatif pemasok, negosiasi, atau pembayaran yang relevan. Tambahkan penghentian dengan alasan `tidak ada perbaikan` ketika rencana dan keadaan yang relevan tidak berubah.
- **Kriteria selesai:** pengulangan deterministik tidak menghabiskan putaran tanpa penjelasan. Untuk kebijakan stokastik, keputusan berhenti juga mempertimbangkan perubahan harga, relasi, dan peluang hasil berikutnya; tindakan sama belum tentu berarti keadaan sama.

### 3. Prioritas menengah: label klarifikasi tidak sesuai proses yang dilakukan

- **Status:** masalah penamaan dan penyederhanaan model. `minta_klarifikasi` saat ini hanya membagi kebutuhan mendesak dan nonmendesak.
- **Rencana:** perjelas label tampilan sebagai pembagian atau revisi kebutuhan. Jika klarifikasi sungguhan diperlukan, tambahkan keadaan menunggu informasi dan jawaban yang benar-benar memengaruhi rencana. Pertahankan pemetaan aksi dan kompatibilitas checkpoint saat mengubah label.
- **Kriteria selesai:** dashboard dan ekspor menjelaskan tindakan yang benar-benar terjadi; tidak mengesankan ada percakapan klarifikasi yang belum diimplementasikan.

### 4. Prioritas menengah: ketidaklayakan kas belum dijelaskan sejak awal

- **Status:** keterbatasan diagnosis skenario, bukan kesalahan aritmetika biaya.
- **Bukti fixture:** kas awal Rp150 juta dikurangi kebutuhan lain Rp70 juta menyisakan Rp80 juta, tanpa pemasukan selama empat bulan. Biaya terendah secara optimistis untuk 1.000 unit adalah Rp99.220.000, memakai harga lantai B dan diskon cepat. Seluruh opsi pembayaran dalam model jatuh dalam horizon empat bulan, sehingga masih ada kekurangan sedikitnya Rp19.220.000. Memasukkan C atau menggeser pembayaran saja tidak mengatasi kekurangan total tersebut.
- **Rencana:** tambahkan pemeriksaan awal yang dapat membuktikan ketidaklayakan dari batas bawah biaya dan kas, beserta penjelasan kebutuhan perubahan pendanaan, jumlah, atau batasan. Untuk kasus yang belum dapat dibuktikan, gunakan status `belum ditemukan solusi`, bukan langsung `tidak mungkin`.
- **Kriteria selesai:** fixture menampilkan alasan kekurangan kas dengan angka yang dapat ditelusuri. Simulasi reproduksi tetap dapat dijalankan untuk kebutuhan demonstrasi.

### 5. Tinjauan aturan: penyaringan pemasok C dan dasar skor komposit

- **Status:** C gugur sesuai aturan 5,41 < 6,00; belum ditemukan bug perhitungan skor pada fixture. Namun, nilai reputasi dihitung balik agar cocok dengan skor laporan, sebagaimana dicatat di `DECISIONS.md`.
- **Rencana:** validasi sumber reputasi, bobot, dan ambang. Bedakan syarat wajib dari preferensi peringkat: kandidat tidak boleh diloloskan hanya agar hasil menjadi sukses. Tampilkan komponen skor dan alasan penolakan; jika ada beberapa pelanggaran, tampilkan semuanya, bukan hanya alasan pertama yang diperiksa.
- **Kriteria selesai:** pengecualian C dapat dijelaskan dari data yang disepakati. Perubahan ambang atau aturan diuji terhadap kapasitas, tenggat, biaya, dan kas, bukan hanya jumlah pemasok yang lolos.

### 6. Tinjauan model: reward berulang dapat mengaburkan kegagalan

- **Status:** keterbatasan desain yang sudah didokumentasikan pada bagian hasil dan `DECISIONS.md`, bukan temuan bahwa kebijakan rule-based belajar mengeksploitasi reward.
- **Rencana:** evaluasi desain reward yang tidak memberi keuntungan dari pengulangan tanpa kemajuan, serta samakan dasar perbandingan horizon dengan rencana satu putaran. Jika desain diubah, lakukan pelatihan dan evaluasi ulang, lalu bedakan hasil baru dari checkpoint dan hasil lama.
- **Kriteria selesai:** perbandingan melaporkan konsensus, pelanggaran, dan biaya selain return. Return yang lebih tinggi pada episode gagal tidak ditafsirkan sebagai solusi pengadaan yang lebih baik.