# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech (Human Resources)

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang bergerak di bidang edutech dan telah berdiri sejak tahun 2000. Perusahaan ini memiliki lebih dari 1.000 karyawan yang tersebar di seluruh Indonesia. Meskipun tergolong sebagai perusahaan besar, Jaya Jaya Maju kesulitan dalam mengelola karyawan, yang berimbas pada tingginya tingkat **attrition rate** (tingkat karyawan yang keluar/resign) hingga mencapai lebih dari 10%.

### Permasalahan Bisnis

- Attrition rate perusahaan Jaya Jaya Maju berada di atas 10%, yang tergolong tinggi dan berdampak pada meningkatnya biaya rekrutmen, pelatihan ulang, serta hilangnya produktivitas akibat kekosongan posisi.
- Departemen HR belum memiliki visibilitas yang jelas mengenai faktor-faktor apa saja yang paling berkontribusi terhadap keputusan karyawan untuk resign, sehingga upaya retensi yang dilakukan masih bersifat reaktif dan tidak tepat sasaran.
- Jika permasalahan ini terus dibiarkan, risiko jangka panjang yang dihadapi perusahaan antara lain: penurunan kualitas layanan akibat hilangnya karyawan berpengalaman, meningkatnya beban kerja tim yang tersisa, meningkatnya biaya operasional HR, serta menurunnya daya saing perusahaan dalam mempertahankan talenta terbaik di industri edutech.
- Diperlukan sebuah pendekatan berbasis data untuk mengidentifikasi karyawan yang berisiko tinggi resign, sehingga HR dapat melakukan intervensi retensi secara proaktif dan terukur.

### Cakupan Proyek

1. **Data Understanding** — mengeksplorasi struktur data karyawan (1.470 baris, 35 kolom), termasuk mengidentifikasi 412 baris data karyawan yang belum memiliki label attrition.
2. **Exploratory Data Analysis (EDA)** — menganalisis pola dan hubungan antara atribut karyawan (overtime, job role, income, tenure, status pernikahan, dsb.) dengan attrition.
3. **Data Preparation** — membersihkan dan mentransformasi data (encoding kategori, scaling) agar siap digunakan untuk pemodelan.
4. **Modeling** — membangun model machine learning (Random Forest Classifier) untuk memprediksi kemungkinan seorang karyawan resign, serta menerapkannya pada 412 data karyawan yang belum berlabel.
5. **Evaluation** — mengevaluasi performa model menggunakan metrik accuracy, precision, recall, f1-score, dan ROC-AUC, serta menganalisis feature importance untuk mengetahui faktor paling berpengaruh.
6. **Business Dashboard** — membangun dashboard interaktif berbasis Streamlit agar departemen HR dapat memonitor attrition rate dan faktor-faktor pendorongnya secara mandiri.
7. Menyusun kesimpulan dan rekomendasi action item bagi departemen HR.

### Persiapan

Sumber data: dataset karyawan (`employee_data.csv`) yang disediakan oleh Dicoding untuk submission proyek "Menyelesaikan Permasalahan Human Resources", yang merupakan adaptasi dari dataset publik **IBM HR Analytics Employee Attrition & Performance** (1.470 baris, 35 kolom — atribut seperti Age, Department, JobRole, MonthlyIncome, OverTime, dsb., dengan target `Attrition`). Dataset publik acuan tersebut dapat diakses melalui: https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee

Setup environment:

```
python -m venv env
env\Scripts\activate      # Windows
source env/bin/activate   # Mac/Linux

pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Business Dashboard

Business dashboard dibangun menggunakan **Streamlit** (`dashboard.py`), yang menampilkan:

- KPI utama: total karyawan, attrition rate, jumlah resign, dan rata-rata monthly income (dapat difilter berdasarkan Department dan Job Role).
- Visualisasi attrition rate berdasarkan Department, Job Role, OverTime, Status Pernikahan, dan Business Travel.
- Perbandingan Monthly Income dan Years At Company antara karyawan yang bertahan dan yang resign.
- Grafik **feature importance** dari model machine learning, yang menunjukkan faktor-faktor paling berpengaruh terhadap attrition.
- Daftar karyawan (dari 412 data yang sebelumnya belum berlabel) dengan probabilitas resign tertinggi, hasil prediksi model, sebagai referensi prioritas program retensi.

Dashboard ini memanfaatkan artefak yang dihasilkan dari notebook (`model_attrition.joblib`, `scaler.joblib`, `encoders.joblib`, `feature_columns.joblib`, `predicted_attrition.csv`) sehingga tidak perlu melatih ulang model saat dashboard dijalankan.

Cara menjalankan dashboard secara lokal:

```
pip install -r requirements.txt
streamlit run dashboard.py
```

Setelah perintah di atas dijalankan, dashboard akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

(Opsional) Jika dashboard sudah di-deploy secara online (misalnya melalui Streamlit Community Cloud), cantumkan tautan aksesnya di sini: `<tautan dashboard>`.

## Conclusion

Berdasarkan hasil analisis dan pemodelan, attrition rate karyawan Jaya Jaya Maju pada data historis berada di sekitar 17%, dengan faktor-faktor yang paling berpengaruh terhadap keputusan karyawan untuk resign antara lain: **OverTime**, **MonthlyIncome**, **Age**, **TotalWorkingYears**, **JobRole** (khususnya Sales Representative dan Laboratory Technician), **DistanceFromHome**, **JobLevel**, **StockOptionLevel**, **YearsAtCompany**, dan **JobSatisfaction**.

Model Random Forest yang dibangun mampu memprediksi kemungkinan attrition dengan performa yang baik dan telah diterapkan pada 412 karyawan yang sebelumnya belum memiliki label attrition, sehingga menghasilkan daftar karyawan dengan risiko resign tertinggi yang dapat menjadi prioritas bagi program retensi HR.

### Rekomendasi Action Items (Optional)

- Evaluasi kebijakan lembur (overtime): karyawan dengan overtime tinggi memiliki attrition rate jauh lebih besar (~32% vs ~11%). Pertimbangkan redistribusi beban kerja atau kompensasi tambahan yang lebih baik untuk mengurangi kelelahan kerja.
- Tinjau kembali struktur gaji dan jenjang karir pada job role dengan attrition tertinggi (Sales Representative, Laboratory Technician), termasuk peluang promosi dan stock option bagi karyawan level bawah.
- Rancang program retensi khusus untuk karyawan dengan masa kerja singkat (early tenure) dan karyawan berstatus single, yang secara historis lebih rentan resign.
- Gunakan hasil prediksi model (daftar karyawan berisiko tinggi pada dashboard) sebagai dasar untuk melakukan sesi 1-on-1 atau survei kepuasan kerja secara proaktif sebelum karyawan tersebut benar-benar mengajukan resign.
- Pantau dashboard secara rutin (bulanan/kuartalan) untuk memantau tren attrition rate dan efektivitas program retensi yang telah dijalankan.
