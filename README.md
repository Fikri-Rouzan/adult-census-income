# Adult Census Income

## 📌 Deskripsi

---

## 💾 Dataset

---

## 🛠️ Tech Stack

| Kategori                    | Teknologi yang Digunakan                                                                                                                 |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| 🌐 **Programming Language** | `Python`                                                                                                                                 |
| 🌱 **Environment**          | `Jupyter Notebook`                                                                                                                       |
| 🧩 **Framework**            | `TensorFlow`                                                                                                                             |
| ⚛️ **Libraries**            | `TFX`, `TensorFlow Transform`, `KerasTuner`, `TensorFlow Model Analysis`,<br>`TensorFlow Serving`, `python-dotenv`, `Requests`, `Pylint` |
| ⚡ **Tools**                | `GitHub Codespaces`, `docker`                                                                                                            |

---

## ⚙️ Petunjuk Pengaturan

1. **Prasyarat**
   - Git terinstal di komputer.
   - Akun [GitHub](https://github.com) aktif.
   - Akses ke GitHub Codespaces.
   - Akun [Railway](https://railway.app) aktif.

2. **Fork Repositori**
   - Buka repositori [Adult Census Income](https://github.com/Fikri-Rouzan/adult-census-income).
   - Klik tombol **Fork** di pojok kanan atas untuk menyalin repositori ini ke akun GitHub kamu.

3. **Menjalankan di GitHub Codespaces**
   - Buka repositori hasil fork di akun GitHub kamu.
   - Klik tombol **Code**, lalu pilih tab **Codespaces**.
   - Klik **Create codespace on main**.
   - Tunggu hingga environment selesai dimuat. Konfigurasi Dev Container akan secara otomatis menyiapkan Python 3.10, docker, ekstensi VS Code, serta memasang seluruh pustaka di `requirements.txt`.

4. **Menjalankan Machine Learning Pipeline**
   - Buka direktori `notebooks/`, lalu buka file `pipeline.ipynb`.
   - Pastikan kamu memilih kernel **venv (Python 3.10)** di pojok kanan atas editor.
   - Jalankan seluruh sel kode (Run All).

5. **Melakukan Commit Artefak Pipeline**
   - Buka file `.gitignore` di direktori root proyek.
   - Cari dan hapus dua baris berikut:
     ```text
     main-pipeline
     serving_model
     ```
   - Buka terminal di Codespaces lalu jalankan perintah berikut:
     ```bash
     git add .
     git commit -m "feat: include pipeline artifacts and serving model"
     git push origin main
     ```

6. **Melakukan Deployment dengan Railway**
   - Buka website [Railway](https://railway.com).
   - Klik tombol **Sign in**, lalu pilih **Continue with GitHub**.
   - Setelah masuk ke dashboard, pilih menu **Projects** di panel sebelah kiri, lalu klik tombol **New**.
   - Masukkan atau paste link repositori GitHub yang sudah kamu fork, lalu klik **Deploy Repo**.
   - Tunggu hingga proses build selesai, lalu klik card proyek yang muncul di tengah layar.
   - Masuk ke tab **Settings**, scroll ke bawah hingga bagian **Networking**, lalu klik **Generate Domain**.
   - Pilih port **8051**, lalu klik **Generate Domain**.
   - Salin URL yang diberikan untuk digunakan pada fase konfigurasi berikutnya.

7. **Konfigurasi Variabel Environment**

```bash
cp .env.example .env
```

- Buka file `.env` lalu sesuaikan nilai variabel berikut
  ```env
  RAILWAY_URL="YOUR_RAILWAY_URL"
  ```

8. **Menjalankan Prediksi**
   - Buka kembali direktori `notebooks/`, lalu buka file `testing.ipynb`.
   - Pastikan memilih kernel **venv (Python 3.10)** di pojok kanan atas editor.
   - Jalankan seluruh sel kode (Run All) untuk menguji prediksi ke layanan yang sedang berjalan.

9. **Konfigurasi Prometheus Monitoring**
   - Buka file `monitoring/prometheus.yml`.
   - Cari bagian `static_configs` dan ganti `<your-railway-domain>` dengan nama domain Railway yang kamu dapatkan
     ```yaml
     static_configs:
       - targets: ["<your-railway-domain>.up.railway.app:443"]
     ```

10. **Membuat dan Menjalankan Container Prometheus & Grafana**

```bash
# Buat jaringan docker untuk Prometheus dan Grafana
docker network create mlops-net
```

- Pindah ke direktori `monitoring`, buat image docker Prometheus, dan jalankan container
  ```bash
  cd monitoring
  docker build -t adult-income-prometheus .
  docker run -d -p 9090:9090 --network mlops-net --name prometheus adult-income-prometheus
  ```
- Jalankan container Grafana
  ```bash
  docker run -d -p 3000:3000 --network mlops-net --name grafana grafana/grafana
  ```

11. **Mengakses Monitoring Prometheus**
    - Buka tab **Ports** di panel sebelah bawah/kanan terminal GitHub Codespaces.
    - Klik **Forward a Port**, lalu masukkan port `9090`.
    - Buka URL dari forwarded address yang terbentuk di browser untuk mengakses antarmuka Prometheus.
    - Pilih menu **Status** pada bagian atas, lalu pilih **Target health** untuk memeriksa status health endpoint.

12. **Mengakses Dashboard Grafana**
    - Buka kembali tab **Ports**, klik **Add Port**, lalu masukkan port `3000`.
    - Buka URL dari forwarded address yang terbentuk di browser untuk masuk ke Grafana.
    - Pada tampilan login, masukkan `admin` sebagai **Username** dan **Password**, lalu klik **Log in**.
    - Ketika diminta untuk memperbarui password, klik **Skip**.
    - Setelah masuk ke dashboard, buka menu **Connections** > **Data sources**, lalu klik **Add data source**.
    - Pilih **Prometheus**, kemudian pada bagian **Connection** masukkan URL berikut:
      ```text
      http://prometheus:9090
      ```
    - Scroll ke bagian paling bawah, lalu klik **Save & test**.
    - Jika verifikasi berhasil, kamu sudah siap membuat dan mengonfigurasi dashboard visualisasi di Grafana.
