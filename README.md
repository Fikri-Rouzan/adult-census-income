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
   - Simpan file `.gitignore`.
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
