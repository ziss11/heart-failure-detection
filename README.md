# Submission 2: Heart Failure Prediction

Nama: Abdul Azis

Username dicoding: zizz1181

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| Masalah | Gagal jantung adalah kondisi ketika jantung melemah sehingga tidak mampu memompa darah yang cukup ke seluruh tubuh. Kondisi ini dapat terjadi pada siapa saja, tetapi lebih sering terjadi pada orang berusia di atas 65 tahun. Penyakit gagal jantung ini merupakan salah satu penyakit yang mematikan dan kematian yang disebabkan oleh penyakit ini pada umumnya masih termasuk yang tertinggi saat ini.|
| Solusi machine learning | Dari permasalahan diatas dengan memanfaatkan teknologi, machine learning menjadi salah satu solusi untuk membantu mengurangi tingkat kematian yang cukup tinggi akibat penyakit ini. Dengan sebuah sistem prediksi penyakit gagal jantung, diharapkan para tenaga medis maupun masyarakat dapat terbantu untuk dapat mendeteksi penyakit ini lebih awal. |
| Metode pengolahan | Data yang digunakan pada proyek ini terdapat dua tipe data, yaitu data kategorikal dan numerik. Metode yang digunakan untuk mengelolah data tersebut yaitu mentransformasikan data kategorikal menjadi bentuk one-hot encoding dan menormalisasikan data numerik kedalam range data yang sama.  |
| Arsitektur model | Model yang dibangun cukup sederhana hanya menggunakan Dense layer dan Dropout layer sebagai hidden layer pada model neural network dan memiliki 1 output layer |
| Metrik evaluasi | Metric yang digunakan pada model yaitu AUC, Precision, Recall, BinaryAccuracy, TruePositive, FalsePositive, TrueNegative, FalseNegative untuk mengevaluasi performa model sebuah klasifikasi |
| Performa model | Model yang dibuat menghasilkan performa yang cukup baik dalam memberikan sebuah prediksi dan dari pelatihan yang dilakukan menghasilkan binary_accuracy sebesar 87% dan val_binary_acuracy sebesar 85%, hasil seperti ini sudah cukup baik untuk sebuah sistem klasifikasi namun masih bisa ditingkatkan lagi  |
| Opsi deployment | Proyek machine learning ini dideploy menggunakan salah satu platfrom as a service yaitu HEROKU yang menyediakan layanan gratis untuk mendeploy sebuah proyek. |
| Web app | <https://hf-pred.herokuapp.com/v1/models/heart-failure-model> |
| Monitoring | Monitoring pada proyek ini dapat dilakukan dengan menggunakan layanan open-source yaitu prometheus. Contohnya setiap perubahan jumlah request yang dilakukan kepada sistem ini dapat dimonitoring dengan baik dan dapat menampilkan status dari setiap request yang diterima |
