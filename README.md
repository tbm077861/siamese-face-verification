# Xác minh khuôn mặt bằng mạng nơ-ron Siamese

Dự án khoa học máy tính về bài toán **xác minh khuôn mặt one-shot** (*one-shot face
verification*): cho hai ảnh, xác định chúng có thuộc về **cùng một người** hay không.

Mô hình là một **mạng Siamese** theo Koch và cộng sự (2015). Hai nhánh mạng tích chập
*dùng chung trọng số* ánh xạ mỗi ảnh thành một vector đặc trưng 4096 chiều; sau đó một lớp
sigmoid học trên **khoảng cách L1** giữa hai vector để cho ra xác suất "cùng người". Cơ sở
lý thuyết và phương pháp thực nghiệm được trình bày trong
[`docs/phuong_phap.md`](docs/phuong_phap.md).

> **Phạm vi.** Từ phiên bản 2.0, kho mã chỉ gồm phần khoa học máy tính: dữ liệu, mô hình,
> huấn luyện, đánh giá và suy luận. Phần ứng dụng (giao diện Kivy, dashboard, tải ảnh và
> tài liệu) đã được gỡ bỏ; mã cũ vẫn xem được trong lịch sử git ở commit `b3f15cf` ("final").

## Điểm chính

- **Mô hình:** mạng Siamese 38.964.545 tham số, đầu vào ảnh RGB 100×100; số chiều vector
  nhúng có thể tuỳ chỉnh.
- **Hai chế độ thực nghiệm:**
  - `lfw`: học từ các cặp cùng người / khác người trong bộ LFW, kiểm thử trên những người
    chưa từng thấy (one-shot đúng nghĩa);
  - `personal`: cá nhân hoá với ảnh *anchor*/*positive* tự thu bằng webcam và ảnh
    *negative* từ LFW.
- **Thực nghiệm không rò rỉ dữ liệu, tái lập được:** chia train/val/test ở mức ảnh *trước
  khi* ghép cặp; ảnh negative chia theo danh tính; seed cố định; danh sách chia được lưu
  cùng mô hình.
- **Hai cách tiền xử lý:** cắt vùng giữa ảnh (`center`), hoặc phát hiện rồi cắt khuôn mặt
  bằng Haar cascade (`face`). Cách đã chọn được ghi vào cấu hình của lần chạy, nên huấn
  luyện, đánh giá và suy luận luôn dùng cùng một cách.
- **Đánh giá theo chuẩn sinh trắc học:** Accuracy, Precision, Recall, F1, FAR, FRR,
  ROC-AUC, EER. Ngưỡng quyết định được chọn trên tập val rồi mới áp dụng cho tập test.
- **Đo ở miền triển khai và so sánh có kiểm định:**
  - `field-test` đo mô hình trên ảnh tự chụp, so với người lạ chưa từng thấy;
  - `compare` so sánh hai lần chạy trên cùng tập test, kèm khoảng tin cậy bootstrap theo
    danh tính.
- **Kiểm thử tự động:** `pytest` cho độ đo, xử lý dữ liệu, mô hình và toàn bộ quy trình
  huấn luyện → đánh giá → suy luận.

## Cấu trúc thư mục

```
.
├── siamese_face/          # Gói Python chứa toàn bộ mã nguồn
│   ├── config.py          # Hằng số, siêu tham số (TrainConfig)
│   ├── data.py            # Liệt kê ảnh, chia train/val/test, ghép cặp
│   ├── pipeline.py        # Tiền xử lý ảnh, tăng cường dữ liệu, tf.data
│   ├── faces.py           # Phát hiện và cắt khuôn mặt (Haar cascade, OpenCV)
│   ├── model.py           # Lớp L1Dist, mạng nhúng, mạng Siamese
│   ├── train.py           # Huấn luyện, dừng sớm, lưu kết quả mỗi lần chạy
│   ├── metrics.py         # Độ đo (NumPy): ma trận nhầm lẫn, ROC, AUC, EER, …
│   ├── evaluate.py        # Giao thức đánh giá: chọn ngưỡng trên val, đo trên test
│   ├── verify.py          # Suy luận: xác minh một ảnh với các ảnh tham chiếu
│   ├── field_test.py      # Đánh giá trên ảnh tự chụp: chủ thể và người lạ so với ảnh tham chiếu
│   ├── compare.py         # So sánh hai lần chạy trên cùng tập test (bootstrap)
│   ├── plots.py           # Biểu đồ huấn luyện, ROC, phân bố điểm
│   ├── collect.py         # Thu thập ảnh anchor/positive bằng webcam
│   ├── negatives.py       # Chuẩn bị ảnh negative (LFW)
│   └── cli.py             # Giao diện dòng lệnh: python -m siamese_face …
├── tests/                 # Kiểm thử pytest
├── docs/
│   ├── phuong_phap.md     # Cơ sở lý thuyết và phương pháp thực nghiệm
│   ├── ket_qua/           # Số liệu và biểu đồ gốc của các lần huấn luyện đã báo cáo
│   └── nhat_ky/           # Nhật ký dự án: thay đổi và kết luận rút ra theo từng phiên
├── data/                  # Dữ liệu (xem data/README.md)
├── models/                # Nơi đặt mô hình đã huấn luyện (không đưa lên git)
├── runs/                  # Kết quả mỗi lần huấn luyện (tự sinh, không đưa lên git)
├── requirements.txt
└── pyproject.toml
```

## Cài đặt

Cần Python 3.9 trở lên, trong số các phiên bản TensorFlow hỗ trợ (khuyến nghị 3.10 hoặc 3.11).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                    # kiểm tra cài đặt
```

### Huấn luyện bằng GPU

Mô hình được viết bằng Keras 3 nên chạy được trên ba backend: TensorFlow (mặc định),
PyTorch và JAX. Chọn backend bằng `--backend` hoặc biến môi trường `KERAS_BACKEND`.
`tf.data` chỉ dùng để đọc ảnh trên CPU, nên với mọi backend vẫn cần cài TensorFlow.

Trên Windows, TensorFlow ≥ 2.11 không hỗ trợ GPU trên Windows gốc. Ngoài ra, TensorFlow 2.20
cũng chưa chạy được trên GPU RTX 50-series (Blackwell, `sm_120`), báo lỗi
`CUDA_ERROR_INVALID_PTX`. Cách đã kiểm chứng trên RTX 5070 Laptop là dùng **WSL2 + backend
PyTorch**:

```bash
python3 -m venv ~/venvs/appface
~/venvs/appface/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/venvs/appface/bin/pip install "tensorflow-cpu" "keras>=3" opencv-python-headless matplotlib pytest
cd /mnt/d/GithubProject/AppFace
~/venvs/appface/bin/python -m siamese_face --backend torch train --mode lfw --augment --batch-size 32
```

Mô hình `.keras` huấn luyện bằng PyTorch vẫn đánh giá và xác minh được bằng backend
TensorFlow trên Windows, và ngược lại.

## Quy trình thực nghiệm

Mọi bước đều chạy qua một giao diện dòng lệnh duy nhất: `python -m siamese_face <lệnh>`.
Xem tuỳ chọn của từng lệnh bằng `python -m siamese_face <lệnh> --help`.

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Ảnh LFW | `prepare-negatives <lfw.tgz \| thư mục \| URL>` | `data/negative/` |
| 2. Ảnh của bạn (chỉ chế độ `personal`) | `collect` | `data/anchor/`, `data/positive/`, `data/verification/` |
| 3. Huấn luyện | `train --mode lfw` hoặc `train --mode personal` | `runs/<thời điểm>/` |
| 4. Đánh giá | `evaluate --run runs/<thời điểm>` | `metrics.json` và biểu đồ |
| 5. Xác minh | `verify --model <mô hình> --image <ảnh>` | khớp / không khớp |
| 6. Đo trên ảnh tự chụp | `field-test --model <mô hình>` | TAR/FAR theo ngưỡng, ROC-AUC |
| 7. So sánh hai lần chạy | `compare runs/<A> runs/<B>` | hiệu số và khoảng tin cậy 95% |
| – | `summary` | kiến trúc và số tham số |

### 1. Chuẩn bị ảnh negative

```powershell
python -m siamese_face prepare-negatives D:\Downloads\lfw.tgz
```

Nguồn có thể là file nén, thư mục đã giải nén hoặc URL. Xem [`data/README.md`](data/README.md)
để biết nơi tải bộ LFW.

### 2. Thu thập ảnh anchor và positive (chỉ cho chế độ `personal`)

```powershell
python -m siamese_face collect
```

Cửa sổ camera hiển thị vùng cắt 250×250 ở giữa khung hình. Các phím tắt:

- `a`: lưu ảnh anchor;
- `p`: lưu ảnh positive;
- `v`: lưu ảnh tham chiếu dùng khi xác minh;
- `q`: thoát.

Nên thu khoảng 300 ảnh mỗi loại, với nhiều góc mặt, điều kiện ánh sáng và biểu cảm khác nhau.

### 3. Huấn luyện

```powershell
python -m siamese_face train --mode lfw --augment        # mô hình tổng quát (chỉ cần LFW)
python -m siamese_face train --mode personal --augment   # mô hình cá nhân hoá (cần bước 2)
python -m siamese_face train --mode lfw --augment --preprocess face   # cắt khuôn mặt trước
```

Chế độ `lfw` chia dữ liệu theo người, nên tập test chỉ gồm những người mô hình chưa từng
thấy. Mô hình thu được có thể dùng ngay để xác minh bạn (bước 5) với vài ảnh tham chiếu,
không cần huấn luyện lại.

Mỗi lần chạy tạo một thư mục `runs/<YYYYmmdd-HHMMSS>/`:

| File | Nội dung |
|---|---|
| `model.keras` | mô hình với trọng số tốt nhất theo `val_loss` |
| `config.json` | siêu tham số của lần chạy |
| `splits.json` | danh sách cặp ảnh train/val/test |
| `history.csv`, `training_curves.png` | loss và các độ đo theo từng epoch |
| `training_summary.json` | epoch tốt nhất, thời gian huấn luyện |
| `metrics.json`, `test_scores.csv` | kết quả đánh giá trên tập test |
| `roc_curve.png`, `score_distribution.png` | đường ROC, phân bố điểm tương đồng |

Huấn luyện khoảng 39 triệu tham số trên CPU khá chậm. Nên dùng GPU, hoặc giảm `--epochs` khi
chỉ muốn chạy thử.

### 4. Đánh giá

```powershell
python -m siamese_face evaluate --run runs/20261109-170501 --criterion eer
```

`--criterion` chọn cách đặt ngưỡng trên tập val:

- `f1` (mặc định): F1 lớn nhất;
- `youden`: chỉ số J = TPR − FPR lớn nhất;
- `eer`: điểm mà FAR = FRR.

### 5. Xác minh một ảnh

```powershell
python -m siamese_face verify --model runs/20261109-170501/model.keras --image anh_moi.jpg
```

Ảnh được so với mọi ảnh tham chiếu trong `data/verification/`; có thể đổi thư mục bằng
`--references`. Một ảnh tham chiếu được tính là "khớp" khi điểm tương đồng đạt ít nhất
`--detection-threshold`. Kết luận "khớp" khi tỉ lệ ảnh tham chiếu khớp đạt ít nhất
`--verification-threshold`.

Nên đặt `--detection-threshold` bằng giá trị `threshold` trong `metrics.json` của lần chạy.

Với mô hình huấn luyện bằng `--preprocess face`, ảnh cần xác minh và ảnh tham chiếu cũng
được cắt khuôn mặt. Lệnh tự đọc cách tiền xử lý từ `config.json` nằm cạnh file mô hình; có
thể chỉ định rõ bằng `--preprocess`.

### 6. Đánh giá trên ảnh tự chụp

```powershell
python -m siamese_face field-test --model runs/<lần chạy>/model.keras --output ket_qua.json
```

Lệnh này đo mô hình ở đúng nơi sẽ dùng nó:
- So các ảnh thử của chủ thể (mặc định `data/anchor/`, chụp riêng với ảnh tham chiếu) và
  300 ảnh người lạ lấy ngẫu nhiên từ `data/negative/` với bộ ảnh tham chiếu trong
  `data/verification/`.
- Người lạ chỉ được lấy từ những danh tính *không* có trong tập train/val của mô hình (đọc
  `splits.json` của lần chạy).
- Kết quả gồm ROC-AUC và EER (theo từng cặp và theo từng ảnh), cùng bảng tỉ lệ chủ thể/người
  lạ được chấp nhận theo quy tắc của lệnh `verify` với nhiều giá trị `τ_d`.

### 7. So sánh hai lần chạy

```powershell
python -m siamese_face compare runs/<A> runs/<B> --output so_sanh.json
```

Lệnh đọc `test_scores.csv` của hai lần chạy và kiểm tra rằng chúng được chấm trên đúng cùng
các cặp. Sau đó nó báo cáo ROC-AUC, EER và TAR tại FAR 1% và 10% của từng mô hình, kèm hiệu
số B − A. Khoảng tin cậy 95% được tính bằng bootstrap theo cụm danh tính: lấy mẫu lại theo
người, vì các cặp của cùng một người không độc lập.

### Dùng lại mô hình cũ `siamesemodel.h5`

Nếu còn giữ file mô hình của phiên bản ứng dụng cũ, đặt nó vào `models/` rồi chạy:

```powershell
python -m siamese_face verify --model models/siamesemodel.h5 --image anh_moi.jpg
python -m siamese_face evaluate --model models/siamesemodel.h5 --output-dir runs/mo_hinh_cu
```

Lưu ý: nếu mô hình cũ đã được huấn luyện trên chính các ảnh trong `data/`, số liệu đánh giá
sẽ lạc quan hơn thực tế.

## Kết quả thực nghiệm

Kết quả trên **tập test gồm 758 người chưa từng thấy khi huấn luyện** (3.112 cặp, cân bằng),
chế độ `lfw`. Hai lần chạy chỉ khác cách tiền xử lý. Phân tích chi tiết và biểu đồ có trong
[`docs/phuong_phap.md`, mục 12 và 13](docs/phuong_phap.md).

| Lần chạy | Cấu hình | ROC-AUC | EER | Ngưỡng τ (chọn trên val) | Accuracy | FAR | FRR |
|---|---|---:|---:|---|---:|---:|---:|
| `20261109-201136` | `lfw`, augment, lô 32, Adam 1e-4; dừng ở epoch 17, tốt nhất epoch 7; RTX 5070 + PyTorch, 9 phút 22 giây | **0,853** | **21,9%** | `f1`: 0,285 | 77,4% | 32,0% | 13,3% |
| | | | | `youden`: 0,410 | 78,0% | 24,0% | 20,1% |
| | | | | `eer`: 0,485 | 77,5% | 20,0% | 24,9% |
| `20261110-162959` | như trên, thêm `--preprocess face`; dừng ở epoch 20, tốt nhất epoch 10; 9 phút 48 giây | **0,852** | **22,8%** | `f1`: 0,404 | 76,9% | 32,4% | 13,8% |
| | | | | `youden`: 0,544 | 77,2% | 24,1% | 21,5% |
| | | | | `eer`: 0,573 | 77,3% | 22,5% | 22,9% |

Trên ảnh tự chụp bằng webcam (`field-test`: 14 ảnh tham chiếu và 6 ảnh thử của một người, so
với 300 người lạ chưa từng thấy):

| Lần chạy | Tiền xử lý | ROC-AUC theo ảnh (KTC 95%) | EER theo ảnh | Ở τ_d = 0,8, τ_v = 0,5: chủ thể / người lạ được chấp nhận |
|---|---|---|---:|---|
| `20261109-201136` | `center` | 0,652 (0,583–0,714) | 33,3% | 0/6 / 9,3% |
| `20261110-162959` | `face` | **0,902** (0,860–0,941) | **15,2%** | 5/6 / 10,0% |

Nhận xét chính:
- Mô hình xác minh được người mới mà không cần huấn luyện lại (AUC 0,85).
- Cắt khuôn mặt **không đổi kết quả trên LFW** (hiệu AUC −0,001, KTC 95% [−0,017; +0,016])
  nhưng **tăng AUC trên ảnh webcam từ 0,65 lên 0,90**. Kết quả tốt trên bộ dữ liệu chuẩn chưa
  đủ để kết luận về nơi triển khai.
- Mô hình vẫn chưa đủ cho xác thực thật: trên LFW, TAR chỉ 16,7% tại FAR 1%.
- Overfitting xuất hiện từ epoch 7, vì mô hình 39 triệu tham số chỉ có khoảng 1.200 người để học.
- Với ứng dụng xác thực, nên chọn ngưỡng theo `eer` hoặc cao hơn để giảm tỉ lệ chấp nhận
  nhầm người lạ (FAR).

Tạo lại kết quả bằng
`python -m siamese_face --backend torch train --mode lfw --augment --batch-size 32`
(seed 42). Quá trình huấn luyện trên GPU không tất định từng bit, nên số liệu có thể xê dịch nhẹ.

## Kiểm thử

```powershell
python -m pytest
```

Các kiểm thử cho độ đo, chia dữ liệu, chuẩn bị LFW và CLI chỉ cần NumPy. Các kiểm thử liên
quan đến mô hình sẽ tự bỏ qua nếu chưa cài TensorFlow.

## Tài liệu tham khảo

- G. Koch, R. Zemel, R. Salakhutdinov. *Siamese Neural Networks for One-shot Image
  Recognition*. ICML Deep Learning Workshop, 2015.
- G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller. *Labeled Faces in the Wild: A Database
  for Studying Face Recognition in Unconstrained Environments*. Technical Report 07-49,
  University of Massachusetts, Amherst, 2007.
- Danh sách đầy đủ có trong [`docs/phuong_phap.md`](docs/phuong_phap.md).
