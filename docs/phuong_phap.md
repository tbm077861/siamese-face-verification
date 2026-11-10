# Phương pháp: xác minh khuôn mặt one-shot bằng mạng Siamese

Tài liệu này trình bày cơ sở lý thuyết, mô hình, dữ liệu và giao thức thực nghiệm được
cài đặt trong gói `siamese_face`.

## 1. Bài toán

**Xác minh khuôn mặt** (*face verification*, so khớp 1:1) trả lời câu hỏi: *hai ảnh
$x_1, x_2$ có phải của cùng một người không?* Khác với **nhận dạng** (*identification*,
1:N) – tìm danh tính trong một tập người đã biết – bài toán xác minh có thể giải bằng một
hàm tương đồng $p(x_1, x_2) \in [0, 1]$ cùng một ngưỡng $\tau$:

$$
\text{cùng người} \iff p(x_1, x_2) \ge \tau .
$$

Trong thiết lập **one-shot**, hệ thống chỉ cần một (hoặc vài) ảnh tham chiếu của người dùng
thay vì hàng nghìn ảnh cho mỗi lớp như bài toán phân loại thông thường, vì mô hình học *cách
so sánh* hai ảnh chứ không học thuộc từng danh tính. Đây cũng là ý tưởng nền của các hệ
thống như DeepFace (Taigman và cộng sự, 2014) và FaceNet (Schroff và cộng sự, 2015).

## 2. Mạng Siamese

Mạng Siamese (Bromley và cộng sự, 1993; Koch và cộng sự, 2015) gồm hai nhánh **giống hệt
nhau và dùng chung trọng số** $\theta$. Mỗi nhánh là một mạng nhúng
$f_\theta : \mathbb{R}^{100 \times 100 \times 3} \to (0, 1)^{4096}$.

```mermaid
flowchart LR
    X1["Ảnh x₁"] --> F1["Mạng nhúng f_θ"]
    X2["Ảnh x₂"] --> F2["Mạng nhúng f_θ"]
    F1 --> D["L1Dist: abs(f_θ(x₁) − f_θ(x₂))"]
    F2 --> D
    D --> S["Dense(1) + sigmoid"]
    S --> P["p(x₁, x₂)"]
    F1 -.-|dùng chung trọng số θ| F2
```

Vector khoảng cách và xác suất đầu ra:

$$
\mathbf{d} = \left| f_\theta(x_1) - f_\theta(x_2) \right| \in \mathbb{R}^{4096},
\qquad
p(x_1, x_2) = \sigma\left(\mathbf{w}^\top \mathbf{d} + b\right),
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}} .
$$

Hai tính chất quan trọng:

- **Đối xứng**: $p(x_1, x_2) = p(x_2, x_1)$ vì hai nhánh dùng chung $f_\theta$ và
  $|a - b| = |b - a|$ (được kiểm thử trong `tests/test_model.py`).
- **Trọng số học được cho từng chiều**: khác với khoảng cách Euclid cố định, vector
  $\mathbf{w}$ cho phép mô hình tự học chiều đặc trưng nào quan trọng khi so sánh.

## 3. Kiến trúc mạng

Cài đặt trong `siamese_face/model.py`; in lại bằng `python -m siamese_face summary`.

| Lớp | Cấu hình | Kích thước đầu ra | Số tham số |
|---|---|---|---:|
| Input | ảnh RGB | 100×100×3 | 0 |
| Conv1 | 64 bộ lọc 10×10, ReLU | 91×91×64 | 19.264 |
| MaxPool1 | 2×2, bước 2, `same` | 46×46×64 | 0 |
| Conv2 | 128 bộ lọc 7×7, ReLU | 40×40×128 | 401.536 |
| MaxPool2 | 2×2, bước 2, `same` | 20×20×128 | 0 |
| Conv3 | 128 bộ lọc 4×4, ReLU | 17×17×128 | 262.272 |
| MaxPool3 | 2×2, bước 2, `same` | 9×9×128 | 0 |
| Conv4 | 256 bộ lọc 4×4, ReLU | 6×6×256 | 524.544 |
| Flatten | | 9.216 | 0 |
| Dense | 4096 nơ-ron, sigmoid | 4.096 | 37.752.832 |
| **Mạng nhúng $f_\theta$** | | | **38.960.448** |
| L1Dist | trị tuyệt đối hiệu hai vector | 4.096 | 0 |
| Dense (phân loại) | 1 nơ-ron, sigmoid | 1 | 4.097 |
| **Toàn mạng Siamese** | | | **38.964.545** |

Nhận xét:

- Khoảng **96,9%** tham số nằm ở lớp Dense 9.216 → 4.096. Đây là điểm nghẽn bộ nhớ
  (≈156 MB trọng số float32) và là ứng viên đầu tiên khi cần thu gọn mô hình (mục 10).
- Cạnh ảnh đầu vào tối thiểu là 58 pixel, để bốn lớp tích chập `valid` và ba lớp gộp còn
  kích thước dương.

## 4. Hàm mục tiêu và tối ưu

Với nhãn $y_i = 1$ (cùng người) hoặc $y_i = 0$ (khác người), mô hình được huấn luyện bằng
**binary cross-entropy**:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Big].
$$

| Siêu tham số | Mặc định | Tham số dòng lệnh |
|---|---|---|
| Thuật toán tối ưu | Adam (Kingma & Ba, 2015) | – |
| Tốc độ học | $10^{-4}$ | `--learning-rate` |
| Kích thước lô | 16 | `--batch-size` |
| Số epoch tối đa | 50 | `--epochs` |
| Dừng sớm | sau 10 epoch `val_loss` không giảm; luôn khôi phục trọng số tốt nhất | `--patience` |
| Tăng cường dữ liệu | tắt | `--augment` |
| Seed ngẫu nhiên | 42 | `--seed` |

## 5. Dữ liệu

Dự án hỗ trợ hai chế độ thực nghiệm (`--mode`):

| Chế độ | Cặp dương | Cặp âm | Mô hình thu được |
|---|---|---|---|
| `personal` | (anchor, positive) của một người | (anchor, ảnh LFW) | cá nhân hoá cho người đó |
| `lfw` | hai ảnh khác nhau của cùng một người trong LFW | hai ảnh LFW của hai người khác nhau | tổng quát, xác minh được cả người chưa từng thấy |

### 5.1. Chế độ `lfw` – tổng quát hoá cho người mới

LFW có 13.233 ảnh của 5.749 người, trong đó 1.680 người có từ hai ảnh trở lên. Quy trình:

1. **Chia theo danh tính:** mọi ảnh của một người nằm trọn trong train, val hoặc test (tỉ lệ
   70/15/15 tính theo số ảnh).
2. **Cặp dương:** mọi tổ hợp $\binom{k}{2}$ của người có $k \ge 2$ ảnh, giới hạn tối đa 20
   cặp mỗi người (`--max-pairs-per-identity`). Nếu không giới hạn, riêng George W. Bush
   (530 ảnh) đã có 140.185 cặp, áp đảo toàn bộ dữ liệu.
3. **Cặp âm:** rút ngẫu nhiên đều hai ảnh thuộc hai người khác nhau, với số lượng bằng số
   cặp dương.

Vì tập test chỉ gồm những người mô hình *chưa từng thấy*, số liệu đo được chính là khả năng
xác minh one-shot đúng nghĩa: chỉ cần một ảnh tham chiếu của người mới, không phải huấn luyện
lại. Đây là thiết lập gốc của Koch và cộng sự.

### 5.2. Chế độ `personal`

| Tập | Nội dung | Nguồn |
|---|---|---|
| Anchor ($A$) | ảnh của người cần xác minh | webcam (`collect`, phím `a`) |
| Positive ($P$) | ảnh khác của chính người đó | webcam (`collect`, phím `p`) |
| Negative ($N$) | ảnh của người khác | LFW: 13.233 ảnh của 5.749 người, mỗi ảnh 250×250 |

**Ghép cặp** (thực hiện riêng trong từng tập train/val/test): mỗi ảnh anchor $a_i$ tạo một
cặp dương $(a_i, p_i, 1)$ và một cặp âm $(a_i, n_i, 0)$. Dữ liệu vì vậy **cân bằng lớp**,
gồm $2n$ cặp với $n = \min(|A|, |P|, |N|)$.

Vì mọi cặp đều chứa một ảnh anchor của cùng một người, mô hình thu được là mô hình **cá
nhân hoá** cho người đó: thực chất là một bộ phân biệt "đúng người này hay không" được
học dưới dạng so sánh cặp. Muốn xác minh người khác thì phải thu ảnh và huấn luyện lại;
chế độ `lfw` không có hạn chế này.

### 5.3. Bộ nhớ đệm theo ảnh

Mỗi ảnh *duy nhất* chỉ được giải mã một lần và giữ trong RAM; mỗi cặp chỉ lưu hai chỉ số
trỏ tới ảnh (`pipeline.make_dataset`). Ở chế độ `lfw`, khoảng 20 nghìn cặp dùng chung
khoảng 13 nghìn ảnh, nên bộ nhớ cần ≈ 13.233 × 100 × 100 × 3 × 4 byte ≈ 1,6 GB. Nếu cache theo
từng cặp, con số này sẽ là vài GB.

## 6. Tiền xử lý và tăng cường dữ liệu

Huấn luyện, đánh giá và suy luận dùng chung một cách tiền xử lý, nhằm tránh lệch tiền xử lý
giữa các giai đoạn (*training/serving skew*). Cách tiền xử lý được ghi vào `config.json` của
lần chạy (trường `preprocess`), nên các lệnh `evaluate`, `verify` và `field-test` tự đọc lại
và dùng đúng cách đó. Có hai cách (`--preprocess`):

| Cách | Bước cắt | Hàm |
|---|---|---|
| `center` (mặc định) | vùng vuông lớn nhất ở giữa ảnh | `pipeline.load_image` |
| `face` | vùng vuông quanh khuôn mặt phát hiện được | `pipeline.load_face_image`, `faces.crop_face` |

Sau bước cắt, cả hai cách giống nhau:
- giải mã JPEG/PNG/BMP, nhận dạng định dạng theo nội dung file chứ không theo đuôi;
- resize song tuyến tính về 100×100;
- chuẩn hoá giá trị điểm ảnh về $[0, 1]$.

Lệnh `collect` cắt sẵn vùng vuông ở giữa khung hình (mặc định 250×250, cùng cỡ ảnh LFW). Nó
cũng vẽ một elip hướng dẫn đúng cỡ khuôn mặt trong ảnh LFW, để ảnh tự chụp có cùng tỉ lệ
khuôn mặt so với khung.

### 6.1. Phát hiện và cắt khuôn mặt (`face`)

1. **Phát hiện:** dùng bộ phân loại Haar cascade `haarcascade_frontalface_default.xml` có sẵn
   trong OpenCV (Viola và Jones, 2001).
   - Chạy trên ảnh xám đã cân bằng histogram.
   - Tham số `scaleFactor = 1,1`, `minNeighbors = 5`.
   - Bỏ qua các vùng nhỏ hơn 20% cạnh ngắn của ảnh.
2. **Chọn khuôn mặt:** nếu phát hiện được nhiều vùng, lấy vùng có tâm gần tâm ảnh nhất, vì
   chủ thể của ảnh LFW và ảnh tự chụp đều ở giữa khung.
3. **Cắt:** lấy vùng vuông cùng tâm với hộp phát hiện, cạnh bằng 1,4 lần cạnh hộp (nới mỗi
   phía 20% để lấy trọn trán và cằm). Nếu vùng này chạm mép ảnh thì dịch vào trong.
4. **Không thấy khuôn mặt:** cắt vùng vuông ở giữa ảnh, cạnh bằng 60% cạnh ngắn. Trên ảnh
   LFW, vùng này xấp xỉ vùng cắt khi phát hiện được mặt.

Ảnh LFW vốn được tạo bằng chính bộ phát hiện Viola–Jones rồi căn khuôn mặt vào giữa, nên trên
LFW bước này chủ yếu *phóng to* khuôn mặt và bỏ bớt nền. Với ảnh tự chụp, bước này đưa khuôn
mặt về cùng vị trí và cùng tỉ lệ khung như ảnh LFW, dù người chụp ngồi gần hay xa camera.

| Đo trên | Kết quả |
|---|---|
| LFW (13.233 ảnh) | Phát hiện được mặt ở 98,5% số ảnh. Cạnh vùng cắt có trung vị 160 px trên ảnh 250 px (khoảng 5%–95%: 147–174 px). Tâm vùng cắt lệch trung vị 2 px so với tâm ảnh. Tốc độ khoảng 3,7 ms/ảnh trên CPU. |
| Ảnh webcam của chủ thể (20 ảnh, khung 450 px) | Phát hiện được mặt ở cả 20/20 ảnh. |

Đây mới là bước *phát hiện và cắt*, chưa phải *căn chỉnh*: ảnh chưa được xoay để hai mắt nằm
ngang và chưa đưa mắt về vị trí cố định, nên khuôn mặt nghiêng vẫn giữ nguyên góc nghiêng.

### 6.2. Tăng cường dữ liệu

Tăng cường là tuỳ chọn, chỉ áp dụng cho tập train và độc lập cho từng ảnh trong cặp:
- lật ngang ngẫu nhiên;
- thay đổi độ sáng ±0,1;
- thay đổi độ tương phản và độ bão hoà trong khoảng ×[0,8; 1,2].

## 7. Giao thức chia dữ liệu – chống rò rỉ

Cài đặt trong `data.build_splits`, theo ba nguyên tắc:

1. **Chia trước, ghép cặp sau.** Ảnh anchor, positive và negative được chia train/val/test
   (mặc định 70/15/15) *trước*, rồi mới ghép cặp bên trong từng tập. Nếu ghép cặp trước rồi
   mới chia, cùng một ảnh anchor có thể xuất hiện ở cả tập train lẫn tập test.
2. **Ảnh negative chia theo danh tính.** Tên file LFW có dạng `Ten_Nguoi_0001.jpg`; mọi ảnh
   của cùng một người nằm chung một tập, nên người lạ trong tập test là người mô hình *chưa
   từng thấy* khi huấn luyện.
3. **Chia tất định và lưu lại.** Seed được cố định và danh sách cặp lưu trong `splits.json`;
   lệnh `evaluate --run` dùng lại đúng tập test đó.

> **Lỗi rò rỉ thường gặp.** Nhiều bản cài đặt chia dữ liệu bằng `data.shuffle(10000)` rồi
> `take()`/`skip()`. Vì `tf.data` mặc định *xáo lại sau mỗi lượt duyệt*
> (`reshuffle_each_iteration=True`), tập "test" thay đổi ở mỗi epoch và lẫn với tập train,
> khiến số liệu đánh giá bị lạc quan. Dự án này chia một lần duy nhất trên danh sách file nên
> không mắc lỗi đó.

**Hạn chế còn lại:** các khung hình webcam chụp liên tiếp có thể gần như trùng nhau. Nên thu
ảnh anchor/positive qua nhiều phiên khác nhau (khác ngày, khác ánh sáng) để tập test đủ độc
lập.

## 8. Độ đo đánh giá

Quy ước: lớp dương là "cùng người"; mô hình dự đoán dương khi $p \ge \tau$.

| Độ đo | Công thức | Ý nghĩa |
|---|---|---|
| Accuracy | $(TP + TN) / (TP + TN + FP + FN)$ | tỉ lệ quyết định đúng |
| Precision | $TP / (TP + FP)$ | trong các lần chấp nhận, tỉ lệ đúng người |
| Recall (TAR) | $TP / (TP + FN)$ | tỉ lệ chủ thể được chấp nhận |
| F1 | $2PR / (P + R)$ | trung bình điều hoà của Precision và Recall |
| FAR | $FP / (FP + TN)$ | người lạ bị chấp nhận nhầm – **rủi ro an ninh** |
| FRR | $FN / (FN + TP) = 1 - \text{Recall}$ | chủ thể bị từ chối nhầm – **gây bất tiện** |
| ROC-AUC | diện tích dưới đường $(\mathrm{FAR}(\tau), \mathrm{TAR}(\tau))$ | chất lượng xếp hạng, không phụ thuộc ngưỡng |
| EER | giá trị tại ngưỡng $\tau$ mà $\mathrm{FAR} = \mathrm{FRR}$ | một con số tóm tắt sự đánh đổi FAR/FRR |

Tất cả các độ đo được cài đặt bằng NumPy trong `metrics.py`. Đường ROC được tính bằng cách
quét ngưỡng qua các điểm đã sắp xếp, với độ phức tạp $O(n \log n)$; AUC được tính bằng quy
tắc hình thang. Các hàm này được kiểm thử bằng những ví dụ có đáp án biết trước.

## 9. Chọn ngưỡng và quy tắc suy luận

**Chọn ngưỡng** (`evaluate.py`): ngưỡng $\tau$ được chọn trên **tập val** theo một trong
ba tiêu chí, rồi giữ cố định khi đo trên **tập test**. Nhờ vậy, ngưỡng không được chọn bằng
chính dữ liệu dùng để báo cáo kết quả.

- `f1`: chọn ngưỡng cho F1 lớn nhất (mặc định);
- `youden`: chọn ngưỡng cho chỉ số Youden $J = \mathrm{TPR} - \mathrm{FPR}$ lớn nhất;
- `eer`: chọn ngưỡng tại điểm $\mathrm{FAR} \approx \mathrm{FRR}$.

Kết quả tại ngưỡng 0,5 cũng được ghi lại (mục `test_at_0.5` trong `metrics.json`) để so sánh.

**Suy luận với nhiều ảnh tham chiếu** (`verify.py`): gọi $r_1, \dots, r_K$ là $K$ ảnh tham
chiếu và $x$ là ảnh cần xác minh. Khi đó

$$
s_k = p(r_k, x), \qquad
\text{xác minh thành công} \iff \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\left[s_k \ge \tau_d\right] \ge \tau_v ,
$$

trong đó $\tau_d$ là ngưỡng phát hiện và $\tau_v$ là ngưỡng xác minh (cả hai mặc định
bằng 0,5). Khi $K = 1$, quy tắc rút gọn thành $s \ge \tau_d$, đúng như cách phiên bản ứng
dụng cũ so sánh với một ảnh anchor duy nhất. Dùng nhiều ảnh tham chiếu giúp quyết định ổn
định hơn trước nhiễu của từng ảnh riêng lẻ.

## 10. Hạn chế và hướng phát triển

- **Thêm dữ liệu huấn luyện:** chế độ `lfw` chỉ có 1.217 người có cặp dương trong tập train,
  và thực nghiệm (mục 12) cho thấy overfitting xuất hiện từ epoch 7. Các bộ lớn hơn
  (CASIA-WebFace, VGGFace2) sẽ cải thiện đáng kể khả năng tổng quát hoá.
- **Chống overfitting:** thêm dropout hoặc weight decay cho lớp Dense(4096), tăng cường dữ
  liệu mạnh hơn, giảm số chiều vector nhúng (`--embedding-dim`).
- **Hàm mất mát học độ đo:** dùng contrastive loss (Hadsell và cộng sự, 2006) hoặc triplet
  loss (Schroff và cộng sự, 2015) thay cho binary cross-entropy.
- **Căn chỉnh khuôn mặt:** đã có bước phát hiện và cắt khuôn mặt bằng Haar cascade (mục 6.1
  và 13). Bước tiếp theo là căn chỉnh theo các điểm mốc (mắt, mũi, miệng), dùng các bộ phát
  hiện mạnh hơn như MTCNN, RetinaFace hoặc YuNet.
- **Thu gọn mô hình:** thay Flatten + Dense(4096) (≈97% tham số) bằng Global Average
  Pooling, hoặc dùng backbone đã huấn luyện trước (MobileNet, ResNet) với học chuyển giao.
- **Đánh giá theo chuẩn:** áp dụng giao thức 10-fold trên `pairs.txt` của LFW để so sánh
  được với các công trình đã công bố.
- **Chống giả mạo** (*liveness detection* – ảnh in, ảnh trên màn hình) nằm ngoài phạm vi
  dự án.

## 11. Thay đổi so với phiên bản ứng dụng cũ

| Khía cạnh | Phiên bản cũ (commit `b3f15cf`, "final") | Phiên bản hiện tại |
|---|---|---|
| Phạm vi | Ứng dụng Kivy: đăng nhập bằng khuôn mặt, dashboard quản lý ảnh và tài liệu | Chỉ gồm mô hình và thực nghiệm |
| Huấn luyện | Không có mã trong kho; chỉ nạp `siamesemodel.h5` từ bên ngoài | `train.py` đầy đủ, lưu cấu hình, danh sách chia và lịch sử huấn luyện |
| Tiền xử lý khi suy luận | Khung hình 640×480 và ảnh anchor 1151×2560 được resize thẳng về 100×100 ⇒ méo tỉ lệ | Một hàm `load_image` chung cho mọi giai đoạn, cắt vuông ở giữa trước khi resize |
| Quyết định | So với một ảnh anchor, ngưỡng cố định 0,5 | Ngưỡng chọn trên tập val; hỗ trợ nhiều ảnh tham chiếu |
| Đánh giá | Không có | Accuracy, Precision, Recall, F1, FAR, FRR, ROC-AUC, EER |
| Tương thích | `L1Dist` chỉ nhận hai tensor rời | Nhận cả hai tensor rời lẫn một danh sách; chạy được với Keras 2 và Keras 3 |

> **Ghi chú cài đặt.** Một số bản cài đặt phổ biến của kiến trúc này viết
> `MaxPooling2D(64, (2, 2), padding="same")`. Trong Keras, tham số vị trí thứ nhất là
> `pool_size`, nên đó thực chất là cửa sổ gộp **64×64** với bước 2, không phải 2×2 như bài
> báo gốc. Vì kích thước đầu ra và số tham số không đổi, lỗi này khó phát hiện. Dự án dùng
> đúng `pool_size=(2, 2), strides=2`.

## 12. Kết quả thực nghiệm: chế độ `lfw`

Số liệu và biểu đồ gốc nằm trong [`ket_qua/lfw_20261109-201136/`](ket_qua/lfw_20261109-201136/).

### 12.1. Thiết lập

| Hạng mục | Giá trị |
|---|---|
| Lệnh | `python -m siamese_face --backend torch train --mode lfw --augment --batch-size 32` |
| Siêu tham số | Adam, tốc độ học $10^{-4}$, lô 32, tối đa 50 epoch, dừng sớm sau 10 epoch, tăng cường dữ liệu bật, seed 42 |
| Dữ liệu (cặp) | train 14.360 · val 2.206 · test 3.112, mỗi tập 50% cặp cùng người |
| Số người | train 3.448 (trong đó 1.217 người có cặp dương) · val 611 (211) · test 758 (252) – ba tập rời nhau |
| Phần cứng | GPU RTX 5070 Laptop 8 GB qua WSL2; Keras 3.15 với backend PyTorch 2.11 (CUDA 12.8) |
| Thời gian | 17 epoch × ~33 giây = 9 phút 22 giây; trọng số tốt nhất ở **epoch 7** (val_loss = 0,503) |

### 12.2. Kết quả trên tập test – những người chưa từng thấy khi huấn luyện

Các độ đo không phụ thuộc ngưỡng: **ROC-AUC = 0,853** và **EER = 21,9%**.

| Cách chọn ngưỡng (trên tập val) | τ | Accuracy | Precision | Recall | F1 | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f1` (mặc định) | 0,285 | 0,774 | 0,730 | **0,867** | **0,793** | 0,320 | **0,133** |
| `youden` | 0,410 | **0,780** | 0,769 | 0,800 | 0,784 | 0,240 | 0,201 |
| `eer` | 0,485 | 0,775 | 0,790 | 0,751 | 0,770 | 0,200 | 0,249 |
| cố định 0,5 | 0,500 | 0,774 | **0,796** | 0,738 | 0,766 | **0,190** | 0,262 |

![Đường cong huấn luyện](ket_qua/lfw_20261109-201136/training_curves.png)

| ROC trên tập test | Phân bố điểm tương đồng trên tập test |
|---|---|
| ![ROC](ket_qua/lfw_20261109-201136/roc_curve.png) | ![Phân bố điểm](ket_qua/lfw_20261109-201136/score_distribution.png) |

### 12.3. Phân tích

1. **Mô hình tổng quát hoá được cho người mới.** AUC 0,853 trên 758 người chưa từng thấy
   cho thấy mạng nhúng học được đặc trưng khuôn mặt chứ không chỉ ghi nhớ danh tính. Kết quả
   trên val (AUC 0,849) và test (0,853) gần nhau, chứng tỏ ngưỡng chọn trên val không bị "khớp
   may" với tập test.
2. **Overfitting rõ rệt.** Từ epoch 7, val_loss tăng trở lại trong khi train accuracy tiếp
   tục lên tới 99,4% ở epoch 17. Val AUC đi ngang quanh 0,85 từ epoch 6. Nguyên nhân là mô
   hình có 39 triệu tham số (97% nằm ở lớp Dense 9.216 → 4.096) nhưng chỉ có khoảng 1.200
   người có cặp dương để học. Val loss tăng trong khi AUC không giảm nghĩa là mô hình ngày
   càng *tự tin quá mức* chứ chưa xếp hạng kém đi. Dừng sớm và khôi phục trọng số tốt nhất
   là bắt buộc.
3. **Chọn ngưỡng là chọn đánh đổi an ninh.** Tiêu chí `f1` cho F1 và recall cao nhất, nhưng
   cứ khoảng 3 người lạ thì có 1 người được chấp nhận (FAR 32%) – không phù hợp cho xác thực.
   Với ứng dụng bảo mật nên dùng `eer` hoặc ngưỡng cao hơn, chấp nhận FRR lớn hơn (người dùng
   phải thử lại) để đổi lấy FAR thấp. Đọc trên đường ROC của tập test:

   | FAR tối đa | 1% | 5% | 10% | 20% |
   |---|---:|---:|---:|---:|
   | TAR đạt được | 10,8% | 33,0% | 51,7% | 75,2% |

   Ở mức an ninh chặt (FAR 1%), mô hình chỉ chấp nhận đúng khoảng 1/10 lần thử của chủ thể.
   Đây là giới hạn thực tế lớn nhất của mô hình hiện tại.
4. **Phân bố điểm.** Phần lớn cặp khác người có điểm gần 0, nhưng còn một "đuôi" các cặp
   khó trải đều tới 1. Cặp cùng người thì phân tán rộng: nhiều cặp có điểm thấp do khác biệt
   về tư thế, ánh sáng hoặc tuổi tác giữa các ảnh LFW, trong khi dự án chưa căn chỉnh khuôn
   mặt (chỉ cắt vuông ở giữa ảnh).
5. **Đặt trong bối cảnh.** DeepFace (97,35%) và FaceNet (99,63%) đạt độ chính xác rất cao
   trên LFW. Tuy nhiên, các hệ thống đó được huấn luyện trên hàng triệu ảnh của hàng nghìn tới
   hàng triệu người, có căn chỉnh khuôn mặt, và đánh giá theo giao thức 10-fold của LFW.
   Mô hình ở đây học từ đầu trên khoảng 9 nghìn ảnh không căn chỉnh, nên hai kết quả *không
   so sánh trực tiếp được*. Khoảng cách này chính là động lực cho các hướng phát triển ở
   mục 10.
6. **Tính tái lập và khả chuyển.** Mô hình huấn luyện bằng PyTorch trên GPU (Keras 3.15)
   được đánh giá lại bằng TensorFlow trên CPU (Keras 3.12, qua cơ chế nạp trọng số dự phòng).
   Accuracy, FAR, FRR và EER trùng khớp hoàn toàn; ROC-AUC chỉ lệch $4 \times 10^{-6}$ do làm
   tròn số thực. Việc chia dữ liệu là tất định (seed 42); riêng quá trình huấn luyện trên GPU
   không tất định từng bit, nên chạy lại có thể cho số liệu xê dịch nhẹ.

## 13. Thực nghiệm: phát hiện và cắt khuôn mặt (`--preprocess face`)

Mô hình ở mục 12 gần như không phân biệt được chủ thể với người lạ trên ảnh webcam (mục 13.3).
Giả thuyết là do ảnh chỉ được cắt vùng giữa (`center`), nên nền, quần áo và tỉ lệ khung hình
đi thẳng vào mạng. Thực nghiệm này kiểm tra giả thuyết đó bằng cách **chỉ đổi bước cắt ảnh**,
giữ nguyên mọi thứ khác. Số liệu gốc nằm trong
[`ket_qua/lfw_face_20261110-162959/`](ket_qua/lfw_face_20261110-162959/) và
[`ket_qua/so_sanh_center_face/`](ket_qua/so_sanh_center_face/).

### 13.1. Thiết lập

| Hạng mục | Giá trị |
|---|---|
| Lệnh | `python -m siamese_face --backend torch train --mode lfw --augment --batch-size 32 --preprocess face` |
| Khác với mục 12 | chỉ khác `--preprocess face` (mục 6.1); kiến trúc, siêu tham số, seed và tập train/val/test giống hệt (14.360 / 2.206 / 3.112 cặp) |
| Phát hiện khuôn mặt | 98,5% ảnh LFW; 20/20 ảnh webcam |
| Thời gian | 20 epoch, 9 phút 48 giây; trọng số tốt nhất ở **epoch 10** (val_loss = 0,513) |

### 13.2. Trên tập test LFW

Bảng dưới là kết quả của lệnh `compare` trên cùng 3.112 cặp test. Khoảng tin cậy 95% (KTC) của
hiệu số được tính bằng bootstrap phân vị với 2.000 lần lấy mẫu, lấy mẫu lại theo **cụm danh
tính** của ảnh anchor (595 cụm), vì các cặp của cùng một người không độc lập.

| Độ đo | `center` (mục 12) | `face` | Hiệu số | KTC 95% của hiệu số |
|---|---:|---:|---:|---|
| ROC-AUC | 0,853 | 0,852 | −0,001 | [−0,017; +0,016] |
| EER | 21,9% | 22,8% | +0,9 điểm | [−1,5; +2,8] |
| TAR tại FAR 1% | 10,8% | 16,7% | +5,9 điểm | [+0,1; +12,8] |
| TAR tại FAR 10% | 51,7% | 53,7% | +2,0 điểm | [−4,6; +8,4] |

Kết quả của mô hình `face` trên tập test, với ngưỡng chọn trên tập val:

| Cách chọn ngưỡng | τ | Accuracy | Precision | Recall | F1 | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f1` | 0,404 | 0,769 | 0,727 | 0,862 | 0,789 | 0,324 | 0,138 |
| `youden` | 0,544 | 0,772 | 0,765 | 0,785 | 0,775 | 0,241 | 0,215 |
| `eer` | 0,573 | 0,773 | 0,774 | 0,771 | 0,772 | 0,225 | 0,229 |
| cố định 0,5 | 0,500 | 0,772 | 0,753 | 0,812 | 0,781 | 0,267 | 0,188 |

Dòng `f1` được đo trong WSL (PyTorch). Các dòng `youden` và `eer` được đo lại trên Windows
(TensorFlow, CPU). Hai môi trường cho ROC-AUC lệch nhau 6·10⁻⁶ và cùng EER, nên bước cắt
khuôn mặt bằng hai bản OpenCV 4.13 cho kết quả như nhau.

### 13.3. Trên ảnh tự chụp bằng webcam (lệnh `field-test`)

- **Thiết lập:** 14 ảnh tham chiếu và 6 ảnh thử của **một** chủ thể, chụp trong cùng một buổi.
  So với chúng là 300 người lạ lấy ngẫu nhiên (seed 0) từ 758 danh tính của tập test, tức
  những người cả hai mô hình chưa từng thấy. Cả hai mô hình dùng đúng cùng các ảnh này.
- **Khoảng tin cậy 95%:** bootstrap lấy lại theo ảnh, phân tầng theo lớp.

| | `center` | `face` |
|---|---|---|
| ROC-AUC theo cặp (ảnh tham chiếu, ảnh cần xác minh) | 0,630 (0,577–0,680) | **0,879** (0,844–0,912) |
| ROC-AUC theo ảnh (điểm trung bình trên 14 ảnh tham chiếu) | 0,652 (0,583–0,714) | **0,902** (0,860–0,941) |
| EER theo ảnh | 33,3% | 15,2% |
| Trung vị điểm: chủ thể / người lạ | 0,43 / 0,26 | 0,83 / 0,21 |

Bảng quyết định theo quy tắc của lệnh `verify` (mục 9), với τ_v = 0,5:

| τ_d | `center`: chủ thể được chấp nhận | `center`: người lạ được chấp nhận | `face`: chủ thể được chấp nhận | `face`: người lạ được chấp nhận |
|---:|---:|---:|---:|---:|
| 0,4 | 5/6 | 36,3% | 6/6 | 37,3% |
| 0,5 | 0/6 | 29,0% | 6/6 | 30,0% |
| 0,7 | 0/6 | 15,0% | 6/6 | 16,7% |
| 0,8 | 0/6 | 9,3% | 5/6 | 10,0% |
| 0,9 | 0/6 | 4,0% | 0/6 | 5,0% |

### 13.4. Phân tích

1. **Trên ảnh webcam, giả thuyết được xác nhận.**
   - Chỉ đổi bước cắt ảnh mà ROC-AUC theo ảnh tăng từ 0,65 lên 0,90; hai khoảng tin cậy không
     giao nhau.
   - Điểm tương đồng của chủ thể tăng từ 0,43 lên 0,83, trong khi điểm của người lạ gần như
     giữ nguyên.
   - Vậy mô hình thất bại trên webcam chủ yếu vì khung hình và nền khác với dữ liệu huấn luyện,
     chứ không phải vì mạng không học được đặc trưng khuôn mặt. Với bước cắt khuôn mặt, AUC
     trên webcam (0,88–0,90) đã ngang AUC trên tập test LFW.
2. **Trên LFW, bước cắt gần như không thay đổi kết quả.**
   - ROC-AUC và EER không đổi: khoảng tin cậy của hiệu số chứa 0.
   - TAR tại FAR 1% tăng 5,9 điểm, nhưng cận dưới khoảng tin cậy chỉ +0,1 điểm, nên đây mới là
     dấu hiệu chứ chưa phải kết luận.
   - Điều này hợp lý: ảnh LFW vốn được căn giữa bằng chính bộ phát hiện Viola–Jones, nên bước
     cắt chủ yếu chỉ phóng to khuôn mặt.
   - Bước cắt còn bỏ đi nền và quần áo, mà trong LFW các ảnh của cùng một người thường chụp
     tại cùng sự kiện nên hay có nền và quần áo giống nhau. Nếu mạng `center` đã dựa một phần
     vào những manh mối đó, thì AUC trên LFW không đổi có thể đang che đi việc mạng `face` dựa
     vào khuôn mặt nhiều hơn. Đây là giả thuyết, chưa kiểm chứng.
3. **Kết quả trên bộ dữ liệu chuẩn có thể che mất vấn đề ở nơi triển khai.** Nếu chỉ đo trên
   LFW, ta sẽ kết luận bước cắt khuôn mặt "vô tác dụng". Trên webcam, đó lại là thay đổi quan
   trọng nhất. Vì vậy dự án đo ở cả hai nơi: `evaluate` trên LFW, `field-test` trên ảnh tự chụp.
4. **Vẫn chưa đủ cho xác thực thật.**
   - Ở τ_d = 0,8, 5/6 ảnh của chủ thể được chấp nhận nhưng 10% người lạ cũng được chấp nhận.
     Ở τ_d = 0,9, chính chủ thể cũng bị từ chối.
   - Điều này khớp với TAR chỉ 16,7% tại FAR 1% trên LFW. Giới hạn còn lại nằm ở bản thân mạng,
     học từ đầu trên khoảng 9 nghìn ảnh (mục 10).
5. **Chọn epoch theo val loss chưa tối ưu cho xếp hạng.**
   - Mô hình `face` tốt nhất theo val loss ở epoch 10 (val AUC 0,842), nhưng val AUC còn tăng tới
     0,868 ở epoch 15, trong khi val loss tăng vì mô hình ngày càng tự tin quá mức.
   - Với mô hình `center`, khoảng chênh nhỏ hơn: 0,848 so với đỉnh 0,856.
   - Khi mục tiêu là AUC hoặc TAR tại FAR cố định, nên thử chọn epoch theo val AUC.
6. **Giới hạn của phép đo trên webcam.**
   - Chỉ có một chủ thể, một buổi chụp và 6 ảnh thử.
   - Khoảng tin cậy chỉ phản ánh độ biến thiên của chính các ảnh này, không phản ánh biến
     thiên giữa các người dùng hay các buổi chụp. Với mẫu nhỏ, bootstrap phân vị còn có xu hướng
     cho khoảng quá hẹp.
   - Người lạ là ảnh báo chí trong LFW, không phải ảnh webcam của người khác, nên FAR ở đây chỉ
     là ước lượng gần đúng cho ứng dụng thật.

## Tài liệu tham khảo

1. J. Bromley, I. Guyon, Y. LeCun, E. Säckinger, R. Shah. *Signature Verification using a
   "Siamese" Time Delay Neural Network*. NIPS, 1993.
2. G. Koch, R. Zemel, R. Salakhutdinov. *Siamese Neural Networks for One-shot Image
   Recognition*. ICML Deep Learning Workshop, 2015.
3. G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller. *Labeled Faces in the Wild: A
   Database for Studying Face Recognition in Unconstrained Environments*. Technical Report
   07-49, University of Massachusetts, Amherst, 2007.
4. R. Hadsell, S. Chopra, Y. LeCun. *Dimensionality Reduction by Learning an Invariant
   Mapping*. CVPR, 2006.
5. Y. Taigman, M. Yang, M. Ranzato, L. Wolf. *DeepFace: Closing the Gap to Human-Level
   Performance in Face Verification*. CVPR, 2014.
6. F. Schroff, D. Kalenichenko, J. Philbin. *FaceNet: A Unified Embedding for Face
   Recognition and Clustering*. CVPR, 2015.
7. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization*. ICLR, 2015.
8. P. Viola, M. Jones. *Rapid Object Detection using a Boosted Cascade of Simple Features*.
   CVPR, 2001.
9. B. Efron, R. J. Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
