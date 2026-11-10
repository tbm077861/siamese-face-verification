# 2026-11-09 – Tái cấu trúc, chế độ `lfw` và huấn luyện trên GPU

## Bối cảnh

Kho mã ban đầu gồm hai phần:
- một ứng dụng Kivy: đăng nhập bằng khuôn mặt, dashboard quản lý ảnh và tài liệu;
- một mô hình Siamese nạp từ file `siamesemodel.h5` được huấn luyện ở nơi khác.

Mục tiêu là chỉ giữ phần khoa học máy tính (dữ liệu, mô hình, huấn luyện, đánh giá, suy luận),
với phương pháp rõ ràng và số liệu kiểm chứng được.

## Thay đổi

**Cấu trúc**
- Gỡ toàn bộ phần ứng dụng Kivy. Mã cũ vẫn còn trong lịch sử git, ở commit "final".
- Tạo gói `siamese_face` với giao diện dòng lệnh `python -m siamese_face`, gồm các lệnh
  `prepare-negatives`, `collect`, `train`, `evaluate`, `verify` và `summary`.
- Viết tài liệu `README.md`, `docs/phuong_phap.md` và `data/README.md`. Số liệu gốc của lần
  huấn luyện được báo cáo nằm trong `docs/ket_qua/lfw_20261109-201136/`.
- Viết bộ kiểm thử pytest (60 test vào cuối ngày).

**Phương pháp**
- Mô hình theo Koch và cộng sự (2015): 4 lớp tích chập → Dense(4096, sigmoid) → khoảng cách
  L1 → Dense(1, sigmoid). Mô hình có 38.964.545 tham số, ảnh vào kích thước 100×100×3.
- Chia dữ liệu chống rò rỉ:
  - chia ảnh (chế độ `personal`) hoặc chia danh tính (chế độ `lfw`) thành train/val/test
    **trước**, rồi mới ghép cặp trong từng tập;
  - seed 42;
  - danh sách chia được lưu vào `splits.json` của mỗi lần chạy.
- Các độ đo viết bằng NumPy: accuracy, precision, recall, F1, FAR, FRR, ROC, AUC, EER. Ngưỡng
  được chọn trên tập val theo một trong ba tiêu chí `f1`, `youden`, `eer`, rồi áp lên tập test.
- Quy tắc xác minh khi có nhiều ảnh tham chiếu:
  - một ảnh tham chiếu được tính là "khớp" khi điểm tương đồng ≥ τ_d;
  - xác minh thành công khi tỉ lệ ảnh tham chiếu khớp ≥ τ_v.
- Thêm chế độ `lfw`, vì lúc này chưa có ảnh cá nhân.
  - Cặp dương là hai ảnh của cùng một người trong LFW, tối đa 20 cặp mỗi người.
  - Cặp âm là ảnh của hai người khác nhau.
  - Ba tập train/val/test gồm những người không trùng nhau, nên tập test đo khả năng xác minh
    người *chưa từng thấy* khi huấn luyện.
- Mọi giai đoạn (huấn luyện, đánh giá, suy luận) dùng chung một hàm tiền xử lý `load_image`:
  cắt vùng vuông ở giữa ảnh rồi resize về 100×100.

**Kỹ thuật**
- Pipeline `tf.data` có bộ nhớ đệm theo ảnh: mỗi ảnh chỉ được giải mã một lần, mỗi cặp chỉ là
  hai chỉ số trỏ vào bộ đệm.
- Mã mô hình và suy luận viết bằng `keras`/`keras.ops`, nên chạy được trên mọi backend của
  Keras 3 (tùy chọn `--backend tensorflow|torch|jax`).
- Mô hình được lưu không kèm trạng thái của bộ tối ưu Adam: 156 MB thay vì 468 MB.
- Khi file mô hình được lưu bằng phiên bản Keras khác, chương trình dựng lại kiến trúc từ file
  `.keras` rồi chỉ nạp trọng số.
- Thêm vào `.gitignore`: dữ liệu, mô hình, thư mục `runs/` và các file nén (`*.tgz`,
  `*.tar.gz`, `*.zip`).

## Kết quả

Chi tiết nằm ở mục 12 của [`phuong_phap.md`](../phuong_phap.md).

- Tập test gồm 758 người chưa từng thấy, 3.112 cặp: **ROC-AUC 0,853**, **EER 21,9%**.
- Với ngưỡng `eer` (τ = 0,485): accuracy 77,5%, FAR 20,0%, FRR 24,9%.
- Tỉ lệ chấp nhận đúng (TAR) đạt 51,7% khi FAR = 10%, và chỉ 10,8% khi FAR = 1%.
- Huấn luyện dừng sớm ở epoch 17, trọng số tốt nhất ở epoch 7. Tổng thời gian 9 phút 22 giây
  trên RTX 5070 Laptop (WSL2 + PyTorch).

## Kết luận rút ra

**Về phương pháp**

1. **Ghép cặp trước rồi mới chia tập là rò rỉ dữ liệu.** Khi đó cùng một ảnh có thể nằm trong
   cả cặp train lẫn cặp test, khiến số liệu test cao giả tạo. Phải chia ảnh hoặc danh tính
   trước.
2. **Ngưỡng phải chọn trên tập val, không chọn trên tập test.** AUC trên val (0,849) xấp xỉ AUC
   trên test (0,853), cho thấy kết quả không phải do "khớp may" với tập test. Nên luôn báo cáo
   kèm AUC và EER, vì hai độ đo này không phụ thuộc ngưỡng.
3. **Chọn ngưỡng là chọn mức đánh đổi về an ninh.** Ngưỡng theo `f1` cho F1 cao nhất nhưng chấp
   nhận nhầm 32% người lạ, nên không dùng được cho xác thực. Ngưỡng `eer` (0,485) hoặc cao hơn
   hợp lý hơn.
4. **Mô hình overfitting rất nhanh.** Kết quả tốt nhất ở epoch 7; sau đó val loss tăng trong khi
   accuracy trên train lên tới 99,4%.
   - Nguyên nhân: mô hình có 39 triệu tham số (97% nằm ở lớp Dense 9.216 → 4.096), nhưng chỉ
     khoảng 1.200 người có cặp dương để học.
   - Vì vậy bắt buộc phải dừng sớm và khôi phục trọng số tốt nhất.
5. **Val loss tăng nhưng AUC đi ngang** nghĩa là mô hình ngày càng tự tin quá mức, chứ chưa
   xếp hạng kém đi.
6. **Không so sánh trực tiếp với DeepFace hay FaceNet.** Các hệ thống đó học trên hàng triệu ảnh,
   có căn chỉnh khuôn mặt, và đánh giá theo giao thức 10-fold của LFW.
7. **Tiền xử lý phải giống nhau ở mọi giai đoạn.** Bản ứng dụng cũ resize thẳng khung hình
   640×480 và ảnh 1151×2560 về 100×100, làm khuôn mặt bị méo so với lúc huấn luyện.
8. **Có những lỗi cài đặt rất khó thấy.** Nhiều bản cài đặt kiến trúc này viết
   `MaxPooling2D(64, (2, 2))`, tức là cửa sổ gộp 64×64 chứ không phải 2×2. Vì kích thước đầu ra
   không đổi nên lỗi không lộ ra.

**Về môi trường và công cụ**

9. **TensorFlow 2.20 không chạy được trên GPU RTX 50** (kiến trúc Blackwell, sm_120).
   - Lỗi gặp phải là `CUDA_ERROR_INVALID_PTX`, kể cả khi chạy trong WSL2 với
     `tensorflow[and-cuda]`.
   - Giải pháp: dùng WSL2 Ubuntu 24.04 và Keras 3 với backend PyTorch 2.11 (CUDA 12.8).
     TensorFlow bản CPU chỉ còn dùng cho `tf.data`.
   - Tốc độ khoảng 33 giây mỗi epoch. Cách cài đặt nằm trong [README](../../README.md).
10. **Viết mô hình bằng `keras.ops` thay vì `tf.*`** giúp đổi backend mà không phải sửa mã. Khi
    backend không phải TensorFlow, phải ẩn GPU khỏi TensorFlow để `tf.data` không chiếm bộ nhớ
    GPU.
11. **Kiểm chứng chéo giữa các backend là một phép thử tính đúng rất rẻ.** Mô hình huấn luyện
    bằng PyTorch trên GPU, đánh giá lại bằng TensorFlow trên CPU, cho số liệu trùng khớp; AUC
    chỉ lệch 4·10⁻⁶.
12. **File `.keras` không tương thích ngược giữa các bản Keras 3.**
    - File lưu bằng Keras 3.15 không nạp được trên Keras 3.12 (lỗi `GlorotUniform` …
      `input_axes`).
    - Đã thêm cơ chế nạp dự phòng và một test hồi quy cho lỗi này.
    - Phiên bản Keras được ghi vào `training_summary.json` để biết mô hình được lưu bằng bản
      nào.
13. **Chỉ lưu trạng thái bộ tối ưu khi cần huấn luyện tiếp**, vì nó làm file mô hình nặng gấp 3.
14. **Khi gọi lệnh WSL từ Windows, dùng `wsl -e bash -lc '…'`.** Với `wsl -- bash -lc`, đối số bị
    phân tích lại nên `$?` và dấu nháy bị sai; từng có lần lệnh báo nhầm mã thoát 0.
15. **Console Windows dùng bảng mã cp1252.** Script Python in tiếng Việt cần đặt
    `PYTHONIOENCODING=utf-8`. Riêng CLI của dự án đã tự cấu hình việc này.
16. **Luôn xem `git status` trước khi commit.** File `lfw.tgz` nặng 180 MB từng suýt bị đưa vào
    commit; sau đó các đuôi file nén đã được thêm vào `.gitignore`.
17. **Số liệu trong tài liệu phải tính từ file kết quả.** Từng có lần ghi TAR ≈ 55% tại
    FAR 10% do đọc bằng mắt trên biểu đồ ROC; con số tính đúng là 51,7%.

## Việc còn dở (tại thời điểm cuối ngày)

- [ ] Thu ảnh cá nhân và chạy chế độ `personal`. Ngày 2026-11-10 mới chụp ảnh để xác minh,
      chưa chụp đủ để huấn luyện.
- [ ] Dọn dẹp trên máy:
      - file `lfw.tgz` ở thư mục gốc;
      - `runs/20261109-201136/model_with_optimizer.keras`;
      - môi trường ảo trong WSL (nếu không huấn luyện thêm).
