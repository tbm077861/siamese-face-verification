# 2026-11-10 – Phát hiện và cắt khuôn mặt, đo lại trên LFW và trên ảnh webcam

## Bối cảnh

Mục [2026-11-10 về webcam](2026-11-10-webcam-quyen-rieng-tu-lich-su-git.md) cho thấy mô hình
LFW gần như không phân biệt được chủ thể với người lạ trên ảnh webcam. Nguyên nhân bị nghi ngờ
nhiều nhất: pipeline chỉ cắt vùng giữa ảnh, nên nền, quần áo và tỉ lệ khung hình đi thẳng vào
mạng.

Hôm nay kiểm tra giả thuyết đó:
- thêm bước phát hiện và cắt khuôn mặt;
- huấn luyện lại với đúng các siêu tham số cũ;
- so sánh hai mô hình trên cả LFW lẫn ảnh webcam.

## Thay đổi

- **`siamese_face/faces.py`:** phát hiện khuôn mặt bằng Haar cascade của OpenCV.
  - Nếu có nhiều mặt, chọn mặt gần tâm ảnh nhất.
  - Cắt vùng vuông có cạnh bằng 1,4 lần cạnh hộp phát hiện.
  - Không thấy mặt thì cắt vùng vuông ở giữa, cạnh bằng 60% cạnh ngắn của ảnh.
- **Tùy chọn `--preprocess center|face` cho lệnh `train`.**
  - Lựa chọn được ghi vào `config.json` của lần chạy.
  - Các lệnh `evaluate`, `verify`, `field-test` ở chế độ mặc định (`auto`) đọc lại tùy chọn
    này, nên mọi giai đoạn luôn dùng cùng một cách tiền xử lý.
  - Lần chạy cũ không có trường này được hiểu là `center`.
- **Lệnh mới `field-test`:** đo trên ảnh tự chụp.
  - Ảnh thử của chủ thể và ảnh người lạ được so với cùng bộ ảnh tham chiếu.
  - Người lạ chỉ lấy từ những danh tính không có trong tập train/val của mô hình.
  - Báo cáo ROC-AUC kèm khoảng tin cậy bootstrap, EER, và bảng TAR/FAR theo τ_d.
- **Lệnh mới `compare`:** so sánh hai lần chạy trên cùng một tập test.
  - Báo cáo hiệu số ROC-AUC, EER, TAR tại FAR 1% và tại FAR 10%.
  - Khoảng tin cậy tính bằng bootstrap theo cụm danh tính.
- **`metrics.py`:** thêm `tar_at_far`, `bootstrap_ci` và `paired_bootstrap`.
- **Kiểm thử:** 83 test (thêm 21).
- **Môi trường huấn luyện trong WSL:** cài `opencv-python-headless` cùng phiên bản với bản
  OpenCV trên Windows (4.13.0.92).
- **Dọn bản dự phòng git trên máy:** xoá các bản dự phòng còn chứa ảnh cũ, rồi chạy `git gc`.
  Chi tiết ở mục 2026-11-10 về quyền riêng tư.

## Thí nghiệm

### Thiết lập

- Hai lần chạy được so sánh:
  - lần chạy gốc `20261109-201136`, tiền xử lý `center`;
  - lần chạy mới `20261110-162959`, tiền xử lý `face`.
- Cả hai dùng cùng lệnh `train --mode lfw --augment --batch-size 32` (seed 42), chỉ khác
  `--preprocess`. Tập train/val/test giống hệt nhau: 14.360 / 2.206 / 3.112 cặp.
- Bước cắt mặt phát hiện được khuôn mặt ở 98,5% số ảnh LFW và ở 20/20 ảnh webcam.
- Lần chạy mới dừng ở epoch 20, trọng số tốt nhất ở epoch 10, tổng thời gian 9 phút 48 giây.
- Số liệu gốc: `docs/ket_qua/lfw_face_20261110-162959/` và
  `docs/ket_qua/so_sanh_center_face/`.

### Trên tập test LFW (758 người chưa từng thấy)

Kết quả từ lệnh `compare`; khoảng tin cậy 95% tính bằng bootstrap theo 595 cụm danh tính,
2.000 lần lấy mẫu.

| Độ đo | center | face | face − center | KTC 95% của hiệu số |
|---|---:|---:|---:|---|
| ROC-AUC | 0,853 | 0,852 | −0,001 | [−0,017; +0,016] |
| EER | 21,9% | 22,8% | +0,9 điểm | [−1,5; +2,8] |
| TAR tại FAR 1% | 10,8% | 16,7% | +5,9 điểm | [+0,1; +12,8] |
| TAR tại FAR 10% | 51,7% | 53,7% | +2,0 điểm | [−4,6; +8,4] |

### Trên ảnh webcam (lệnh `field-test`)

- 14 ảnh tham chiếu và 6 ảnh thử của chủ thể.
- 300 người lạ, lấy từ 758 danh tính của tập test (seed 0). Cả hai mô hình dùng đúng cùng
  bộ ảnh này.
- Khoảng tin cậy 95% tính bằng bootstrap lấy lại theo ảnh, phân tầng theo lớp.

| | center | face |
|---|---|---|
| ROC-AUC theo cặp | 0,630 (0,577–0,680) | **0,879** (0,844–0,912) |
| ROC-AUC theo ảnh (điểm trung bình trên các ảnh tham chiếu) | 0,652 (0,583–0,714) | **0,902** (0,860–0,941) |
| EER theo ảnh | 33,3% | 15,2% |
| Trung vị điểm: chủ thể / người lạ | 0,43 / 0,26 | 0,83 / 0,21 |

Quy tắc xác minh với τ_v = 0,5:

| τ_d | center: chủ thể | center: người lạ | face: chủ thể | face: người lạ |
|---:|---:|---:|---:|---:|
| 0,4 | 5/6 | 36,3% | 6/6 | 37,3% |
| 0,5 | 0/6 | 29,0% | 6/6 | 30,0% |
| 0,7 | 0/6 | 15,0% | 6/6 | 16,7% |
| 0,8 | 0/6 | 9,3% | 5/6 | 10,0% |
| 0,9 | 0/6 | 4,0% | 0/6 | 5,0% |

Phép đo tạm hôm trước cho AUC 0,607 với mô hình `center`. Con số đó khác 0,630 ở đây vì hôm
trước người lạ được lấy từ mọi danh tính, kể cả những người có trong tập train. Từ nay số liệu
của lệnh `field-test` là số liệu chuẩn.

## Kết luận rút ra

1. **Giả thuyết được xác nhận trên ảnh webcam.**
   - Chỉ đổi bước cắt ảnh, còn kiến trúc, siêu tham số và dữ liệu giữ nguyên, mà AUC trên ảnh
     webcam tăng từ 0,65 lên 0,90 (tính theo ảnh). Hai khoảng tin cậy không giao nhau.
   - Nguyên nhân chính khiến mô hình thất bại trên webcam là khung hình và nền khác dữ liệu
     huấn luyện, chứ không phải mạng không học được đặc trưng khuôn mặt.
2. **Trên LFW, cắt khuôn mặt gần như không thay đổi kết quả.**
   - AUC và EER không đổi: khoảng tin cậy của hiệu số chứa 0.
   - Chỉ TAR tại FAR 1% tăng 5,9 điểm, nhưng cận dưới của khoảng tin cậy sát 0, nên chưa thể
     khẳng định.
   - Lý do khả dĩ: ảnh LFW vốn đã được căn giữa bằng chính bộ phát hiện Viola–Jones, nên bước
     cắt chủ yếu chỉ phóng to khuôn mặt.
   - Ngoài ra, bước cắt bỏ đi nền và quần áo. Trong LFW, các ảnh của cùng một người thường
     chụp tại cùng sự kiện nên hay có nền và quần áo giống nhau, tức đó cũng là một manh mối.
     Đây là giả thuyết, chưa kiểm chứng.
3. **Bộ dữ liệu chuẩn có thể che mất vấn đề ở nơi triển khai.** Chỉ nhìn LFW thì bước cắt
   khuôn mặt "vô tác dụng"; trên webcam, đó lại là thay đổi quan trọng nhất. Phải đo ở đúng
   miền triển khai, bằng lệnh `field-test`.
4. **Vẫn chưa đủ cho xác thực thật.**
   - Ở τ_d = 0,8, 5/6 ảnh thử của chủ thể được chấp nhận, nhưng 10% người lạ cũng được chấp
     nhận. Ở τ_d = 0,9 thì chủ thể cũng bị từ chối.
   - Điều này khớp với TAR chỉ 16,7% tại FAR 1% trên LFW: giới hạn còn lại nằm ở bản thân mạng,
     vốn học từ đầu trên khoảng 9 nghìn ảnh.
5. **Chọn epoch theo val loss làm mất AUC.**
   - Lần chạy mới tốt nhất theo val loss ở epoch 10, khi val AUC là 0,842. Nhưng val AUC còn
     tăng tới 0,868 ở epoch 15, trong lúc val loss tăng, tức mô hình ngày càng tự tin quá mức.
   - Lần chạy gốc chênh ít hơn: 0,848 so với đỉnh 0,856 ở epoch 14.
   - Nếu mục tiêu là xếp hạng (AUC, TAR tại FAR cố định), nên thử chọn epoch theo val AUC.
6. **So sánh phải kèm khoảng tin cậy.** Chênh lệch AUC 0,001 trên LFW nằm gọn trong nhiễu
   (±0,017). Bootstrap phải lấy mẫu theo cụm danh tính, vì các cặp của cùng một người không
   độc lập với nhau.
7. **Khoảng tin cậy của `field-test` chỉ phản ánh độ biến thiên của chính các ảnh đã chụp**
   (1 người, 1 buổi chụp, 6 ảnh thử). Với mẫu nhỏ như vậy, bootstrap phân vị còn có xu hướng
   cho khoảng quá hẹp. Muốn kết luận chung, cần nhiều người và nhiều buổi chụp.
8. **Cách tiền xử lý phải đi theo mô hình.** Ghi `preprocess` vào `config.json` và để các lệnh
   tự đọc lại, như vậy không thể vô tình xác minh bằng một cách cắt khác lúc huấn luyện.
9. **Kiểm chứng chéo vẫn khớp khi có bước cắt mặt.** Đánh giá lại bằng TensorFlow và OpenCV
   trên Windows cho AUC lệch 6·10⁻⁶, EER trùng khớp; chỉ 1 cặp đổi quyết định do ngưỡng làm
   tròn khác nhau. Vậy hai bản OpenCV cùng phiên bản cho kết quả phát hiện giống nhau.
10. **Chọn hộp gần tâm ảnh chứ không chọn hộp lớn nhất.** Chủ thể của ảnh LFW và ảnh tự chụp
    luôn ở giữa, còn ảnh LFW thường có người khác ở phía sau.
11. **Với bước cắt mặt, khoảng cách tới camera ít quan trọng hơn.** Trên ảnh webcam, hộp phát
    hiện chiếm khoảng 55% khung, lớn hơn cỡ elip hướng dẫn (45%), nhưng bước cắt đã đưa về
    cùng tỉ lệ. Elip vẫn có ích để giữ mặt ở giữa và nhìn thẳng.

## Việc còn dở

- [ ] Thử chọn epoch theo val AUC (giữ thiết lập `face`), rồi so sánh bằng `compare`.
- [ ] Thu thêm dữ liệu:
      - ảnh của chủ thể ở nhiều buổi chụp khác nhau;
      - ảnh webcam của người khác, làm người lạ cùng miền để đo FAR đúng nghĩa.
- [ ] Căn chỉnh khuôn mặt theo điểm mốc (vị trí mắt), hoặc dùng bộ phát hiện mạnh hơn
      (YuNet, RetinaFace).
- [ ] Cải thiện bản thân mạng, vì giới hạn còn lại nằm ở đây:
      - thay Flatten + Dense(4096) bằng Global Average Pooling;
      - hoặc dùng backbone đã huấn luyện trước.
