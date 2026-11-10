# Nhật ký dự án

Thư mục này ghi lại **những gì đã thay đổi** và **những kết luận rút ra** trong quá trình làm
dự án, theo thứ tự thời gian. Các tài liệu khác chỉ mô tả trạng thái *hiện tại*:

| Tài liệu | Trả lời câu hỏi |
|---|---|
| [`../phuong_phap.md`](../phuong_phap.md) | Phương pháp là gì, vì sao làm như vậy |
| [`../ket_qua/`](../ket_qua/) | Số liệu gốc của từng lần huấn luyện đã báo cáo |
| `nhat_ky/` (thư mục này) | Đã làm gì, đã thử gì, kết quả ra sao, rút ra được gì, kể cả những hướng không thành công |

## Quy ước

- Mỗi phiên làm việc (hoặc mỗi ngày) ghi vào một file `YYYY-MM-DD-chu-de-ngan.md`.
- Mỗi file gồm bốn phần: **Bối cảnh**, **Thay đổi**, **Kết luận rút ra**, **Việc còn dở**.
  - Nếu có thí nghiệm, ghi thêm thiết lập, cách đo và số liệu để người khác làm lại được.
  - Việc còn dở dùng ô đánh dấu `- [ ]`. Khi xong thì đổi thành `- [x]` và ghi ngày hoàn thành.
- Không sửa kết luận cũ khi hiểu biết thay đổi.
  - Viết mục mới, và ở mục cũ ghi thêm *"(đã được thay thế bởi …)"*.
  - Nhật ký phải phản ánh đúng những gì đã biết tại thời điểm đó.
- Số liệu phải lấy từ file kết quả hoặc từ một phép đo chạy lại được, không ước lượng bằng mắt
  trên biểu đồ.
- Không ghi dữ liệu cá nhân như ảnh, email, mật khẩu, hay đặc điểm nhận dạng của người được
  chụp, vì kho mã đang ở chế độ công khai.
- Hạn chế ghi mã hash của commit, vì lịch sử git từng được viết lại (mục 2026-11-10) và mã hash
  có thể đổi. Nên ghi thông điệp commit hoặc tên file.

## Mục lục

| Ngày | Nội dung |
|---|---|
| [2026-11-09](2026-11-09-tai-cau-truc-va-thuc-nghiem-lfw.md) | Tái cấu trúc thành dự án khoa học máy tính; chế độ `lfw`; huấn luyện trên GPU RTX 50; kết quả trên LFW |
| [2026-11-10](2026-11-10-webcam-quyen-rieng-tu-lich-su-git.md) | Thử mô hình LFW trên ảnh webcam; quyền riêng tư; xoá ảnh cá nhân khỏi lịch sử git |

## Các kết luận chính đến hiện tại

Đây là tóm tắt các kết luận còn hiệu lực. Bằng chứng và chi tiết nằm ở mục tương ứng.

1. Phải chia dữ liệu theo ảnh hoặc theo danh tính *trước*, rồi mới ghép cặp. Làm ngược lại thì
   dữ liệu bị rò rỉ giữa các tập và số liệu test cao giả tạo. *(2026-11-09)*
2. Ngưỡng được chọn trên tập val và báo cáo trên tập test, luôn kèm AUC và EER. Chọn ngưỡng
   tức là chọn cách đánh đổi giữa FAR và FRR. *(2026-11-09)*
3. Mạng Koch huấn luyện từ đầu trên LFW đạt AUC 0,853 và EER 21,9% với người chưa từng thấy.
   Mạng bị overfitting từ epoch 7. *(2026-11-09)*
4. Mô hình đó **không tổng quát hoá được sang ảnh webcam**: AUC giảm còn 0,607 và không có
   ngưỡng nào dùng được. Căn tỉ lệ khuôn mặt trong khung giống LFW là cần thiết nhưng chưa đủ.
   *(2026-11-10)*
5. Với GPU RTX 50 (sm_120), TensorFlow 2.20 không dùng được. Dự án huấn luyện bằng Keras 3 với
   backend PyTorch trong WSL2. *(2026-11-09)*
6. Thư mục chứa ảnh khuôn mặt phải có trong `.gitignore` *trước* khi chụp. Xoá ảnh khỏi commit
   mới nhất là chưa đủ, phải viết lại lịch sử git. *(2026-11-10)*
