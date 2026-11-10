# Dữ liệu

| Thư mục | Nội dung | Cách tạo | Đưa lên git? |
|---|---|---|---|
| `anchor/` | Ảnh khuôn mặt của người cần xác minh | `python -m siamese_face collect` (phím `a`) | Không |
| `positive/` | Ảnh khác của chính người đó | `collect` (phím `p`) | Không |
| `negative/` | Ảnh của người khác | `python -m siamese_face prepare-negatives <nguồn>` | Không |
| `verification/` | Ảnh tham chiếu dùng khi xác minh (`verify`) | `collect` (phím `v`) hoặc chép tay | Không |
| `downloads/` | File nén tải về khi nguồn là URL | tự sinh | Không |

Ảnh có thể ở định dạng `.jpg`, `.jpeg`, `.png` hoặc `.bmp`.

## Ảnh negative: bộ dữ liệu LFW

*Labeled Faces in the Wild* gồm 13.233 ảnh khuôn mặt của 5.749 người, mỗi ảnh 250×250.
Tải file `lfw.tgz` từ trang chính thức <http://vis-www.cs.umass.edu/lfw/> hoặc từ một bản
sao (ví dụ trên Kaggle), rồi chạy:

```powershell
python -m siamese_face prepare-negatives D:\Downloads\lfw.tgz
# hoặc từ một thư mục đã giải nén / một URL:
python -m siamese_face prepare-negatives D:\Downloads\lfw
```

Lệnh chỉ chép file ảnh và giữ nguyên tên dạng `Ten_Nguoi_0001.jpg`. Tên này được dùng để
chia tập dữ liệu theo danh tính, nên một người lạ chỉ xuất hiện trong đúng một tập
train, val hoặc test.

## Lưu ý khi chụp ảnh bằng `collect`

Mô hình học từ ảnh LFW: khung vuông, mặt ở giữa, chiếm khoảng nửa chiều cao khung, thấy
cả tóc và một phần vai. Ảnh chụp càng giống như vậy thì kết quả càng tốt.

- **Click vào cửa sổ camera trước khi bấm phím.** OpenCV chỉ nhận phím `a`/`p`/`v`/`q`
  khi cửa sổ camera đang được chọn; mỗi ảnh lưu thành công sẽ in một dòng ra terminal.
- **Khung hình:** vùng hiện trên cửa sổ chính là vùng được lưu. Đặt mặt (trán đến cằm)
  vừa trong **hình elip xanh** - đúng cỡ mặt trong ảnh LFW (khoảng 45% cạnh ảnh). Nếu mặt
  to hơn elip, hãy lùi xa webcam hoặc tăng vùng cắt, ví dụ `collect --crop-size 450`.
  Elip chỉ để hiển thị, không có trong ảnh lưu. Với mô hình huấn luyện bằng
  `--preprocess face`, ảnh được cắt quanh khuôn mặt trước khi so khớp, nên khoảng cách tới
  camera ít quan trọng hơn; vẫn nên giữ mặt ở giữa và nhìn thẳng.
- **Ánh sáng:** đều, chiếu từ phía trước; tránh ngồi ngược sáng hoặc đèn gắt một bên.
- **Rõ nét:** không che mặt, không có người khác trong khung, giữ yên đầu khi bấm.
- **Bấm từng lần, không giữ phím** (giữ phím sẽ lưu hàng chục khung gần như trùng nhau);
  mỗi lần đổi nhẹ góc mặt hoặc biểu cảm.

Số lượng:

- Ảnh tham chiếu (`v`) cho mô hình LFW: 5–10 ảnh chính diện, chụp trong điều kiện giống
  lúc sẽ xác minh.
- Chế độ `personal` (`a`/`p`): khoảng 300 ảnh mỗi loại, thay đổi góc mặt (15–30°), biểu
  cảm, có và không đeo kính, và chụp qua nhiều phiên khác nhau (khác ngày, khác ánh
  sáng). Nếu chụp hết trong một phiên, tập test gần như trùng tập train và số liệu đánh
  giá sẽ cao giả tạo.

## Quyền riêng tư

Mọi ảnh trong `anchor/`, `positive/`, `negative/` và `verification/` đều bị `.gitignore`
bỏ qua, chỉ nằm trên máy. Ảnh `verification/anchor.png` của phiên bản ứng dụng cũ (ảnh
dọc 1151×2560, không phù hợp làm ảnh tham chiếu) đã được gỡ khỏi git nhưng vẫn còn trong
lịch sử các commit cũ.
