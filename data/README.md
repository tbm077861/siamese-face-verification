# Dữ liệu

| Thư mục | Nội dung | Cách tạo | Đưa lên git? |
|---|---|---|---|
| `anchor/` | Ảnh khuôn mặt của người cần xác minh | `python -m siamese_face collect` (phím `a`) | Không |
| `positive/` | Ảnh khác của chính người đó | `collect` (phím `p`) | Không |
| `negative/` | Ảnh của người khác | `python -m siamese_face prepare-negatives <nguồn>` | Không |
| `verification/` | Ảnh tham chiếu dùng khi xác minh (`verify`) | `collect` (phím `v`) hoặc chép tay | Có |
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

## Gợi ý khi thu thập ảnh anchor/positive

- Thu tối thiểu khoảng 300 ảnh anchor và 300 ảnh positive.
- Giữ khuôn mặt ở giữa khung hình (vùng cắt 250×250 hiện trên cửa sổ camera).
- Thay đổi góc mặt, khoảng cách, ánh sáng, biểu cảm, có và không đeo kính.
- Thu ảnh qua nhiều phiên khác nhau (khác ngày, khác ánh sáng), vì các khung hình chụp
  liên tiếp gần như trùng nhau.

## `verification/anchor.png`

Đây là ảnh tham chiếu duy nhất của phiên bản ứng dụng cũ. File thực chất là ảnh PNG
1151×2560 nên đã được đổi đuôi từ `.jpg` sang `.png`. Vì là ảnh dọc, khi suy luận ảnh sẽ bị
cắt vùng vuông ở giữa. Nên thay hoặc bổ sung bằng các ảnh chính diện chụp bằng `collect`
(phím `v`).
