# 2026-11-10 – Thử mô hình LFW trên ảnh webcam, quyền riêng tư và lịch sử git

## Bối cảnh

Hôm nay dùng mô hình huấn luyện ngày 2026-11-09 (lần chạy `20261109-201136`, chế độ `lfw`) để
xác minh một người thật bằng ảnh chụp từ webcam qua lệnh `collect`. Mô hình không được huấn
luyện lại.

## Thay đổi

- **Ảnh tham chiếu không còn lên git.**
  - Thêm `data/verification/` vào `.gitignore`.
  - Gỡ ảnh tham chiếu cũ của bản ứng dụng khỏi git; file vẫn giữ trên máy.
  - Thêm hướng dẫn chụp ảnh vào `data/README.md`.
- **`collect` vẽ một elip hướng dẫn ở giữa cửa sổ camera.**
  - Elip rộng 45% và cao 55% cạnh ảnh, đúng bằng cỡ khuôn mặt trong ảnh LFW.
  - Elip chỉ hiển thị trên màn hình, không có trong ảnh được lưu.
  - Thêm 2 test, nâng tổng số lên 62 test.
- **Dự án chuyển sang kho `tbm077861/siamese-face-verification`.** Kho cũ `AppFace` đã bị xoá.
- **Viết lại lịch sử git để xoá ảnh khuôn mặt cá nhân của bản ứng dụng cũ khỏi mọi commit.**
  - Ảnh đó là `faceapp/data/anchor/anchor.jpg`, về sau được đổi tên thành
    `data/verification/anchor.png`.
  - Nội dung mã ở commit mới nhất không đổi, nhưng mã hash của mọi commit đều đổi.
  - Các tài liệu đang trỏ tới commit "final" đã được cập nhật mã hash mới.
  - Kho trên GitHub được xoá rồi tạo lại cùng tên, để GitHub không còn lưu các commit cũ.
- **Tạo thư mục nhật ký này.**

## Thí nghiệm: mô hình LFW trên ảnh webcam

**Cách đo**
- Mỗi ảnh thử được so với mọi ảnh tham chiếu. Ghi lại điểm tương đồng và tỉ lệ ảnh tham chiếu
  khớp.
- Để ước lượng tỉ lệ chấp nhận nhầm, ảnh người lạ lấy ngẫu nhiên từ `data/negative/` (LFW) cũng
  được so với cùng bộ ảnh tham chiếu đó.
- Kích thước khuôn mặt trong ảnh được đo bằng bộ phát hiện Haar cascade của OpenCV.

**Lần 1 – chụp sát.** Vùng cắt 250×250; 14 ảnh tham chiếu, 5 ảnh thử, 40 người lạ.
- Khuôn mặt rộng khoảng 180–220 px trên khung 250 px. Trong ảnh LFW, khuôn mặt chỉ rộng
  khoảng 110 px.
- Cùng một ảnh thử, khi so với các ảnh tham chiếu khác nhau, cho điểm dao động từ 0,001 đến
  0,93.
- Với τ_d = 0,485:
  - 4/5 ảnh thử có tỉ lệ khớp 0,57–0,64, tức là đạt với τ_v = 0,5;
  - ảnh thử còn lại chỉ đạt 0,36.
- Người lạ có tỉ lệ khớp trung bình 0,17, nhưng cao nhất tới 0,71: người lạ đó sẽ được chấp
  nhận.

**Lần 2 – khung hình giống LFW.** Khuôn mặt nằm vừa trong elip, chạy `collect --crop-size 450`;
14 ảnh tham chiếu, 6 ảnh thử, 300 người lạ.
- ROC-AUC tính theo từng cặp (ảnh chủ thể so với ảnh người lạ) chỉ đạt **0,607**, trong khi trên
  tập test LFW là 0,853.
- Trung vị điểm của ảnh chủ thể là 0,43, của người lạ là 0,28. Tuy vậy, 5% người lạ có điểm trên
  0,90.

| τ_d | τ_v | Ảnh chủ thể được chấp nhận | Người lạ được chấp nhận |
|---:|---:|---:|---:|
| 0,40 | 0,5 | 5/6 | 117/300 (39%) |
| 0,485 | 0,5 | 0/6 | 97/300 (32%) |
| 0,70 | 0,7 | 0/6 | 37/300 (12%) |

- Điểm của người lạ không tương quan với độ sáng hay độ tương phản của ảnh (|r| ≤ 0,13).
- Những người lạ có điểm cao nhất (khoảng 0,95) nhìn bằng mắt không hề giống chủ thể.

## Kết luận rút ra

**Về thu thập ảnh**

1. **Cửa sổ OpenCV chỉ nhận phím khi đang được chọn.** Mỗi ảnh lưu thành công sẽ in một dòng ra
   terminal; dựa vào dòng này để biết ảnh đã được lưu.
2. **Tỉ lệ khuôn mặt so với khung phải khớp với dữ liệu huấn luyện.** Khuôn mặt gần kín khung
   làm điểm số dao động thất thường. Elip hướng dẫn giúp căn khung đúng.

**Về khả năng tổng quát hoá**

3. **Khung hình khớp là cần nhưng chưa đủ.** Ngay cả khi khung đã giống LFW, mô hình vẫn gần như
   không phân biệt được chủ thể với người lạ (AUC 0,607). Kết quả tốt trên tập test LFW không
   bảo đảm kết quả tốt trên dữ liệu thật ở nơi triển khai.
4. **Các nguyên nhân khả dĩ** (chưa kiểm chứng riêng từng yếu tố):
   - mạng CNN học từ đầu trên ảnh 100×100 cắt ở giữa, không phát hiện và căn chỉnh khuôn mặt,
     nên nền và quần áo cũng đi vào mạng;
   - dữ liệu khác miền: LFW là ảnh báo chí, còn đây là ảnh webcam (đeo kính, nền văn phòng, ánh
     đèn trần);
   - LFW mất cân bằng về nhân khẩu học (chủ yếu là người nổi tiếng phương Tây), nên mô hình có
     thể kém hơn với các nhóm người ít xuất hiện trong dữ liệu.

   *(Nguyên nhân thứ nhất đã được kiểm chứng ở mục
   [2026-11-10 – cắt khuôn mặt](2026-11-10-cat-khuon-mat.md): chỉ thêm bước phát hiện và cắt
   khuôn mặt, AUC trên ảnh webcam đã tăng từ 0,65 lên 0,90.)*
5. **Ngưỡng chọn trên tập val của LFW không dùng được cho ảnh webcam.** Ở τ_d = 0,485 (ngưỡng
   `eer`), lần 2 không có ảnh thử nào của chủ thể được chấp nhận.
6. **Người lạ lấy từ LFW chỉ là phép đo gần đúng.** Muốn đo FAR đúng nghĩa cho ứng dụng webcam,
   cần ảnh người lạ chụp bằng cùng webcam, trong cùng điều kiện.
7. **Chế độ `personal` có rủi ro "học đường tắt".**
   - Ảnh dương đều chụp từ webcam, còn ảnh âm đều lấy từ LFW.
   - Mạng có thể chỉ học cách phân biệt *nguồn ảnh* thay vì *danh tính*, và cho số liệu test
     cao giả tạo.
   - Trước khi tin vào kết quả của chế độ này, cần có ảnh âm cùng miền, tức là ảnh webcam của
     người khác.
8. **Thí nghiệm còn nhỏ:** chỉ có một chủ thể, 6 ảnh thử và 14 ảnh tham chiếu, nên các con số chỉ
   mang tính chỉ báo. Riêng kết luận về hướng đi (mô hình cần được cải thiện) thì rõ ràng.

**Về quyền riêng tư và git**

9. **Ảnh khuôn mặt là dữ liệu cá nhân.** Thư mục chứa ảnh tự chụp phải có trong `.gitignore`
   *trước* khi chụp tấm đầu tiên.
10. **Xoá file khỏi commit mới nhất không xoá được file khỏi lịch sử.**
    - Với kho công khai, ai cũng mở được các commit cũ.
    - Phải viết lại lịch sử bằng `git filter-branch` hoặc `git filter-repo`, rồi force-push.
    - Sau đó GitHub có thể vẫn hiển thị commit cũ theo mã hash trong một thời gian.
    - **Đã kiểm chứng:** sau khi force-push, API GitHub vẫn trả về commit "final" cũ và cả file
      ảnh, nếu gọi đúng mã hash. Nhánh `main` thì đã sạch. Muốn xoá hẳn, phải nhờ GitHub
      Support hoặc xoá kho đi rồi tạo lại.
    - Đã xoá kho rồi tạo lại cùng tên, sau đó đẩy lịch sử sạch lên. Từ đó API trả về "không tìm
      thấy" cho cả commit cũ lẫn file ảnh.
11. **Làm sạch lịch sử càng sớm càng tốt.** Khi kho chưa có fork, pull request hay người clone,
    việc viết lại lịch sử gần như không ảnh hưởng tới ai.
12. **Force-push an toàn:** dùng `git push --force-with-lease=<nhánh>:<hash mong đợi>`. Nếu ai đó
    vừa đẩy commit mới lên, lệnh sẽ tự từ chối thay vì xoá mất commit đó.
13. **Viết lại lịch sử làm đổi mọi mã hash.** Commit có chữ ký GPG cũng đổi mã hash, vì chữ ký bị
    bỏ đi. Vì vậy tài liệu nên tham chiếu thông điệp commit thay vì mã hash.
14. **Nhiều phiên làm việc song song trên cùng một thư mục có thể đổi nhánh hoặc viết lại lịch sử
    giữa chừng.**
    - Trước khi commit hoặc push, kiểm tra `git status -sb`, `git log --graph --oneline --all`
      và `git remote -v`.
    - Nếu cần sửa code song song, dùng `git worktree` để mỗi phiên có một thư mục riêng.
15. **Với kho còn mới, xoá rồi tạo lại kho là cách tự làm nhanh nhất để GitHub bỏ hẳn các commit
    cũ.** Cách này có hiệu lực ngay, không phải chờ GitHub Support.
    - Nếu kho đã có sao hoặc issue, xoá kho sẽ mất những thứ đó.
    - Nếu kho đã có fork, các fork vẫn giữ commit cũ.
    - Trong hai trường hợp này nên nhờ GitHub Support.

## Việc còn dở

- [x] Force-push nhánh `main` đã làm sạch lên GitHub, rồi kiểm tra trên GitHub rằng ảnh không
      còn trong lịch sử. *(Xong ngày 2026-11-10. Không commit nào trên `main` còn chứa ảnh.)*
- [x] Xoá hẳn các commit cũ còn được GitHub lưu: chúng vẫn mở được nếu biết đúng mã hash (xem
      kết luận 10). Có hai cách: nhờ GitHub Support, hoặc xoá kho rồi tạo lại. *(Xong ngày
      2026-11-10 bằng cách xoá kho rồi tạo lại.)*
- [x] Sau khi kiểm tra xong, dọn các bản dự phòng trên máy vẫn còn chứa ảnh, rồi chạy `git gc`:
      - nhánh `backup/before-rewrite`;
      - `refs/original/`;
      - remote `appface-old`.

      *(Xong ngày 2026-11-10. Không đối tượng nào trong kho trên máy còn chứa ảnh.)*
- [x] Quyết định hướng cải thiện mô hình. Đề xuất:
      - thêm bước phát hiện và cắt khuôn mặt (Haar cascade của OpenCV) cho cả huấn luyện, đánh
        giá và xác minh;
      - huấn luyện lại trên LFW;
      - so sánh trước và sau trên tập test LFW và trên ảnh webcam.

      *(Xong ngày 2026-11-10, xem [mục cắt khuôn mặt](2026-11-10-cat-khuon-mat.md).)*
- [x] Đưa phép đo "ảnh webcam so với người lạ" thành một lệnh của CLI để chạy lại được. Hiện
      phép đo này chỉ là một script tạm, chưa có trong kho. *(Xong ngày 2026-11-10: lệnh
      `field-test`.)*
