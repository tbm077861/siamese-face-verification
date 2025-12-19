# Hướng Dẫn Cài Đặt và Sử Dụng Ứng Dụng Face Recognition (Nhận Diện Khuôn Mặt)

## 📋 Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cài Đặt](#cài-đặt)
4. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
5. [Các Tính Năng Chính](#các-tính-năng-chính)
6. [Khắc Phục Sự Cố](#khắc-phục-sự-cố)

---

## 🎯 Giới Thiệu

**Face Recognition Application** là ứng dụng nhận diện khuôn mặt sử dụng công nghệ **Siamese Neural Network** (mạng nơ-ron Siamese) để xác thực danh tính người dùng thông qua camera. Ứng dụng cung cấp:

- ✅ Xác thực khuôn mặt bằng camera real-time
- ✅ Lưu trữ ảnh tài liệu và ảnh cá nhân
- ✅ Giao diện thân thiện, dễ sử dụng
- ✅ Sử dụng mô hình học sâu (Deep Learning) tiên tiến

---

## 🖥️ Yêu Cầu Hệ Thống

### Phần Cứng:
- **CPU**: Intel Core i5 trở lên (hoặc tương đương)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Ổ cứng**: 2GB dung lượng trống
- **Camera**: Webcam hoặc camera tích hợp
- **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux

### Phần Mềm:
- **Python**: Phiên bản 3.7 trở lên (khuyến nghị 3.9 hoặc 3.10)
- **pip**: Trình quản lý gói Python

---

## 📦 Cài Đặt

### Bước 1: Cài Đặt Python

1. Truy cập trang web chính thức: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Tải xuống **Python 3.9** hoặc **3.10**
3. Chạy trình cài đặt và **đánh dấu** ✓ "Add Python to PATH"
4. Nhấp "Install Now"

**Kiểm tra cài đặt:**
```bash
python --version
pip --version
```

### Bước 2: Sao Chép Kho Lưu Trữ

Nếu chưa có, sao chép hoặc tải xuống thư mục dự án vào máy tính của bạn.

### Bước 3: Cài Đặt Thư Viện Phụ Thuộc

1. Mở **Command Prompt** hoặc **PowerShell**
2. Điều hướng đến thư mục dự án:
   ```bash
   cd "d:\GitHub Repository\Face_recognition_application\faceapp"
   ```
3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

**Các thư viện sẽ được cài đặt:**
- **Kivy**: Framework xây dựng giao diện người dùng
- **OpenCV (cv2)**: Xử lý hình ảnh và video từ camera
- **TensorFlow**: Framework học máy
- **NumPy**: Xử lý dữ liệu số

### Bước 4: Xác Minh Mô Hình

Ứng dụng sử dụng mô hình Siamese được lưu trong file `siamesemodel.h5`. Đảm bảo file này có trong thư mục `faceapp`:
- File: `siamesemodel.h5` (lưu ý: file này có thể nặng, ~100-200MB)

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Chạy Ứng Dụng

1. Mở **Command Prompt** hoặc **PowerShell**
2. Điều hướng đến thư mục ứng dụng:
   ```bash
   cd "d:\GitHub Repository\Face_recognition_application\faceapp"
   ```
3. Chạy ứng dụng:
   ```bash
   python main.py
   ```
4. Cửa sổ ứng dụng sẽ mở ra

### Bước 2: Giao Diện Chính

Ứng dụng hiển thị:
- **Tiêu đề**: "Face Authentication System"
- **Camera View**: Khung nhìn camera real-time
- **Capture & Authenticate**: Nút để chụp ảnh và xác thực
- **Trạng thái**: Thông báo về kết quả xác thực

### Bước 3: Xác Thực Khuôn Mặt

1. **Chuẩn bị**:
   - Đảm bảo camera của bạn được bật
   - Định vị khuôn mặt của bạn vào khung camera
   - Đảm bảo đủ ánh sáng

2. **Chụp ảnh**:
   - Nhấp nút **"Capture & Authenticate"**
   - Ứng dụng sẽ chụp ảnh từ camera

3. **Xác thực**:
   - Ứng dụng so sánh ảnh chụp với ảnh anchor được lưu trữ
   - Hiển thị kết quả: **MATCH** (khớp) hoặc **NO MATCH** (không khớp)
   - Xem thông báo trạng thái cập nhật

---

## ✨ Các Tính Năng Chính

### 1. **Xác Thực Khuôn Mặt Real-time**
- Camera phát trực tiếp
- Chụp và xác thực ngay lập tức
- Hiển thị kết quả trực quan

### 2. **Quản Lý Dữ Liệu**
Ứng dụng tổ chức dữ liệu trong các thư mục:
- **data/anchor/**: Lưu ảnh anchor (ảnh tham chiếu)
- **photos/**: Lưu ảnh cá nhân
- **uploaded_documents/**: Lưu tài liệu đã tải lên
- **uploaded_photos/**: Lưu ảnh đã tải lên

### 3. **Mô Hình Siamese Neural Network**
- Sử dụng mạng nơ-ron Siamese để so sánh khuôn mặt
- Tính toán khoảng cách L1 giữa các embedding
- Độ chính xác cao trong nhận diện

---

## 🛠️ Khắc Phục Sự Cố

### ❌ Lỗi: "No module named 'kivy'"
**Giải pháp:**
```bash
pip install kivy
```

### ❌ Lỗi: "No module named 'tensorflow'"
**Giải pháp:**
```bash
pip install tensorflow
```
⚠️ *Lưu ý: Cài đặt TensorFlow có thể mất vài phút*

### ❌ Camera không hoạt động
**Giải pháp:**
1. Kiểm tra xem camera đã được bật trong cài đặt hệ thống
2. Kiểm tra quyền truy cập camera cho Python
3. Thử khởi động lại ứng dụng

### ❌ Ứng dụng chạy chậm
**Giải pháp:**
- Đảm bảo máy tính có đủ RAM (ít nhất 4GB)
- Đóng các ứng dụng khác đang chạy
- Cập nhật driver GPU (nếu có)

### ❌ Lỗi: "siamesemodel.h5 not found"
**Giải pháp:**
- Đảm bảo file `siamesemodel.h5` có trong thư mục `faceapp`
- File này là mô hình đã được huấn luyện trước
- Nếu thiếu, tải từ nguồn dự án hoặc huấn luyện lại mô hình

### ❌ Lỗi về phiên bản Python
**Giải pháp:**
```bash
pip install --upgrade tensorflow opencv-python
```

---

## 💡 Mẹo và Lời Khuyên

1. **Ánh sáng**: Sử dụng ứng dụng ở nơi có ánh sáng tốt để độ chính xác cao nhất
2. **Tư thế**: Hướng thẳng vào camera và tránh che khuôn mặt
3. **Khoảng cách**: Duy trì khoảng cách 30-60cm từ camera
4. **Nhiều lần thử**: Nếu xác thực thất bại lần đầu, thử lại
5. **Cập nhật**: Giữ Python và các thư viện được cập nhật

---

## 📧 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra lại các bước cài đặt
2. Xem phần "Khắc Phục Sự Cố"
3. Tham khảo tài liệu chính thức:
   - [Kivy Documentation](https://kivy.org/doc/stable/)
   - [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
   - [OpenCV Documentation](https://docs.opencv.org/)

---

## 📝 Ghi Chú

- Ứng dụng sử dụng **Kivy** cho giao diện đa nền tảng
- Mô hình nhận diện khuôn mặt dựa trên **Siamese Network**
- Dữ liệu được lưu trữ cục bộ trên máy tính của bạn
- Ứng dụng không yêu cầu kết nối Internet

---

**Cập nhật lần cuối:** Tháng 12, 2025
