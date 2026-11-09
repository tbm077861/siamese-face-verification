"""Xác minh khuôn mặt one-shot bằng mạng nơ-ron Siamese.

Gói chỉ chứa phần khoa học máy tính của dự án: dữ liệu, mô hình, huấn luyện,
đánh giá và suy luận. Các module dùng TensorFlow (``model``, ``pipeline``,
``train``, ``verify``) chỉ được import khi cần, nhờ vậy phần thuần Python
(``data``, ``metrics``, ``negatives``) chạy và kiểm thử được mà không cần GPU
hay TensorFlow.
"""

__version__ = "2.0.0"
