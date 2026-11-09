"""Chuẩn bị ảnh negative (ảnh của người khác), ví dụ từ bộ dữ liệu LFW.

Nguồn có thể là file nén (.tgz/.tar/.zip), một thư mục ảnh hoặc một URL. Ảnh được
"làm phẳng" vào một thư mục duy nhất, giữ nguyên tên file - LFW đặt tên dạng
``Ten_Nguoi_0001.jpg`` nên vẫn suy ra được danh tính khi chia tập dữ liệu.

Chỉ file ảnh được chép và chỉ lấy *tên file* (bỏ đường dẫn trong file nén), nên một
file nén độc hại cũng không thể ghi ra ngoài thư mục đích (path traversal).
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .config import UsageError
from .data import is_image_file


def prepare_negatives(source: str, dest: str | Path, limit: int | None = None) -> int:
    """Chép ảnh từ ``source`` vào ``dest``; trả về số ảnh *mới* được thêm."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if source.startswith(("http://", "https://")):
        archive = download(source, dest.parent / "downloads")
        return extract_images(archive, dest, limit)
    path = Path(source)
    if path.is_dir():
        return copy_images(path, dest, limit)
    if path.is_file():
        return extract_images(path, dest, limit)
    raise FileNotFoundError(f"Không tìm thấy nguồn dữ liệu: {source}")


def download(url: str, directory: Path) -> Path:
    """Tải file về ``directory`` (bỏ qua nếu đã có)."""
    directory.mkdir(parents=True, exist_ok=True)
    name = PurePosixPath(urllib.parse.urlparse(url).path).name or "negatives.tgz"
    target = directory / name
    if target.exists():
        print(f"Đã có {target}, không tải lại.")
        return target
    partial = target.with_name(target.name + ".part")
    print(f"Đang tải {url} -> {target} ...")
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)
    return target


def extract_images(archive: str | Path, dest: Path, limit: int | None = None) -> int:
    """Giải nén *chỉ các file ảnh* trong ``archive`` vào ``dest`` (bỏ cấu trúc thư mục)."""
    archive = Path(archive)
    count = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if _limit_reached(count, limit):
                    break
                target = _target_path(info.filename, dest)
                if info.is_dir() or target is None:
                    continue
                with zf.open(info) as source:
                    count += _save(source, target)
        return count
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                if _limit_reached(count, limit):
                    break
                target = _target_path(member.name, dest)
                if not member.isfile() or target is None:
                    continue
                source = tar.extractfile(member)
                if source is not None:
                    with source:
                        count += _save(source, target)
        return count
    raise UsageError(f"Định dạng không được hỗ trợ (cần .zip/.tar/.tgz): {archive}")


def copy_images(source_dir: Path, dest: Path, limit: int | None = None) -> int:
    """Chép (đệ quy) mọi ảnh trong ``source_dir`` vào ``dest``."""
    dest_resolved = dest.resolve()
    count = 0
    for path in sorted(source_dir.rglob("*")):
        if _limit_reached(count, limit):
            break
        if not path.is_file() or path.parent.resolve() == dest_resolved:
            continue
        target = _target_path(path.name, dest)
        if target is None:
            continue
        shutil.copy2(path, target)
        count += 1
    return count


def _limit_reached(count: int, limit: int | None) -> bool:
    return limit is not None and count >= limit


def _target_path(name: str, dest: Path) -> Path | None:
    """Đường dẫn đích (chỉ dùng tên file); ``None`` nếu không phải ảnh hoặc đã tồn tại."""
    filename = PurePosixPath(name.replace("\\", "/")).name
    if not is_image_file(filename):
        return None
    target = dest / filename
    return None if target.exists() else target


def _save(source, target: Path) -> int:
    with open(target, "wb") as out:
        shutil.copyfileobj(source, out)
    return 1
