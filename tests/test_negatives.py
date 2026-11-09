import io
import tarfile
import zipfile

import pytest

from siamese_face.negatives import prepare_negatives


def _make_tar(path, members: dict[str, bytes]):
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def test_tar_extraction_flattens_filters_and_is_safe(tmp_path):
    archive = tmp_path / "lfw.tgz"
    _make_tar(
        archive,
        {
            "lfw/Alice_Smith/Alice_Smith_0001.jpg": b"a",
            "lfw/Bob_Jones/Bob_Jones_0002.jpg": b"b",
            "lfw/README.txt": b"khong phai anh",
            "../escape.jpg": b"x",  # path traversal: phải nằm trong thư mục đích
        },
    )
    dest = tmp_path / "data" / "negative"
    assert prepare_negatives(str(archive), dest) == 3
    assert sorted(p.name for p in dest.iterdir()) == [
        "Alice_Smith_0001.jpg",
        "Bob_Jones_0002.jpg",
        "escape.jpg",
    ]
    assert not (tmp_path / "data" / "escape.jpg").exists()
    # Chạy lại không chép trùng
    assert prepare_negatives(str(archive), dest) == 0


def test_zip_extraction_respects_limit(tmp_path):
    archive = tmp_path / "faces.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(5):
            zf.writestr(f"faces/Person{i}_0001.png", b"img")
    dest = tmp_path / "negative"
    assert prepare_negatives(str(archive), dest, limit=2) == 2
    assert len(list(dest.iterdir())) == 2


def test_copy_from_directory(tmp_path):
    source = tmp_path / "source"
    (source / "A").mkdir(parents=True)
    (source / "A" / "A_0001.jpg").write_bytes(b"a")
    (source / "notes.txt").write_text("x")
    dest = tmp_path / "negative"
    assert prepare_negatives(str(source), dest) == 1
    assert (dest / "A_0001.jpg").read_bytes() == b"a"


def test_unsupported_file_and_missing_source(tmp_path):
    bogus = tmp_path / "not_an_archive.bin"
    bogus.write_bytes(b"hello")
    with pytest.raises(ValueError):
        prepare_negatives(str(bogus), tmp_path / "negative")
    with pytest.raises(FileNotFoundError):
        prepare_negatives(str(tmp_path / "missing.tgz"), tmp_path / "negative")
