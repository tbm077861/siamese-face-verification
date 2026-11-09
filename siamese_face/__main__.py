"""Cho phép chạy ``python -m siamese_face <lệnh>``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
