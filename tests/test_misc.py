"""Kiểm thử không cần TensorFlow: cấu hình, CLI, cắt ảnh webcam."""

import json
import math

import numpy as np
import pytest

from siamese_face.cli import build_parser
from siamese_face.collect import center_crop, draw_guide, guide_box
from siamese_face.config import TrainConfig, write_json


def test_train_config_roundtrip(tmp_path):
    config = TrainConfig(epochs=3, augment=True, max_images=100)
    path = tmp_path / "config.json"
    config.save(path)
    assert TrainConfig.load(path) == config


def test_write_json_handles_nan_and_numpy(tmp_path):
    path = tmp_path / "x.json"
    write_json(
        {"a": math.nan, "b": [math.inf, 1.5], "c": "tiếng Việt", "d": np.float32(0.5), "e": np.int64(3)},
        path,
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "a": None,
        "b": [None, 1.5],
        "c": "tiếng Việt",
        "d": 0.5,
        "e": 3,
    }


def test_center_crop_takes_middle_square():
    frame = np.arange(480 * 640).reshape(480, 640)
    crop = center_crop(frame, 250)
    assert crop.shape == (250, 250)
    assert crop[0, 0] == frame[115, 195]


def test_center_crop_clamps_to_frame():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    assert center_crop(frame, 250).shape == (200, 200, 3)


def test_guide_box_is_centered_and_lfw_sized():
    x1, y1, x2, y2 = guide_box(250)
    assert (x1 + x2) / 2 == 125 and (y1 + y2) / 2 == 125
    assert 100 <= x2 - x1 <= 125  # mặt LFW rộng ~110 px trên ảnh 250 px


def test_draw_guide_does_not_modify_saved_crop():
    crop = np.zeros((250, 250, 3), dtype=np.uint8)
    preview = draw_guide(crop)
    assert not crop.any() and preview.any()


def test_cli_train_defaults_match_config():
    args = build_parser().parse_args(["train"])
    defaults = TrainConfig()
    assert (args.epochs, args.batch_size, args.learning_rate) == (
        defaults.epochs,
        defaults.batch_size,
        defaults.learning_rate,
    )
    assert args.augment is False and args.patience == defaults.patience
    assert args.mode == "personal"
    assert build_parser().parse_args(["train", "--mode", "lfw"]).mode == "lfw"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--mode", "abc"])


def test_cli_evaluate_needs_exactly_one_source():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--run", "a", "--model", "b"])
    assert parser.parse_args(["evaluate", "--run", "runs/x"]).criterion == "f1"


def test_cli_verify_uses_verification_folder_by_default():
    args = build_parser().parse_args(["verify", "--model", "m.keras", "--image", "x.jpg"])
    assert args.references == ["data/verification"]


def test_cli_backend_option():
    parser = build_parser()
    assert parser.parse_args(["train"]).backend is None
    assert parser.parse_args(["--backend", "torch", "train"]).backend == "torch"
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "abc", "train"])


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cli_preprocess_and_new_commands():
    parser = build_parser()
    assert parser.parse_args(["train"]).preprocess == "center"
    assert parser.parse_args(["train", "--preprocess", "face"]).preprocess == "face"
    assert parser.parse_args(["verify", "--model", "m", "--image", "x"]).preprocess == "auto"
    assert parser.parse_args(["evaluate", "--run", "r"]).preprocess == "auto"
    args = parser.parse_args(["field-test", "--model", "m"])
    assert args.references == ["data/verification"] and args.probes == ["data/anchor"]
    assert (args.impostors, args.num_impostors, args.preprocess) == ("data/negative", 300, "auto")
    assert parser.parse_args(["compare", "a", "b"]).runs == ["a", "b"]
    with pytest.raises(SystemExit):
        parser.parse_args(["compare", "a"])
