## README.md File for YOLOv8 Object Detection Project

**Project Overview**

This project utilizes a YOLO object detection model to analyze a Basketball video and identify basketballs and rings. The main components of the project are two Python scripts:

1. **main.py**: Trains the YOLO model using the `config.yaml` configuration file and saves the best model as `best.pt` in the `runs/weights` directory.
2. **shot_detector.py**: Detects balls and rings in a video using the trained YOLO model (`Yolo-Weights/best.pt`). It analyzes the ball's trajectory to determine whether it makes or misses a shot, and records the results in a CSV file (`shot_results.csv`).

**File Directory Structure**

```
root_dir
├── Yolo-Weights
│   ├── best.pt
│   └── best6.pt
├── config.yaml
├── HoopVids
│   ├── DNvsTW.mp4
└── shot_detector.py
└── shot_results.csv
```

**Instructions**

**1. Training**

**Prerequisites:** Install the required libraries listed inside `requirements.txt`.

**Run:** `python main.py`

**2. Shot Detection**

**Prerequisites:** Install the required libraries listed inside `requirements.txt`.

**Run:** `python shot_detector.py --model Yolo-Weights/best1.pt --video HoopVids/DNvsTW.mp4`

**Replace `best1.pt` and `DNvsTW.mp4` with your desired model and video paths, respectively.**

**Additional Notes**

- The `config.yaml` file contains the training configuration parameters for the YOLO model.
- The `shot_detector.py` script includes functions for:
    - Loading the YOLO model
    - Processing video frames
    - Detecting objects (balls and rings)
    - Determining shot attempts and makes/misses
    - Saving results to a CSV file

**Further Development**

The current model result is not even close to the ideal results, so the utmost priority is to find the suitable training material, and keep on fine tuning the best model.

**References**

- [YOLO Object Detection](https://pjreddie.com/darknet/yolov1/)
- [PyTorch](https://pytorch.org/)
- [Cv2](https://opencv.org/)

**Original GitHub Project:** [https://github.com/github](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker)

**日本語版**

## YOLOv8オブジェクト検出プロジェクトのREADME

**プロジェクト概要**

このプロジェクトは、YOLOオブジェクト検出モデルを使用してバスケットボールのビデオを分析し、バスケットボールとリングを検出します。プロジェクトの主要なコンポーネントは、以下の2つのPythonスクリプトです。

1. **main.py**: `config.yaml` 設定ファイルを使用して YOLO モデルをトレーニングし、最良のモデルを `runs/weights` ディレクトリに `best.pt` として保存します。
2. **shot_detector.py**: トレーニング済みの YOLO モデル (`Yolo-Weights/best.pt`) を使用して、ビデオ内のバスケットボールとリングを検出します。ボールの軌跡を分析してシュート成功/失敗を判定し、結果を CSV ファイル (`shot_results.csv`) に記録します。

**ファイルディレクトリ構成**

```
root_dir
├── Yolo-Weights
│   ├── best.pt
│   └── best6.pt
├── config.yaml
├── HoopVids
│   ├── DNvsTW.mp4
└── shot_detector.py
└── shot_results.csv
```

**実行手順**

**1. トレーニング**

**前提条件:** `requirements.txt` に記載されているライブラリをインストールしてください。

**実行:** `python main.py`

**2. ショット検出**

**前提条件:** `requirements.txt` に記載されているライブラリをインストールしてください。

**実行:** `python shot_detector.py --model Yolo-Weights/best1.pt --video HoopVids/DNvsTW.mp4`

**`best1.pt` と `DNvsTW.mp4` を、ご希望のモデルパスとビデオパスに置き換えてください。**

**補足事項**

- `config.yaml` ファイルには、YOLO モデルのトレーニング設定パラメータが記述されています。
- `shot_detector.py` スクリプトには、以下の機能が含まれています。
    - YOLO モデルの読み込み
    - ビデオフレームの処理
    - オブジェクト (バスケットボールとリング) の検出
    - シュート