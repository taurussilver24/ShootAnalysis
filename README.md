## YOLOv8オブジェクト検出プロジェクトのREADME

**プロジェクト概要**

このプロジェクトは、YOLOオブジェクト検出モデルを使用してバスケットボールのビデオを分析し、バスケットボールとリングを検出します。プロジェクトの主要なコンポーネントは、以下の2つのPythonスクリプトです。

1. **main.py**: `config.yaml` 設定ファイルを使用して YOLO モデルをトレーニングし、最良のモデルを `runs/weights` ディレクトリに `best.pt` として保存します。
2. **shot_detector.py**: トレーニング済みの YOLO モデル (`Yolo-Weights/best.pt`) を使用して、ビデオ内のバスケットボールとリングを検出します。ボールの軌跡を分析してシュート成功/失敗を判定し、結果を CSV ファイル (`shot_results.csv`) に記録します。

**ファイルディレクトリ構成**

```
├── .dvcignore
├── .gitignore
├── config.yaml
├── cvat.yaml
├── getContents.py
├── HoopVids
    ├── .filen.trash.local
    ├── Done_Requested
├── HoopVids.dvc
├── models
    ├── 1220v1.pt
├── Programs
    ├── clearGPUCache.py
    ├── csv_gen.py
    ├── main.py
    ├── ROC_curve_create.py
    ├── ROC_curve_create_case2.py
    ├── shot_detector.py
    ├── shot_detector_manual.py
    ├── shot_detector_test.py
    ├── utils.py
    ├── __pycache__
        ├── utils.cpython-312.pyc
├── README.md
├── requirements.txt
├── Results
├── runs
├── Yolo-Weights
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