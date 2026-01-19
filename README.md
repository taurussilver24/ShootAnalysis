# バスケットボール・シュート分析システム

**プロジェクト状況:** 高度研究開発 / 進行中 (WIP)  
**技術スタック:** Python, YOLOv8, OpenCV, Dxcam, NumPy, Pandas  
**作成者:** アロラリシット  
**最終更新:** 2026年1月

---

## 📋 目次

1. [プロジェクト概要](#プロジェクト概要)
2. [NBA 2K データファクトリー構想](#nba-2k-データファクトリー構想)
3. [コアロジックの仕組み](#コアロジックの仕組み)
4. [モジュール構成](#モジュール構成)
5. [ディレクトリ構造](#ディレクトリ構造)
6. [インストールと実行](#インストールと実行)
7. [出力データ形式](#出力データ形式)
8. [自動学習ループ](#自動学習ループ)
9. [パフォーマンス最適化](#パフォーマンス最適化)
10. [制限事項と今後の課題](#制限事項と今後の課題)
11. [トラブルシューティング](#トラブルシューティング)
12. [参考資料](#参考資料)

---

## プロジェクト概要

本プロジェクトは、単なる物体検出（Object Detection）を超えた、**時間的ロジック（Temporal Logic）と物理演算を組み込んだシュート分析システム**です。

YOLOv8によるフレームごとの検出に加え、状態遷移マシン（State Machine）と線形回帰アルゴリズムを使用することで、バスケットボールのシュートの物理挙動を理解し、その成否（Make/Miss）を自動判定します。

### 主要機能

- ✅ リアルタイムシュート検出（60FPS対応）
- ✅ Make/Miss自動判定
- ✅ NBA 2K25ゲームプレイ分析対応
- ✅ 軌道予測・可視化
- ✅ 統計データ自動生成
- ✅ ROC曲線による精度検証

---

## NBA 2K データファクトリー構想

本プロジェクトの真の目的は、**高品質な学習データを「無限に」生成する自動化ループの構築**です。

現実の映像データ（20GB以上）の手動収集・ラベリングのコストを解決するため、**NBA 2K25（CPU vs CPU）**の試合を「無限のデータソース」として利用します。

### データファクトリーの3つの柱

1. **無限生成**  
   ゲーム内の正確な物理挙動を利用し、多様なシュートシーンを自動生成

2. **自動ラベリング**  
   ロジック判定（`utils.py`）が「シュート」と認識したシーケンスを、信頼できる正解データとして自動保存

3. **自己進化**  
   蓄積されたデータでモデルを再学習させ、精度を向上させるループを回す

---

## コアロジックの仕組み

システムの「頭脳」にあたる `utils.py` は、以下の3段階のプロセスでシュートを認識します。

### ステップ1: 検出とクリーニング

YOLOモデルがフレーム毎に「ボール」と「リング」を検出し、物理フィルタリングを実行します。

- **物理フィルタリング:** `clean_ball_pos` 関数により、物理的にあり得ない動きを除外
  - 例：5フレーム以内に直径の4倍以上移動したボールは「テレポート（ノイズ）」とみなして無視
- **持続性確認:** 一定フレーム数以上連続して検出されたオブジェクトのみを追跡

### ステップ2: 状態遷移マシン

リングに対するボールの相対位置を常に監視し、以下のシーケンスが発生した時のみ「シュート試行」とみなします。

1. **フェーズ1 (UP):** ボールがリングの「上」の領域に侵入（`detect_up`）
2. **フェーズ2 (DOWN):** その後、ボールがリングの「下」の領域へ通過（`detect_down`）

この `UP → DOWN` の順序が守られない限り、単に画面内をボールが横切ってもシュートとは判定されません。

### ステップ3: スコアリング判定

シュート試行が確定した瞬間にトリガーされます。

- **線形回帰:** `np.polyfit` を使用して、ボールの直近の軌跡から落下軌道を計算し、予測直線を引く
- **交差判定:** その予測線がリングの座標内（リムの内側）を通過していれば「成功（Make）」、外れていれば「失敗（Miss）」と判定

---

## モジュール構成

### A. 推論・検出プログラム (`Programs/`)

| ファイル名 | 役割と特徴 |
|:-----------|:----------|
| `shot_detector.py` | **【安定版・動画分析用】**<br>標準的なMP4動画ファイルの分析用。入力動画を **1280x736** にリサイズ。<br>*技術メモ:* YOLOv8は入力サイズが32の倍数（Stride 32）であることを要求するため、720p (720px) ではなく736pxを採用し、警告と精度低下を防止。 |
| `shot_detector_game-2.py` | **【データファクトリー・ソース】**<br>NBA 2K25分析に特化した高速版。**Dxcam**を使用してGPUから直接フレームを取得し、CPU vs CPUの試合を監視。検出されたシュートデータが、将来的な再学習の源泉となる。 |
| `shot_detector_FP.py` | **【誤検知対策研究用】**<br>ネットの揺れや観客席の光をボールと誤認する問題に対処。YOLOの信頼度閾値を `0.15` まで下げて「見逃し」を無くした上で、形状（正方形度）や移動履歴に基づく厳格なロジックフィルタでノイズを除去。 |
| `shot_detector_max.py` | **【高精度デバッグモード】**<br>非同期処理を排除したシングルスレッドモード。**Pキーで一時停止/再生**が可能で、「なぜこのシュートが外れ判定になったのか？」を1フレームずつ目視確認するために開発。 |
| `shot_detector_manual.py` | **【正解データ作成用】**<br>動画を見ながら手動で **Sキー(成功)** / **Fキー(失敗)** を押すことで、モデルの精度検証に使うための「正解ラベル（Ground Truth）」CSVを作成。 |

### B. 統計分析・検証 (`ExtraPrograms/`)

| ファイル名 | 役割 |
|:-----------|:-----|
| `GroundVsYOLO_ROC.py` | 手動ラベルとAI判定を比較し、ROC曲線を生成して精度を検証 |
| `yoloDSexport.py` | **【データセットパッケージャー】**<br>自動収集された画像とラベルを「学習用(80%)」「検証用(20%)」に自動分割し、YOLO学習用のZIPファイルを作成するMLOpsツール |
| `Confusion_matrix.py` | 混同行列を生成し、Accuracy、Precision、Recallを算出 |
| `aggCM.py` / `aggRoC.py` | 複数の動画やデータセットにまたがる統計データを集計 |

### C. 可視化・詳細分析 (`TestPrograms/`)

| ファイル名 | 役割 |
|:-----------|:-----|
| `prepare_sideview_data.py` | カメラ映像（2D）からシュートの「横方向の軌道（Side View）」を推定・生成し、アーチの高さや入射角を分析 |
| `render_video.py` | 分析結果の軌道ラインや統計データを、元の動画にオーバーレイ描画してレンダリング |
| `visualize_simple.html` | 生成されたJSONデータをブラウザ上でインタラクティブに確認するためのビューワー |

---

## ディレクトリ構造

```text
Project Root
├── models/
│   ├── Rishit.onnx          # メインモデル (速度重視・ONNX形式)
│   └── RokkenV2.pt          # サブモデル (別データセットで学習・PT形式)
│
├── Programs/                # メイン実行スクリプト群
│   ├── shot_detector.py
│   ├── shot_detector_game-2.py
│   ├── shot_detector_FP.py
│   ├── shot_detector_max.py
│   ├── shot_detector_manual.py
│   └── utils.py             # コアロジック実装
│
├── ExtraPrograms/           # 統計・検証・データセット作成
│   ├── GroundVsYOLO_ROC.py
│   ├── yoloDSexport.py
│   ├── Confusion_matrix.py
│   ├── aggCM.py
│   └── aggRoC.py
│
├── TestPrograms/            # 可視化・実験用スクリプト
│   ├── prepare_sideview_data.py
│   ├── render_video.py
│   └── visualize_simple.html
│
├── Results/                 # 出力データ保存先
│   └── {VideoName}/
│       └── {ModelName}_shot_results.csv
│
├── HoopVids/               # 入力動画ソース (MP4)
│
└── Runs/                   # 学習ログとチェックポイント
```

---

## インストールと実行

### 必要なライブラリのインストール

```bash
pip install ultralytics opencv-python cvzone numpy pandas dxcam
```

### 実行例

#### 1. 動画ファイルの分析（標準）

```bash
python Programs/shot_detector.py --model "Rishit.onnx" --video "DNvsTW.mp4"
```

#### 2. ゲーム画面の分析（NBA 2K25 - データ収集モード）

ゲームをモニタ0（メイン）または1（サブ）で起動した状態で実行してください。

```bash
python Programs/shot_detector_game-2.py --model "Rishit.onnx" --name "NBA2K_Session"
```

#### 3. 正解データの作成（手動ラベリング）

```bash
python Programs/shot_detector_manual.py --model "Rishit.pt" --video "TestGame.mp4"
```

操作方法:
- **Sキー:** 成功（Make）をマーク
- **Fキー:** 失敗（Miss）をマーク
- **Qキー:** 終了

#### 4. ROC曲線の生成（精度検証）

```bash
python ExtraPrograms/GroundVsYOLO_ROC.py
```

---

## 出力データ形式

安定版スクリプトは `Results/{VideoName}/{ModelName}_shot_results.csv` に以下の形式で保存します。

### CSVカラム構成

| カラム名 | データ型 | 説明 |
|:---------|:---------|:-----|
| `Shot Taken` | Integer | シュートのID（連番） |
| `Result` | String | 判定結果（`Successful` / `Failed`） |
| `Ball Coordinates` | String | シュート瞬間のボール中心座標 `(x, y)` |
| `Hoop Coordinates` | String | リングの中心座標 `(x, y)` |
| `Current Score` | String | その時点でのスコア表記（例: "3 / 5"） |
| `Video Timing` | Float | 動画内の秒数（タイムスタンプ） |

### サンプルデータ

```csv
Shot Taken,Result,Ball Coordinates,Hoop Coordinates,Current Score,Video Timing
1,Successful,"(640, 360)","(640, 200)",1 / 1,12.5
2,Failed,"(620, 370)","(640, 200)",1 / 2,18.3
3,Successful,"(650, 355)","(640, 200)",2 / 3,24.7
```

---

## 自動学習ループ

本システムの最終目標である「自動学習」は以下のフローで設計されています。

### Data Factory Loop の5ステップ

```
1. Source
   ↓ shot_detector_game-2.py がNBA 2Kの試合を監視
   
2. Trigger
   ↓ utils.py の物理演算ロジックが、ボールの UP → DOWN 挙動を検知
   
3. Auto-Labeling
   ↓ 検知されたフレームとバウンディングボックスを「正解データ」として自動エクスポート
   ↓ ※現在 render_video.py に実装されている機能を統合中
   
4. Packaging
   ↓ yoloDSexport.py がデータを学習用フォーマットに変換・圧縮
   
5. Training
   ↓ 新しいデータセットでモデルを再学習し、精度を向上
   
   → ループ継続
```

### 実装状況

- ✅ ステップ1-2: 完成
- 🚧 ステップ3: 統合作業中（`render_video.py` → `game-2.py`）
- ✅ ステップ4: 完成
- 🚧 ステップ5: 手動実行可能、自動化は未実装

---

## パフォーマンス最適化

### リアルタイム処理のための技術

1. **Dxcam使用**  
   Windows環境でのGPUダイレクトキャプチャにより、OpenCVより約30%高速化

2. **ONNX形式**  
   PyTorch (.pt) 形式に比べて推論速度が約2倍向上

3. **適応的リサイズ**  
   入力解像度を1280x720に制限し、4K環境でも安定した60FPS動作を実現

4. **非同期処理**  
   `shot_detector_game-2.py` では検出と描画を並列化し、遅延を最小化

### 推奨システム要件

| 項目 | 要件                         |
|:-----|:---------------------------|
| **GPU** | NVIDIA GTX 2060 以上（CUDA対応） |
| **RAM** | 8GB以上                      |
| **OS** | Windows 10/11（Dxcam使用時）    |
| **Python** | 3.8以上                      |


---

## 制限事項と今後の課題

### 現在の制限

- **カメラアングル依存:** 真横や真上からの視点では精度が低下します
- **複数ボール:** 画面内に複数のボールが存在する場合、追跡が不安定になります
- **照明条件:** 極端な逆光や暗所では検出精度が落ちます
- **リアルタイム性:** CPU環境では60FPS動作が困難

### 開発中の機能（WIP）

- 🚧 **自動エクスポートの統合**  
  `render_video.py` のエクスポート機能を `game-2.py` に完全移植し、放置によるデータ収集を実現

- 🚧 **モデル対決**  
  「旧モデル」vs「2K強化モデル」を同一動画で競わせ、精度の向上率を可視化

- 📋 **3D軌道再構成**  
  ステレオカメラや深度推定を用いた3次元空間での軌道分析

- 📋 **選手認識統合**  
  シュートした選手の識別と個人統計の自動集計

- 📋 **リアルタイムダッシュボード**  
  Web UIによるライブ統計表示

---

## トラブルシューティング

### よくある問題と解決策

#### Q: "CUDA out of memory" エラーが発生する

**A:** `shot_detector.py` 内の `imgsz` パラメータを640に下げるか、バッチサイズを削減してください。

```python
# 修正例
model = YOLO('Rishit.onnx')
results = model(frame, imgsz=640)  # 1280から640に変更
```

#### Q: ゲーム画面が認識されない

**A:** Dxcamが正しいモニタをキャプチャしているか確認してください。`--monitor` パラメータで指定できます。

```bash
# モニタ1を使用する場合
python Programs/shot_detector_game-2.py --model "Rishit.onnx" --monitor 1
```

#### Q: 誤検知が多い

**A:** `shot_detector_FP.py` の使用を検討してください。または `conf_threshold` を0.15から0.25に上げることで精度が向上する場合があります。

```python
# utils.py 内で調整
CONF_THRESHOLD = 0.25  # 0.15から変更
```

#### Q: FPSが低い

**A:** 以下の対策を試してください：

1. 解像度を下げる（1280x720 → 640x480）
2. ONNX形式のモデルを使用
3. GPU推論を有効化（CUDAインストール）
4. Dxcamを使用（Windows環境）

#### Q: CSVファイルが生成されない

**A:** 書き込み権限を確認し、`Results/` ディレクトリが存在することを確認してください。

```bash
# ディレクトリを手動作成
mkdir -p Results/TestVideo
```

---

## 参考資料

### 公式ドキュメント

- **YOLOv8 Documentation:** https://docs.ultralytics.com/
- **OpenCV Python Tutorial:** https://docs.opencv.org/
- **Dxcam GitHub:** https://github.com/ra1nty/DXcam
- **NumPy Documentation:** https://numpy.org/doc/

### 関連論文・技術

- **Object Detection:** "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)
- **State Machines:** "Introduction to the Theory of Computation" (Sipser, 2012)
- **Linear Regression:** "The Elements of Statistical Learning" (Hastie et al., 2009)

### コミュニティ

- **Issues & Bug Reports:** プロジェクトのGitHub Issuesページ
- **ディスカッション:** プロジェクトのDiscussionsページ

---
