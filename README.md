# バスケットボール・シュート分析システム：技術仕様書

**プロジェクト状況:** 高度研究開発 / 進行中 (WIP)
**技術スタック:** Python, YOLOv8, OpenCV, Dxcam, NumPy, Pandas
**最終更新:** 約6ヶ月前
**作成者:** Rishit

---

## 1. プロジェクト概要 & システムアーキテクチャ

本プロジェクトは、単なる物体検出（Object Detection）を超え、「時間的ロジック（Temporal Logic）」**と**「物理演算」を組み込んだシュート分析システムです。

YOLOv8によるフレームごとの検出に加え、状態遷移マシンと線形回帰アルゴリズムを使用することで、バスケットボールのシュートの物理挙動を理解し、その成否（Make/Miss）を自動判定します。

### 「NBA 2K データファクトリー」構想
本プロジェクトの真の目的は、**高品質な学習データを「無限に」生成する自動化ループの構築**です。
現実の映像データ（20GB以上）の手動収集・ラベリングのコストを解決するため、NBA 2K25（CPU vs CPU）の試合を「無限のデータソース」として利用します。

1. **無限生成:** ゲーム内の正確な物理挙動を利用し、多様なシュートシーンを自動生成。
2. **自動ラベリング:** ロジック判定（`utils.py`）が「シュート」と認識したシーケンスを、信頼できる正解データとして自動保存。
3. **自己進化:** 蓄積されたデータでモデルを再学習させ、精度を向上させるループを回します。

---

## 2. モジュール構成と役割詳細

### A. 推論・検出プログラム (`Programs/`)

| ファイル名 | 役割と特徴 |
| :--- | :--- |
| **`shot_detector.py`** | **【安定版・動画分析用】**<br>標準的なMP4動画ファイルの分析用です。入力動画を **1280x736** にリサイズします。<br>*技術メモ:* YOLOv8は入力サイズが32の倍数（Stride 32）であることを要求するため、720p (720px) ではなく736pxを採用し、警告と精度低下を防いでいます。 |
| **`shot_detector_game-2.py`** | **【データファクトリー・ソース】**<br>NBA 2K25分析に特化した高速版。<br>**Dxcam**を使用してGPUから直接フレームを取得し、CPU vs CPUの試合を監視します。ここで検出されたシュートデータが、将来的な再学習の源泉となります。 |
| **`shot_detector_FP.py`** | **【誤検知 (False Positive) 対策研究用】**<br>ネットの揺れや観客席の光をボールと誤認する問題に対処するための実験コード。<br>YOLOの信頼度閾値を `0.15` まで下げて「見逃し」を無くした上で、形状（正方形度）や移動履歴に基づく厳格なロジックフィルタでノイズを除去します。 |
| **`shot_detector_max.py`** | **【高精度デバッグモード】**<br>非同期処理を排除したシングルスレッドモード。<br>**Pキーで一時停止/再生**が可能になっており、「なぜこのシュートが外れ判定になったのか？」を1フレームずつ目視確認するために開発されました。 |
| **`shot_detector_manual.py`** | **【正解データ作成用】**<br>動画を見ながら手動で **Sキー(成功)** / **Fキー(失敗)** を押すことで、モデルの精度検証に使うための「正解ラベル（Ground Truth）」CSVを作成します。 |

### B. 統計分析・検証 (`ExtraPrograms/`)
* **`GroundVsYOLO_ROC.py`**: 手動ラベルとAI判定を比較し、ROC曲線を生成して精度を検証します。
* **`yoloDSexport.py`**: **【データセットパッケージャー】**<br>自動収集された画像とラベルを「学習用(80%)」「検証用(20%)」に自動分割し、YOLO学習用のZIPファイルを作成するMLOpsツールです。

---

## 3. ディレクトリ構造

```text
Project Root
├── models/
│   ├── Rishit.onnx      # メインモデル (速度重視・ONNX形式)
│   ├── RokkenV2.pt      # サブモデル (別データセットで学習・PT形式)
├── Programs/            # メイン実行スクリプト群
│   ├── shot_detector.py
│   ├── shot_detector_game-2.py
│   └── utils.py
├── ExtraPrograms/       # 統計・検証・データセット作成
├── TestPrograms/        # 可視化・実験用スクリプト
├── Results/             # 出力データ保存先
│   └── {VideoName}/
│       └── {ModelName}_shot_results.csv  # 分析結果CSV
├── HoopVids/            # 入力動画ソース (MP4)
└── Runs/                # 学習ログとチェックポイント


## 4. 実行コマンド (Quick Start)

### 必要なライブラリのインストール:
```bash
pip install ultralytics opencv-python cvzone numpy dxcam
```

### 実行例

#### 1. 動画ファイルの分析 (標準):
```bash
python Programs/shot_detector.py --model "Rishit.onnx" --video "DNvsTW.mp4"
```

#### 2. ゲーム画面の分析 (NBA 2K25 - 高速版):
ゲームをモニタ0（メイン）または1（サブ）で起動した状態で実行してください。

```bash
python Programs/shot_detector_game-2.py --model "Rishit.onnx" --name "NBA2K_Session"
```

#### 3. 正解データの作成 (手動ラベリング):
```bash
python Programs/shot_detector_manual.py --model "Rishit.pt" --video "TestGame.mp4"
```

#### 4. ROC曲線の生成 (精度検証):
```bash
python ExtraPrograms/ROC_curve_create.py
```

---

## 5. 出力データ形式

分析結果は `Results/{VideoName}/{ModelName}_shot_results.csv` に保存されます。

### CSVカラム構成:
| カラム名 | 説明 |
| :--- | :--- |
| `frame_number` | シュートが検出されたフレーム番号 |
| `result` | 判定結果 (`Make` / `Miss`) |
| `confidence` | YOLOモデルの検出信頼度 (0.0~1.0) |
| `ball_trajectory` | ボールの軌跡座標リスト (JSON形式) |
| `rim_position` | リムの中心座標 `(x, y)` |

---

## 6. パフォーマンス最適化

### リアルタイム処理のための技術
1. **Dxcam使用:** Windows環境でのGPUダイレクトキャプチャにより、OpenCVより約30%高速化
2. **ONNX形式:** PyTorch (.pt) 形式に比べて推論速度が約2倍向上
3. **適応的リサイズ:** 入力解像度を1280x720に制限し、4K環境でも安定した60FPS動作を実現

### 推奨システム要件
* **GPU:** NVIDIA GTX 1060 以上 (CUDA対応)
* **RAM:** 8GB以上
* **OS:** Windows 10/11 (Dxcam使用時)

---

## 7. 既知の制限事項と今後の課題

### 現在の制限
* **カメラアングル依存:** 真横や真上からの視点では精度が低下します
* **複数ボール:** 画面内に複数のボールが存在する場合、追跡が不安定になります
* **照明条件:** 極端な逆光や暗所では検出精度が落ちます

### 開発中の機能 (WIP)
* **3D軌道再構成:** ステレオカメラや深度推定を用いた3次元空間での軌道分析
* **選手認識統合:** シュートした選手の識別と個人統計の自動集計
* **リアルタイムダッシュボード:** Web UIによるライブ統計表示

---

## 8. 貢献とライセンス

**作成者:** Rishit  
**プロジェクト状態:** 研究開発中 (WIP)  
**最終更新:** 約6ヶ月前

本プロジェクトは研究・教育目的で開発されています。商用利用については作成者にご相談ください。

---

## 9. トラブルシューティング

### よくある問題と解決策

**Q: "CUDA out of memory" エラーが発生する**  
A: `shot_detector.py` 内の `imgsz` パラメータを640に下げるか、バッチサイズを削減してください。

**Q: ゲーム画面が認識されない**  
A: Dxcamが正しいモニタをキャプチャしているか確認してください。`--monitor` パラメータで指定できます。

**Q: 誤検知が多い**  
A: `shot_detector_FP.py` の使用を検討してください。または `conf_threshold` を0.15から0.25に上げることで精度が向上する場合があります。

---

## 10. 参考資料

* **YOLOv8 Documentation:** https://docs.ultralytics.com/
* **OpenCV Python Tutorial:** https://docs.opencv.org/
* **Dxcam GitHub:** https://github.com/ra1nty/DXcam

---

**最終更新日:** 2025年1月  
**バージョン:** 1.0  
**お問い合わせ:** プロジェクトのIssueページまで