# BullBullet - DogRun

BullBullet は、PyBullet を使用して四足歩行ロボットのシミュレーションを行うためのフレームワークです。歩容パターンの生成、障害物の回避、ゴールへの到達など、基本的なロボット制御タスクを実験できます。また、強化学習（PPO アルゴリズム）を用いた自律的な行動の獲得を目指します。

_開発者: Shun Aizawa_

## 特徴

- 複数の歩容パターン（トロット、ウォーク、バウンドなど）のサポート
- LiDAR センサーによる環境認識
- 障害物コースの自動生成（シンプル、高密度、ランダムの 3 種類）
- ゴール達成機能
- カスタマイズ可能な環境とロボットパラメータ
- コマンドラインからの簡単な操作
- 強化学習モード
  - PPO（Proximal Policy Optimization）アルゴリズムによる学習
  - カスタマイズ可能な報酬関数
  - トレーニングと評価の分離
  - 学習曲線の自動プロット
  - ログ出力

## インストール

### 前提条件

- Python 3.6 以上
- pip（Python パッケージマネージャー）

### 手順

1. リポジトリをクローン：

```bash
git clone https://github.com/yourusername/bull_bullet_dogrun.git
cd bull_bullet_dogrun
```

2. 依存パッケージのインストール：

```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な実行

通常のシミュレーションモードを実行するには：

```bash
python main.py
```

### コマンドラインオプション

BullBullet は様々なコマンドラインオプションをサポートしています：

#### 基本オプション

```bash
python main.py --config CONFIG_FILE   # 設定ファイルを指定
python main.py --export-default PATH  # デフォルト設定をエクスポート
python main.py --verbose              # 詳細なログ出力
python main.py --console-only         # ログをコンソールのみに出力
python main.py --quiet                # コンソールへのログ出力を抑制
```

#### シミュレーションモード

```bash
python main.py sim                    # 通常シミュレーション実行（デフォルト）
```

#### 強化学習モード

```bash
# トレーニングモード
python main.py rl --rl-mode train --epochs 100

# 評価モード
python main.py rl --rl-mode evaluate --load-model MODEL_PATH

# その他のオプション
python main.py rl --render            # トレーニング中も描画
python main.py rl --rl-config FILE    # 強化学習用設定ファイル
```

### 設定ファイル

BullBullet は 2 種類の設定ファイルを提供しています：

- `configs/config.yaml`: 通常のシミュレーション用
- `configs/rl_config.yaml`: 強化学習用

デフォルト設定をエクスポートするには：

```bash
python main.py --export-default configs/my_config.yaml
```

### 設定ファイルの構成

設定ファイルは以下のセクションで構成されています：

#### ロボット設定

```yaml
robot:
  urdf_path: models/urdf/svdog2_2_description/svdog2_2.urdf
  position: [0, 0, 0.08] # 初期位置 [x, y, z]
  rotation: [0, 0, 135] # 初期姿勢（オイラー角）[roll, pitch, yaw]
  max_force: 5.0 # アクチュエータの最大トルク
```

#### 環境設定

```yaml
environment:
  use_gui: true # GUIを使用するかどうか
  camera_follow: true # カメラがロボットに追従するかどうか
  gravity: [0, 0, -9.8] # 重力ベクトル [x, y, z]
  timestep: 0.00416667 # シミュレーションのタイムステップ (1/240)
```

#### LiDAR 設定

```yaml
lidar:
  enabled: true # LiDARを有効にするかどうか
  num_rays: 36 # レイの数
  ray_length: 1.0 # レイの最大長
  ray_start_length: 0.01 # レイの開始距離
  ray_color: [0, 1, 0] # 通常時のレイの色 [R, G, B]
  ray_hit_color: [1, 0, 0] # 衝突時のレイの色 [R, G, B]
```

#### 歩容設定

```yaml
gait:
  amplitude: 0.25 # 関節角度の振幅
  frequency: 1.5 # 歩行周期の周波数
  pattern: trot # 歩容パターン (trot, walk, bound)
  turn_direction: 0 # 旋回方向 (-1.0: 左, 0: 直進, 1.0: 右)
  turn_intensity: 0 # 旋回の強さ (0.0 - 1.0)
```

#### 障害物設定

```yaml
obstacles:
  enabled: true # 障害物を有効にするかどうか
  course_type: simple # コースタイプ (simple, dense, random)
  length: 5.0 # コースの長さ
```

#### ゴール設定

```yaml
goal:
  enabled: true # ゴールを有効にするかどうか
  position: [2.0, 0, 0] # ゴールの位置 [x, y, z]
  radius: 0.3 # ゴールの半径
  color: [0.0, 0.8, 0.0, 0.5] # ゴールの色 [R, G, B, A]
```

#### シミュレーション設定

```yaml
simulation:
  max_steps: 5000 # 最大シミュレーションステップ数
  debug_interval: 100 # デバッグ情報の表示間隔
```

## 歩容パターン

BullBullet は以下の歩容パターンをサポートしています：

- `trot`: 対角の脚を同時に動かす（デフォルト）
- `walk`: 各脚を順番に動かす
- `bound`: 前脚と後脚のペアを同時に動かす

歩容パターンは設定ファイルで指定できます：

```yaml
gait:
  pattern: "trot" # 'trot', 'walk', 'bound'のいずれかを指定
  amplitude: 0.25
  frequency: 1.5
```

## 障害物コース

BullBullet は複数のタイプの障害物コースを生成できます：

- `simple`: 基本的な障害物配置
- `dense`: 密に配置された障害物
- `random`: ランダムに配置された障害物

コースタイプは設定ファイルで指定できます：

```yaml
obstacles:
  enabled: true
  course_type: "simple" # 'simple', 'dense', 'random'のいずれかを指定
  length: 5.0
```

## プロジェクト構成

```
bull_bullet_dogrun/
├── configs/            # 設定ファイル
│   ├── config.yaml          # 通常シミュレーション設定
│   └── rl_config.yaml       # 強化学習設定
├── env/                # 環境関連のクラス
│   ├── config_loader.py     # 設定読み込み
│   ├── gait.py              # 歩容生成
│   ├── goal_marker.py       # ゴールマーカー
│   ├── lidar_sensor.py      # LiDARセンサー
│   ├── obstacle_generator.py # 障害物生成
│   ├── quad_env.py          # 四足環境基本クラス
│   └── simulation_runner.py  # シミュレーション実行クラス
├── models/             # ロボットモデル（URDF）
├── rl/                 # 強化学習関連
│   ├── evaluate.py          # 評価環境
│   ├── ppo_agent.py         # PPOエージェント実装
│   ├── rl_environment.py    # 強化学習環境
│   └── trainer.py           # 学習トレーナー
├── utils/              # ユーティリティ関数
│   ├── logger.py            # ロギング管理
│   └── logging_setup.py     # ロギング設定
├── logs/               # ログ出力ディレクトリ
├── results/            # 強化学習結果ディレクトリ
└── main.py             # メインエントリーポイント
```

## 強化学習モード

BullBullet は、PPO（Proximal Policy Optimization）アルゴリズムを用いた強化学習モードを提供しています。ロボットの制御方針を自動的に学習し、障害物を回避しながらゴールに到達する能力を獲得できます。

### 強化学習の実行

```bash
# トレーニング（新規モデル）
python main.py rl --rl-mode train --epochs 100

# 既存モデルからのトレーニング継続
python main.py rl --rl-mode train --load-model PATH --epochs 50

# 評価モード
python main.py rl --rl-mode evaluate --load-model PATH
```

### 強化学習の設定

強化学習の設定は `configs/rl_config.yaml` で行います：

```yaml
# 強化学習関連設定（一部抜粋）
obstacles:
  enabled: true
  course_type: random # 学習時にはランダムコースが効果的
  length: 8.0

goal:
  enabled: true
  position: [2.0, 0, 0]
  radius: 0.4
```

### 学習結果

トレーニング結果は `results/ppo_quadruped_TIMESTAMP/` ディレクトリに保存されます：

- `model_final.pt`: 学習済みモデル
- `metrics_final.json`: 学習メトリクス
- `learning_curves_final.png`: 学習曲線
- `evaluation_results.json`: 評価結果
- `hyperparameters.json`: ハイパーパラメータ

### PPO アルゴリズムの実装

BullBullet の PPO 実装は以下の特徴を持ちます：

- アクターとクリティックのネットワークを含む共有モデル
- GAE（Generalized Advantage Estimation）による安定した学習
- エントロピー正則化による探索促進
- KL ダイバージェンスに基づく学習制限
- 歩容の安定性を考慮した行動選択

## ライセンス

[MIT License](LICENSE)

## 貢献

大きな変更を加える前には、まず Issue を開いて議論してください。

## 作者

Shun Aizawa
