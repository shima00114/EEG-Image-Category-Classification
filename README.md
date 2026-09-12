# EEG-Image-Category-Classification
PyTorch implementation for EEG-based visual object category classification.

# EEG-Image-Category-Classification

PyTorch implementation for EEG-based visual object category classification.

脳波データを用いた画像カテゴリ分類 (PyTorch)

※ 注意: 本リポジトリには脳波データセット本体および画像データは含まれていません。利用にあたっては [DATASET_NAME] のライセンスおよび利用規約に従ってください。

## 概要

多チャンネル時系列脳波シグナル (C: チャンネル数, T: タイムステップ数) を入力とし、画像カテゴリを予測するPyTorch実装です。

## ディレクトリ構造

```text
data/
├── raw/        # [DATASET_NAME] から取得した生データ
└── processed/  # 前処理済みのテンソルデータ (.pt)
```

## 環境構築

```bash
pip install -r requirements.txt
```

## 使用方法

1. データの前処理

```bash
python scripts/preprocess.py --data_dir data/raw --output_dir data/processed
```

2. モデルの学習

```bash
python train.py --config configs/default.yaml
```

3. 評価

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## 免責事項・著作権

- 本プロジェクトは学習目的で公開されています。
- 使用しているデータセット [DATASET_NAME] の著作権および権利は原著作者に帰属します。

---

# EEG-Image-Category-Classification (English)

PyTorch implementation for EEG-based visual object category classification.

Note: This repository does not contain the dataset itself. Please follow the license and terms of [DATASET_NAME].

## Overview

A PyTorch pipeline that predicts image categories from multi-channel time-series EEG signals (C: channels, T: time steps).

## Directory Structure

```text
data/
├── raw/        # Raw data from [DATASET_NAME]
└── processed/  # Processed tensor data (.pt)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Preprocessing

```bash
python scripts/preprocess.py --data_dir data/raw --output_dir data/processed
```

2. Training

```bash
python train.py --config configs/default.yaml
```

3. Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Disclaimer & Copyright

- This repository is for research and educational purposes only.
- All rights and copyrights of [DATASET_NAME] belong to their respective owners.
