# 自然言語で対象を選ぶロボットアーム

日本語の指示に応じて対象の箱を選び、ロボットアームがその上を指し示すシミュレーションである。Physical AI応用1講座の最終課題として作成した。

## 実行例

以下の3つの指示で動作を確認した。

| 入力指示 | AIの選択 |
|---|---|
| トマトと同じ色の箱を指して | 赤 |
| 赤い箱ではなく青い箱を指して | 青 |
| バナナと同じ色の箱を指して | 黄 |

3例すべてで意図した色を選択し、アーム先端が目標位置から3 cm未満の範囲に到達した。

## 仕組み

学習済み文章埋め込みモデル
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
を使用する。

入力指示と、以下の比較用候補文をベクトル化し、コサイン類似度が最大の候補の色を選択する。

- 赤い箱を指してください。
- 青い箱を指してください。
- 黄色い箱を指してください。

選択した色に対応する既知の座標を目標とし、PyBulletの逆運動学と位置制御でKUKA iiwaの先端を移動させる。追加学習は行わず、学習済みモデルの推論を利用している。

## 実行方法

Python 3.11を想定する。初回はライブラリと学習済みモデルのダウンロードにインターネット接続が必要である。CPUで実行でき、APIキーは不要である。

```bash
pip install -r requirements.txt
python robot_demo.py
```

`commands.json`を`robot_demo.py`と同じディレクトリに置くことで、実行する指示を指定できる。

日本語字幕の表示にはNoto Sans CJKフォントが必要である。Ubuntuでは以下で導入できる。

```bash
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk
```

## ファイル構成

- `robot_demo.py`：AIによる対象選択、シミュレーション、動画生成
- `commands.json`：入力指示
- `requirements.txt`：必要なPythonライブラリ
- `README.md`：概要と実行方法

## 出力

実行すると、`pai_output`ディレクトリに以下を生成する。

- `robot_demo.mp4`：字幕付き実行動画（標準の3指示では70秒）
- `results.json`：AIの選択結果、類似度、目標座標、到達誤差
- `preview.png`：動画の初期フレーム

## 制約・今後の課題

- 箱の色と位置は事前に設定しており、画像認識は行っていない。
- 把持や物体の移動は行わない。
- 類似度は正解確率ではない。
- 候補外の指示でも、必ず3色のいずれかを選択する。
- 今回の3例での成功は、任意の連想や否定表現への対応を保証しない。

今後は、指示の言い換えや曖昧な表現で評価し、対象を判断できない場合に選択を保留する仕組みを検討する。

## 参考・利用素材

- [PyBullet](https://github.com/bulletphysics/bullet3)
  - ロボットモデルは`pybullet_data`同梱のKUKA iiwaを使用。
- [学習済み文章埋め込みモデル](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [当初参考にしたROS 2・Gazeboの記事](https://qiita.com/N622/items/9d89f77d85d9da0af29e)
  - 本実装ではPyBulletを使用している。

利用したライブラリ・モデル・素材には、それぞれのライセンスが適用される。