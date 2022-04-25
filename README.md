# NDLOCR用テキスト認識モジュール
画像中のテキストを認識するモジュールのリポジトリです。 

本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については
[LICENSE](./LICENSE
)をご覧ください。

## 環境構築
python3.7かつ、cuda 11.1をインストール済みの環境の場合
text_recognitionディレクトリ直下で以下のコマンドを実行する。
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install tqdm lmdb opencv-python six natsort nltk more-itertools
wget https://lab.ndl.go.jp/dataset/ndlocr/text_recognition/mojilist_NDL.txt -P ./models
wget https://lab.ndl.go.jp/dataset/ndlocr/text_recognition/ndlenfixed64-mj0-synth1.pth -P ./models
```

くわえて、元リポジトリ(https://github.com/clovaai/deep-text-recognition-benchmark)
をカスタマイズした[deep-text-recognition-benchmark](https://github.com/ndl-lab/deep-text-recognition-benchmark)
に依存しているため、下記のようにリポジトリの追加と環境変数の追加を行う。

```bash
git clone https://github.com/ndl-lab/deep-text-recognition-benchmark
cd deep-text-recognition-benchmark
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## ファイルの説明
- text_recognition.py
    - モデルとデータセットのディレクトリを受け取り推論を行う
    - 入力可能なデータ形式はxmlとimgを子に持つディレクトリか、create_xmldataset.py後のデータセット
    - 標準出力形式はdiff, acc, levenの三種類
```bash
python text_recognition.py --db_path input_dir/ --db_type xmlraw $(cat arg_train-model_info) --character "$(cat data/charset | tr -d '\n')" --batch_max_length 100 --imgW 1200 --imgH 32 --PAD --saved_model models/best_accuracy.pth --batch_size 32 --diff wrong
```

- create_xmldataset.py
    - NDLOCRXMLDataset形式のxmlとimgを子に持つディレクトリを指定して、学習に使用するデータベースを作成する
```bash
python create_xmldataset.py --input_path data/sample/??_大衆人事録?之部/ --output_path databases/train/大衆人事録 databases/valid/大衆人事録 databases/test/大衆人事録
```

## 使い方
- モデルとデータセットのディレクトリを受け取り推論を行う
- 入力可能なデータ形式はNDLOCRXMLDataset形式のxmlとimgを子に持つディレクトリか、create_xmldataset.pyで処理した後のデータセット

### Accuracyを出力する機能
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/ndlenfixed64-mj0-synth1.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --acc
```

### Normalized Edit Distanceを出力する機能
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/ndlenfixed64-mj0-synth1.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --leven
```

### どの文字を間違えているか可視化する機能
- --outimage_dir行は画像として出力するオプション
- --stat行はどの文字をどの文字として間違えているかの統計を出力するオプション
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/ndlenfixed64-mj0-synth1.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --diff wrong \
    --outimage_dir outimg --font font.ttf \
    --stat
```

### XMLとして出力する機能
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/ndlenfixed64-mj0-synth1.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --xml --outxml_dir outxml
```




#### (参考)学習時のパラメータ([deep-text-recognition-benchmark](https://github.com/ndl-lab/deep-text-recognition-benchmark)を参照)
```:arg_train-model_info
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC
```
