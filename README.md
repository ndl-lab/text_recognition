# Text recognition
画像中のテキストを認識するプロジェクトのリポジトリです。 
本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託 して作成したものです。

## 依存関係
元リポジトリ(https://github.com/clovaai/deep-text-recognition-benchmark)をカスタマイズした[deep-text-recognition-benchmark](https://github.com/ndl-lab/deep-text-recognition-benchmark)に依存している
```bash
git clone https://https://github.com/ndl-lab/deep-text-recognition-benchmark
cd deep-text-recognition-benchmark
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install tqdm lmdb opencv-python six natsort nltk more-itertools
```

## text_recognition.pyの使い方
- モデルとデータセットのディレクトリを受け取り推論を行う
- 入力可能なデータ形式はxmlとimgを子に持つディレクトリか、create_xmldataset.py後のデータセット

### Accuracyを出力する
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/best_norm_ED_ndl2.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --acc
```

### Normalized Edit Distanceを出力する
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/best_norm_ED_ndl2.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --leven
```

### どの文字を間違えているか可視化
- --outimage_dir行は画像として出力するオプション
- --stat行はどの文字をどの文字として間違えているかの統計を出力するオプション
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/best_norm_ED_ndl2.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --diff wrong \
    --outimage_dir outimg --font font.ttf \
    --stat
```

### XMLとして出力する(create_xmldataset.py後のデータセットには非対応)
```bash
python text_recognition.py \
    $(cat arg_train-model_info) \
    --saved_model models/best_norm_ED_ndl2.pth \
    --character "〓$(cat models/mojilist_NDL.txt | tr -d '\n')" \
    --batch_max_length 100 --imgW 1200 --PAD \
    --batch_size 160 \
    --db_path input_dir/ --db_type xmlraw \
    --xml --outxml_dir outxml
```

## ファイル
- text_recognition.py
    - モデルとデータセットのディレクトリを受け取り推論を行う
    - 入力可能なデータ形式はxmlとimgを子に持つディレクトリか、create_xmldataset.py後のデータセット
    - 標準出力形式はdiff, acc, levenの三種類
    - `python text_recognition.py --db_path input_dir/ --db_type xmlraw $(cat arg_train-model_info) --character "$(cat data/charset | tr -d '\n')" --batch_max_length 100 --imgW 1200 --imgH 32 --PAD --saved_model models/best_accuracy.pth --batch_size 32 --diff wrong
- create_xmldataset.py
    - xmlとimgを子に持つディレクトリを指定して、学習に使用するデータベースを作成する
    - `python create_xmldataset.py --input_path data/sample/??_大衆人事録?之部/ --output_path databases/train/大衆人事録 databases/valid/大衆人事録 databases/test/大衆人事録


#### arg_train-model_info
```:arg_train-model_info
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC
```
