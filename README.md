# EC_final_project

## STGP

### `ExtractContent.py`
This is for extracting contents from original mail.
原始email可能為.eml檔案，裡面有很多資訊，此程式主要將content部分提取出來
```
Usage:
python3 ExtractContent.py
input source path
input dest. path
```

### `extract_features.py`
This is for extracting features(words) from contents.<br>
主要進行：
1. 把裡面的字都撈出來(`create_bag_of_words()`)
2. 數目少於20的會先移除(`cutoff參數`)
3. 移除stop words(`remove_stop_words()`)，例如a, the, at, ...，從`nltk`這個套件拿
4. 詞性還原(`lemmatization()`)，例如works、working都變回work。使用`pattern3`套件，不過這套件有bug，例如this會變成thi XD
5. 將那些詞彙轉換成出現頻率(`regularize_vectors()`)
6. 在`spam-mail.csv`中有label是spam(0)或ham(1)，把每筆email的label附加到每個vector。
7. 輸出成`trainX.csv` (一部份可拿來`testX.csv`)

```
Usage:
python3 extract_features.py
input source path
```

### `STGP.py`
主程式，會產生`tree.pdf`，此為最佳的individual
```
Usage:
python3 STGP.py
```

安裝pygraphviz時，用以下指令：
```
pip3 install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
```