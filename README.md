# Action Recognition using Detectron2

本 repo 藉由人體骨架關鍵點的移動軌跡訓練一個動作識別的模型

## 開發環境

```
Python 3.7.6
pytorch=='1.6.0+cu101' # need cuda 10.1
detectron2==0.3
moviepy==1.0.3
numpy==1.20.1
pandas==0.25.3
```
## Dataset

[HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)有六千多個影片(video), 每個影片為一種人物動作, 共 51 個動作類別

## Code

- Step 0. 將下載的資料集壓縮檔解壓縮 格式整理
- Step 1. 探索如何從單一影像(image)擷取人體骨架關鍵點
- Step 2. 探索如何從單一影片(video)擷取人體骨架關鍵點
- Step 3. 資料集中的所有video資訊取得/整理
- Step 4. 用 detectron2 識別出人, 擷取人體骨架關鍵點, 並且利用物件追蹤方法找到影片中最值得關注的人物的關鍵點移動軌跡
- Step 5. 利用擷取出的關鍵點軌跡作為資料, 訓練一個RNN模型(pytorch實現)來做到動作分類的預測

## Result

test_set: accuracy ~40%
