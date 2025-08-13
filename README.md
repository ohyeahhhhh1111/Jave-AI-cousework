# Jave-AI-cousework

在這個專案當中，我利用了Java從原理開始撰寫多層感知機。其中我在資料前處理的部分做了標準化，並讓每個元素介於0~1。
接著在訓練的途中還多新增了Batch Training、Dropout以及step的機制，以此來讓訓練的效果更好。
為了兼顧模型的效能，我採用了輸入層以及一層隱藏層和輸出層，並特別設置超參數Hidden來設定隱藏層的大小。
在資料進入模型後給出的輸出也理所當然地有被拿去做反向傳播。
在訓練MLP的前置作業都設定好後，我開始做訓練，起初我發現結果並非如預期般的優秀。因此我使用了多種回歸函數來做測試例如L2Loss與Cross Entropy Loss。

以下是我測試出來最好的超參數

Hidden Size = 128, Output Size = 10, Learning Rate = 0.1314, step = 450, drop = 0.0, epoch = 500, batch size = 32, Loss function = CrossEntropyLoss
