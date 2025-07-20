

# **專家指南：修改YOLOv8以實現無實體關鍵點偵測——以羽球場交點偵測為例**

## **第一節：概念框架：從關節姿態到無實體關鍵點**

本節旨在建立理論基礎，將使用者提出的問題從一個標準的電腦視覺任務，重新定義為一個更專業化的挑戰。此處將闡述為何簡單套用YOLOv8-Pose並不足以解決問題，並為後續的客製化修改奠定理論依據。

### **1.1 YOLOv8-Pose 패러다임：以物件為中心、具關節結構的關鍵點**

標準的Ultralytics YOLOv8-Pose模型，其核心設計理念是基於「由上而下」(top-down)的偵測策略。此方法論的第一步是透過物件偵測，在影像中定位出一個具體的實例（例如，「人」），並以邊界框（bounding box）將其框出。隨後，模型會在這個已識別的邊界框內部，回歸（regress）出一系列預先定義的關鍵點（例如，人體的各個關節）的精確位置。  
此框架的架構也反映了這個兩階段的邏輯流程。它包含一個用於從輸入影像中提取深層特徵的骨幹網路（Backbone），一個用於聚合與融合不同層級特徵的頸部網路（Neck），以及一個負責最終預測的多任務頭部（Head）。這個頭部網路本身又被劃分為兩個子任務：其一負責物件偵測，輸出邊界框位置與物件類別；其二則專門負責關鍵點回歸，輸出每個關鍵點的座標與信賴度。  
YOLOv8-Pose模型通常在如COCO Keypoints這樣的大型資料集上進行預訓練。這些資料集的標註特性，例如COCO中的單一「person」類別及其對應的17個關節點，深深地塑造了模型的內在假設。其根本假設是：所有關鍵點都依附於一個可被偵測的「物件」之上，這個物件為關鍵點的定位提供了上下文（context）與尺度（scale）資訊。此外，這些關鍵點並非孤立存在，而是構成一個具有關節性（articulated）的結構，模型在訓練過程中會隱性地學習到這些關鍵點之間的空間相對關係。

### **1.2 使用者的挑戰：獨立、無實體的關鍵點**

使用者所面臨的任務——偵測羽球場上的T字、十字、L角交點——在根本上顛覆了YOLOv8-Pose的設計前提。這些交點並非某個單一、大型物件（如羽球場）的附屬部分，而是散佈於影像中的、各自獨立的特徵點。它們是「無實體」(disembodied)的關鍵點，不存在一個明確的、可供偵測的「父物件」來為它們提供上下文。  
若嘗試直接應用標準的YOLOv8-Pose模型，將會遭遇概念與效率上的雙重困境。我們不期望模型先偵測出整個羽球場的邊界框，然後再於框內搜尋所有的交點。這種做法不僅效率低下，也與問題的本質不符。我們真正的目標是將每一個線條交點視為一個獨立的、自成一體的實體來進行偵測與分類。  
因此，本報告提出的核心策略是：對YOLOv8-Pose框架進行深度改造，使其能夠處理這種新型態的任務。我們將引導模型將每一個交點視為一個特殊的「物件」，而這個「物件」的全部構成就是一個帶有類別屬性的單一關鍵點。在此策略下，傳統意義上的邊界框雖然在資料格式層面仍被要求存在，但其語意將被徹底虛化，成為一個模型在學習過程中會逐漸忽略的、最小化的佔位符。

### **1.3 深度洞察：無實體化帶來的核心技術挑戰**

將關鍵點「無實體化」的核心技術挑戰，源於將關鍵點偵測與其父物件邊界框進行解耦的過程。YOLOv8-Pose的預設損失函數（Loss Function）與評估指標（Evaluation Metrics）都與邊界框的面積（area）有著密不可分的關聯，這種依賴性是我們必須克服的首要技術障礙。  
其邏輯鏈如下：

1. YOLOv8-Pose的標準關鍵點損失函數KeypointLoss，在計算損失值時，會使用物件邊界框的面積來對關鍵點的座標誤差進行正規化（normalization）。這樣的設計是為了確保尺度不變性：在一個小尺寸的人物上產生5個像素的誤差，其懲罰應遠大於在一個大尺寸人物上產生同樣5個像素的誤差。  
2. 同樣地，姿態估計任務中最權威的評估指標——物件關鍵點相似度（Object Keypoint Similarity, OKS），其計算公式也包含了以物件尺度（通常是其分割區域面積的平方根）作為正規化因子。  
3. 然而，在我們的羽球場交點偵測任務中，這些「物件」（即T、十字、L角）本身沒有有意義的、可變的尺度。一個L角無論在影像的哪個位置、佔據多大，其本身的幾何定義是不變的。  
4. 因此，若我們在標註資料時為每一個交點都使用一個極小的、固定尺寸的邊界框作為佔位符，那麼無論是標準的損失函數還是OKS評估指標，其計算結果都將被嚴重扭曲且不具參考價值。正規化因子將會是一個恆定且武斷的數值，完全喪失了其應有的尺度調節功能。  
5. **結論：** 上述分析導出一個必然的結論——我們必須徹底重新設計損失函數，移除其對邊界框面積的依賴。同時，我們需要棄用OKS，轉而採用一個更適合此任務的評估指標，例如基於像素距離的「正確關鍵點百分比」（Percentage of Correct Keypoints, PCK）。這一根本性的認知將指導本報告後續所有關於模型修改、損失函數設計與評估方案的技術決策。

---

## **第二節：基礎建設：資料集準備與增強**

本節將提供一套嚴謹、細緻的步驟，指導如何建構一個既符合Ultralytics框架規範，又針對本任務特殊挑戰進行了最佳化的資料集。一個高品質的資料集是模型成功的基石。

### **2.1 標註策略：「單點物件」格式**

為了讓模型能夠學習，我們需要遵循YOLO的資料格式規範。這意味著每張影像都需要一個對應的.txt標註檔案。在該檔案中，每一行代表影像中的一個被偵測實例，也就是一個線條交點。  
每一行的具體格式如下：  
\<class\_id\> \<x\_center\> \<y\_center\> \<width\> \<height\> \<kpt\_x\> \<kpt\_y\> \<kpt\_visibility\>  
**邊界框的「技巧性處理」**：儘管我們的目標是避免使用邊界框，但YOLO的資料載入器（data loader）強制要求提供邊界框的四個參數。為了解決這個矛盾，我們將為每個關鍵點創建一個*無資訊性*的、*最小化*的佔位符邊界框：

* 邊界框的中心點座標\<x\_center\>和\<y\_center\>將被設定為與關鍵點的座標\<kpt\_x\>和\<kpt\_y\>完全相同。  
* 邊界框的寬度\<width\>和高度\<height\>將被設定為一個非常小的、固定的正規化值。例如，0.01。在一個640x640像素的影像中，這大約對應於一個6x6像素的區域。這個微小的方框僅僅是為了滿足格式要求，其本身不包含任何有意義的語意資訊。

**關鍵點座標**：

* \<kpt\_x\>和\<kpt\_y\>是交點精確中心位置的正規化座標（即座標值除以影像的寬或高）。  
* \<kpt\_visibility\>是可見性標記。根據COCO資料集的標準，我們應將其統一設定為2，代表「已標註且可見」，因為我們所有的目標交點在本質上都是可見的。

### **2.2 表格：羽球場交點的標註格式規範**

為了確保標註過程的準確無誤，下表提供了一個清晰、明確的參考標準。遵循此規範可以有效避免因格式錯誤導致的訓練失敗。

| 欄位 | 變數名稱 | 描述 | 範例值 |
| :---- | :---- | :---- | :---- |
| 1 | class\_id | 交點類型的整數索引 (0: T字, 1: 十字, 2: L角)。 | 0 |
| 2 | x\_center | 佔位符邊界框的正規化x座標。**設定為與kpt\_x相同。** | 0.453 |
| 3 | y\_center | 佔位符邊界框的正規化y座標。**設定為與kpt\_y相同。** | 0.671 |
| 4 | width | 佔位符邊界框的正規化寬度。**使用一個固定的微小值。** | 0.01 |
| 5 | height | 佔位符邊界框的正規化高度。**使用一個固定的微小值。** | 0.01 |
| 6 | kpt\_x | 關鍵點的正規化x座標。 | 0.453 |
| 7 | kpt\_y | 關鍵點的正規化y座標。 | 0.671 |
| 8 | kpt\_visibility | 可見性標記。**恆定為2。** | 2 |

### **2.3 資料集設定檔 (badminton\_kpts.yaml)**

此YAML檔案是Ultralytics訓練框架的「指揮中心」，它告訴訓練器在哪裡找到資料、資料包含哪些類別，以及關鍵點的結構是什麼。

* **路徑指定**：path, train, val, test 條目需要指向包含影像檔案的對應資料夾。  
* **類別定義**：nc: 3 代表我們有三個目標類別。names: 為這些類別提供可讀的名稱。  
* **關鍵點定義**：這是最關鍵的修改之一。我們需要將kpt\_shape參數設定為。第一個維度\`1\`表示每個「物件」（即交點）只包含一個關鍵點。第二個維度\`3\`表示每個關鍵點由三個數值描述：\`x\`, \`y\`, 和\`visibility\`。這與標準人體姿態估計中常見的（17個關節點）有著本質的區別。

以下是一個完整的badminton\_kpts.yaml檔案範例：

YAML

\# 資料集根目錄路徑  
path:../datasets/badminton\_court  
\# 訓練集、驗證集、測試集的相對路徑  
train: images/train  
val: images/val  
test: images/test \# 可選

\# 類別定義  
nc: 3  \# number of classes  
names:

\# 關鍵點定義  
kpt\_shape:   \# number of keypoints, number of dims (x, y, visible)

### **2.4 使用 albumentations 進行進階資料增強**

雖然YOLOv8內建了基本的資料增強功能，但對於本任務而言，模型的穩健性（robustness）——特別是對於不同的攝影機角度、透視變換和光照條件——至關重要。albumentations是一個功能極其強大的影像增強函式庫，提供了比預設更豐富的轉換選項，且Ultralytics框架原生支援其整合。  
實作方式：  
我們可以在啟動訓練時，透過修改訓練腳本或利用Ultralytics提供的掛鉤（hooks），傳入一個客製化的albumentations增強管線（pipeline）。  
**建議的增強管線**：

* **幾何變換**：  
  * Perspective：模擬不同攝影機視角造成的透視失真。  
  * ShiftScaleRotate：隨機平移、縮放和旋轉影像。  
  * HorizontalFlip：水平翻轉。  
    這些幾何變換對於模擬本任務中最主要的變異來源——攝影機位置與角度的變化——至關重要。  
* **光度變換（像素級）**：  
  * RandomBrightnessContrast：隨機調整亮度與對比度。  
  * ColorJitter：隨機調整色相、飽和度和明度。  
  * GaussNoise：添加高斯噪點。  
    這些變換能有效提升模型對不同光照條件、陰影及攝影機感光元件雜訊的適應能力。  
* 關鍵點參數設定：  
  在定義增強管線時，必須傳入keypoint\_params參數，以確保關鍵點座標能夠隨著影像的幾何變換而正確更新。設定應為keypoint\_params=A.KeypointParams(format='xy', remove\_invisible=False)。format='xy'告知函式庫我們的關鍵點座標格式。remove\_invisible=False是一個重要的設定，因為某些強烈的幾何變換（如大幅度裁切或旋轉）可能會將關鍵點暫時移出影像邊界，但我們仍希望保留這些標註，因為它們在原始影像中是有效的。

---

## **第三節：架構修改：設計特製化的預測頭**

本節將深入探討如何修改模型的核心架構，將通用的YOLOv8-Pose預測頭，替換為一個專為本任務設計的、更精簡、更高效的特製化預測頭。

### **3.1 解析 yolov8-pose.yaml 中的預設 Pose 預測頭**

標準的YOLOv8-Pose模型，其預測頭部是在ultralytics/nn/modules/head.py中定義的Detect模組。這是一個為多任務學習而設計的複雜模組。其輸出張量（tensor）的結構相當複雜。對於來自頸部網路的每一個尺度的特徵圖（feature map），它都會產生一個形狀類似於\[batch\_size, 4 \* reg\_max (邊界框) \+ 1 (物件信賴度) \+ num\_classes (物件類別) \+ num\_keypoints \* 3 (kpt\_x, kpt\_y, kpt\_vis), num\_anchors\]的輸出。  
對於我們的任務而言，這樣的設計顯然是冗餘且低效的。我們不需要複雜的邊界框回歸（特別是YOLOv8中的Distribution Focal Loss所使用的4 \* reg\_max部分），而且我們的「物件類別」與「關鍵點類別」在本質上是合一的。

### **3.2 提出新的 IntersectionKeypointHead**

核心概念：  
我們將設計一個全新的預測頭模組，其輸出直接對應我們需要的資訊，從而簡化模型架構與後續的損失計算。這個新的預測頭將包含三個平行的卷積分支，每個分支都接收來自頸部網路的特徵圖作為輸入。  
**輸出分支設計**：

1. **物件性/存在性分支 (Objectness/Presence Branch)**：這是一個單通道的卷積層，用於預測在特徵圖的每一個網格單元（grid cell）中，是否存在一個關鍵點。其輸出張量形狀為\`\`。  
2. **分類分支 (Classification Branch)**：這是一個擁有nc（在我們的案例中為3）個輸出通道的卷積層。它負責預測若該位置存在關鍵點，其類別為何（T字、十字或L角）。其輸出張量形狀為\`\`。  
3. **座標回歸分支 (Coordinate Regression Branch)**：這是一個擁有nk \* 2（在我們的案例中為1 \* 2 \= 2）個輸出通道的卷積層。它負責預測該單一關鍵點相對於其所在網格單元左上角的 (dx, dy)偏移量。其輸出張量形狀為\`\`。

優勢：  
這種設計更為高效，且其輸出結構能直接對應到一個更簡潔、更直觀的多任務損失函數（詳見第四節）。它徹底移除了與錨點（anchor）和邊界框回歸相關的非必要複雜性。

### **3.3 實作步驟**

**步驟一：建立客製化預測頭模組 (custom\_head.py)**

1. 在Ultralytics專案目錄下，例如在ultralytics/nn/modules/路徑中，建立一個新的Python檔案，命名為custom\_head.py。  
2. 在該檔案中，定義一個繼承自torch.nn.Module的新類別，命名為IntersectionKeypointHead。  
3. 在其\_\_init\_\_建構函式中，定義上述的三個卷積層（例如，self.obj\_conv, self.cls\_conv, self.kpt\_conv）。這些可以是簡單的nn.Conv2d層。  
4. 在其forward方法中，接收來自頸部網路的特徵圖列表（通常是3個不同尺度的特徵圖）。對每一個特徵圖應用這三個卷積層，並將結果以結構化的方式（例如，一個元組或字典）返回。

**步驟二：建立客製化模型YAML設定檔 (yolov8-badminton.yaml)**

1. 複製一份現有的模型設定檔，例如yolov8n-pose.yaml，並將其重新命名為yolov8-badminton.yaml。  
2. 在檔案的head:區塊中，將原有的Detect模組替換為我們新定義的模組。  
3. 為了讓YOLOv8的parse\_model函式能夠識別並建立我們的新模組，可能需要對ultralytics/nn/tasks.py中的模型解析邏輯進行小幅修改，將IntersectionKeypointHead註冊為一個可用的模組類型。

**YAML設定檔片段 (概念性)**：

YAML

\#... (從 yolov8n-pose.yaml 複製的 backbone 和 neck 部分)...

\# Custom Head  
head:  
  \# from, number, module, args  
  \- \[-1, 1, IntersectionKeypointHead, \[nc, 1\]\] \# nc 將在訓練時被替換為3, 1代表每個物件只有1個關鍵點

架構純粹性與實用主義的權衡：  
雖然從工程角度看，建立一個全新的、乾淨的預測頭模組是「最純粹」的解決方案，但它需要對Ultralytics的原始碼進行更深度的修改（例如，修改tasks.py中的模型建構函式），這不僅增加了實作的複雜度，也可能在未來Ultralytics函式庫更新時產生相容性問題。  
一個更具實用主義、雖然略顯「取巧」但能更快達成目標的方法是，*重新詮釋並利用*現有的Detect預測頭。我們可以這樣設定：

1. 將nc（物件偵測的類別數）設定為1，代表一個通用的「交點」類別。  
2. 將kpt\_shape中的關鍵點數量設定為3，分別對應我們的三種交點類型（T字、十字、L角）。

在這種實用主義的設定下，模型會學習偵測一個通用的「交點」物件（及其無意義的佔位符邊界框），而真正的分類資訊則隱含在關鍵點的預測中。我們可以透過檢查三個關鍵點預測中哪一個的可見性/信賴度分數最高，來判斷該交點的最終類別。  
**結論**：使用者面臨一個策略性的選擇。本報告將詳細闡述這兩種路徑：一是架構純粹的方案（工作量較大，但長期維護性更佳），二是實用主義的方案（工作量較小，能更快地驗證原型）。這讓使用者可以根據自身的專案時程、技術能力與最終目標，做出最合適的決策。  
---

## **第四節：損失函數設計：兩種進階實作策略**

這是本報告最核心的技術章節，將提供兩種完整、包含程式碼思路的解決方案，以應對前述的核心技術挑戰——設計一個不依賴邊界框面積的損失函數。

### **4.1 策略A：透過客製化 MultiTaskKeypointLoss 進行直接座標回歸**

核心概念：  
此方法是對現有YOLOv8損失計算範式的直接擴充與適應。它建構一個多任務損失函數，該函數整合了對關鍵點存在性、分類準確性以及座標精確度的懲罰。由於它不要求改變資料載入器的輸出格式，因此整合起來相對簡單。此策略可視為對標準v8KeypointLoss的直接演進。  
損失函數組成：  
總損失由三個加權部分組成：Ltotal​=wobj​⋅Lobj​+wcls​⋅Lcls​+wkpt​⋅Lkpt​。其中，wobj​, wcls​, wkpt​ 是可調整的權重超參數。

1. **Lobj​：物件性/存在性損失 (Objectness/Presence Loss)**  
   * **目的**：判斷在特徵圖的某個網格單元中，是否存在一個關鍵點。  
   * **函式**：通常使用二元交叉熵損失 torch.nn.BCEWithLogitsLoss。  
   * **對象**：作用於我們在第三節設計的「物件性/存在性分支」的輸出上。  
2. **Lcls​：分類損失 (Classification Loss)**  
   * **目的**：如果某個位置確定存在關鍵點，此損失用於判斷其類別（T、十字、L）是否正確。  
   * **函式**：使用標準的多類別交叉熵損失 torch.nn.CrossEntropyLoss。  
   * **對象**：作用於「分類分支」的輸出，但僅針對存在物件的（positive）網格單元進行計算。  
3. **Lkpt​：關鍵點回歸損失 (Keypoint Regression Loss)**  
   * **目的**：懲罰預測的關鍵點座標與真實座標之間的誤差。  
   * **函式**：可選用 SmoothL1Loss 或 L2Loss (均方誤差)。SmoothL1Loss 對於離群值（outliers）較不敏感，通常是更穩健的選擇。  
   * **關鍵修改**：此損失的計算**不應**再使用邊界框面積進行正規化。我們可以選擇不進行正規化，或使用一個固定的常數因子，或根據影像尺寸進行正規化。這直接解決了第一節中提出的核心問題。

**實作步驟**：

1. 在ultralytics/utils/loss.py檔案中，建立一個新的Python類別，例如MultiTaskKeypointLoss。  
2. 在其forward方法中，它將接收來自模型的預測（preds）和來自資料載入器的真實標籤（targets）。  
3. 需要實作一個目標分配（target assignment）邏輯，將預測結果與真實標籤進行匹配。對於這種單點偵測，可以採用簡單的基於位置的分配策略（例如，落入同一個網格單元的預測與標籤視為一對）。  
4. 對所有匹配上的正樣本對，分別計算上述三個損失分量。  
5. 將三個損失分量根據其權重 wobj​, wcls​, wkpt​ 加總，返回最終的總損失。

**程式碼片段 (概念性 forward 方法)**：

Python

import torch  
import torch.nn as nn

class MultiTaskKeypointLoss(nn.Module):  
    def \_\_init\_\_(self, w\_obj=1.0, w\_cls=1.0, w\_kpt=5.0):  
        super().\_\_init\_\_()  
        self.w\_obj \= w\_obj  
        self.w\_cls \= w\_cls  
        self.w\_kpt \= w\_kpt  
          
        self.bce\_obj \= nn.BCEWithLogitsLoss()  
        self.ce\_cls \= nn.CrossEntropyLoss()  
        self.smooth\_l1 \= nn.SmoothL1Loss(reduction='mean')

    def preprocess\_targets(self, targets, preds\_shape):  
        \# 此處需要實作邏輯，將YOLO格式的targets轉換為  
        \# 與模型輸出對應的網格化gt\_obj\_mask, gt\_cls, gt\_kpts  
        \#...  
        pass

    def forward(self, preds, targets):  
        \# 解包來自新預測頭的預測結果  
        pred\_obj, pred\_cls, pred\_kpts \= preds   
          
        \# 預處理真實標籤，使其與預測的形狀對齊  
        gt\_obj\_mask, gt\_cls, gt\_kpts \= self.preprocess\_targets(targets, pred\_obj.shape)

        \# 計算物件性損失  
        loss\_obj \= self.bce\_obj(pred\_obj, gt\_obj\_mask)

        \# 僅在存在物件的網格上計算分類和回歸損失  
        pos\_mask \= gt\_obj\_mask \> 0  
        if pos\_mask.sum() \> 0:  
            loss\_cls \= self.ce\_cls(pred\_cls\[pos\_mask\], gt\_cls\[pos\_mask\])  
            loss\_kpt \= self.smooth\_l1(pred\_kpts\[pos\_mask\], gt\_kpts\[pos\_mask\])  
        else:  
            loss\_cls \= torch.tensor(0.0).to(preds.device)  
            loss\_kpt \= torch.tensor(0.0).to(preds.device)

        \# 加權求和  
        total\_loss \= self.w\_obj \* loss\_obj \+ self.w\_cls \* loss\_cls \+ self.w\_kpt \* loss\_kpt  
        return total\_loss, torch.stack((loss\_obj, loss\_cls, loss\_kpt)).detach()

整合：  
需要修改訓練流程的程式碼（例如ultralytics/engine/trainer.py或針對pose任務的訓練器），在初始化損失函數的地方，替換為我們自訂的MultiTaskKeypointLoss類別。

### **4.2 策略B：受CenterNet啟發的熱圖與偏移量回歸**

核心概念：  
這是一種更先進但實作更複雜的策略，它將關鍵點偵測問題重新定義為一個類似於密集分割（dense segmentation）的任務。此方法通常能為小型目標帶來更優越的定位精度。模型不再直接回歸座標，而是預測：

1. **熱圖 (Heatmap)**：為每一個關鍵點類別（T、十字、L）預測一張熱圖。熱圖上的峰值（peak）位置即代表了該類別關鍵點的所在位置。  
2. **局部偏移圖 (Local Offset Map)**：由於骨幹網路的降採樣（downsampling）會帶來量化誤差（quantization error），此偏移圖用於對熱圖峰值位置進行亞像素級的精確校正。

損失函數組成：  
總損失由兩個加權部分組成：Ltotal​=wh​⋅Lheatmap​+woff​⋅Loffset​。

1. **Lheatmap​：熱圖損失 (Heatmap Loss)**  
   * **目的**：使模型預測的熱圖盡可能接近真實的熱圖。  
   * **函式**：必須使用**焦點損失 (Focal Loss)** 的變體。這是因為熱圖中存在極端的類別不平衡問題：只有極少數的像素是正樣本（位於關鍵點中心），而絕大多數像素都是負樣本（背景）。Focal Loss能夠動態地降低大量簡單負樣本的權重，讓模型專注於學習困難的樣本。  
   * **對象**：作用於模型輸出的三張熱圖上。  
2. **Loffset​：偏移量回歸損失 (Offset Regression Loss)**  
   * **目的**：學習亞像素級的定位校正值。  
   * **函式**：使用簡單的 L1Loss。  
   * **對象**：此損失**僅**在真實關鍵點所在的稀疏位置上進行計算，而非在整個特徵圖上計算。

**實作步驟**：

1. **修改資料載入器**：這是此策略最主要的挑戰。需要修改資料集類別的\_\_getitem\_\_方法（位於ultralytics/data/dataset.py）。除了返回影像外，還必須在運行時動態生成真實的目標熱圖。這通常涉及以下步驟：  
   * 創建一個\`\`的零張量，其中nc是類別數，H, W是輸入影像尺寸，R是網路的總降採樣率（output stride）。  
   * 對於每一個真實關鍵點，在其對應的降採樣後座標上，繪製一個二維高斯核（Gaussian kernel），形成一個以關鍵點為中心、向外平滑衰減的「山峰」。  
2. **修改模型預測頭**：模型的預測頭需要被重新設計，使其輸出nc個通道的熱圖和2個通道的偏移圖。分類資訊現在隱含在「哪一張熱圖出現了峰值」之中。  
3. **實作損失函數**：在loss.py中建立一個新的CenterNetLoss類別。其forward方法將在整個熱圖上計算Focal Loss，並在稀疏的真實關鍵點位置上計算L1 Offset Loss。

**程式碼片段 (概念性高斯熱圖生成)**：

Python

import numpy as np

def generate\_heatmap(center\_kpt, output\_res\_w, output\_res\_h, sigma):  
    """  
    在給定解析度的熱圖上，於center\_kpt位置生成一個2D高斯核。  
    center\_kpt: (x, y) 座標  
    output\_res\_w, output\_res\_h: 熱圖的寬和高  
    sigma: 高斯核的標準差  
    """  
    heatmap \= np.zeros((output\_res\_h, output\_res\_w), dtype=np.float32)  
      
    \# 計算高斯核的半徑，通常取3倍sigma  
    tmp\_size \= sigma \* 3  
      
    \# 獲取高斯核的上下左右邊界  
    mu\_x, mu\_y \= int(center\_kpt), int(center\_kpt)  
      
    ul \= \[int(mu\_x \- tmp\_size), int(mu\_y \- tmp\_size)\]  
    br \= \[int(mu\_x \+ tmp\_size \+ 1), int(mu\_y \+ tmp\_size \+ 1)\]

    \# 處理邊界情況  
    if ul \>= output\_res\_w or ul \>= output\_res\_h or br \< 0 or br \< 0:  
        return heatmap

    \# 生成高斯核網格  
    size \= 2 \* tmp\_size \+ 1  
    x \= np.arange(0, size, 1, np.float32)  
    y \= x\[:, np.newaxis\]  
    x0 \= y0 \= size // 2  
      
    \# 高斯函式  
    g \= np.exp(-((x \- x0) \*\* 2 \+ (y \- y0) \*\* 2\) / (2 \* sigma \*\* 2))

    \# 將高斯核安全地放置到熱圖上  
    g\_x \= max(0, \-ul), min(br, output\_res\_w) \- ul  
    g\_y \= max(0, \-ul), min(br, output\_res\_h) \- ul  
    img\_x \= max(0, ul), min(br, output\_res\_w)  
    img\_y \= max(0, ul), min(br, output\_res\_h)

    heatmap\[img\_y:img\_y, img\_x:img\_x\] \= np.maximum(  
        heatmap\[img\_y:img\_y, img\_x:img\_x\],  
        g\[g\_y:g\_y, g\_x:g\_x\])  
          
    return heatmap

### **4.3 表格：損失函數策略比較分析**

為了幫助使用者在兩種複雜的策略之間做出明智的選擇，下表提供了一個多維度的比較，總結了各自的優劣與權衡。

| 特性 | 策略A：直接回歸 | 策略B：熱圖回歸 |
| :---- | :---- | :---- |
| **核心思想** | 多任務學習，直接回歸類別與座標。 | 密集熱圖預測，結合偏移量精修。 |
| **潛在精度** | 良好，但可能在像素級精確定位上遇到瓶頸。 | 優異，通常是實現高精度定位的SOTA方法。 |
| **實作複雜度** | 中等。需自訂損失類別，對訓練器做少量修改。 | 高。需自訂損失、預測頭，並對資料載入器進行重大修改。 |
| **訓練穩定性** | 相對穩定，但對各損失項的權重超參數敏感。 | 可能對Focal Loss的超參數（alpha, beta）敏感。 |
| **推論速度** | 非常快。後處理簡單。 | 快速。後處理需在熱圖上尋找峰值（如用MaxPool或NMS）。 |
| **最佳適用場景** | 快速原型開發、對定位精度要求「足夠好」的應用。 | 追求極致定位精度的應用、學術研究。 |

---

## **第五節：訓練、評估與推論**

本節將前述的資料準備與模型修改工作付諸實踐，詳細說明如何啟動訓練、如何客觀地衡量模型性能，以及如何使用最終訓練好的模型進行預測。

### **5.1 執行訓練管線**

使用Ultralytics框架，可以透過命令列（CLI）或Python腳本來啟動訓練。關鍵在於要指定我們客製化的模型設定檔。  
**命令列範例**：

Bash

yolo pose train data=path/to/badminton\_kpts.yaml model=path/to/yolov8-badminton.yaml epochs=100 imgsz=640 device=0

重要提示：  
由於我們對Ultralytics的原始碼進行了修改（例如，添加了新的模組類別或損失函數），訓練必須在我們修改過的本地ultralytics儲存庫克隆版本中啟動，而不能使用pip安裝的官方發行版。使用者需要從本地克隆的根目錄執行上述命令或相關的Python腳本。

### **5.2 一個更有意義的評估指標：正確關鍵點百分比 (PCK)**

OKS指標的缺陷：  
如第一節所述，物件關鍵點相似度（OKS）指標對於本任務是不適用的。其公式嚴重依賴物件尺度s（由邊界框或分割區域面積計算得出）和每個關鍵點類型的標準差sigma。在我們的場景中，s是無意義的，而為三種交點類型手動定義合理的sigma值本身就是一項複雜且缺乏理論依據的工作，需要大量的重複標註與統計分析。  
採用PCK指標：  
因此，我們強烈建議採用正確關鍵點百分比（Percentage of Correct Keypoints, PCK） 作為主要的評估指標。PCK的定義非常直觀：如果一個預測關鍵點與其對應的真實關鍵點之間的歐幾里得距離（Euclidean distance）小於一個預設的像素閾值，則該預測被視為正確。  
定義PCK閾值：  
我們可以定義多個不同級別的PCK指標，以獲得對模型性能更全面的了解。例如：

* PCK@5px：一個嚴格的指標，要求定位誤差在5個像素以內。  
* PCK@10px：一個較為寬鬆的指標，允許10個像素的誤差。  
  這比單一的OKS分數更能直觀地反映模型在應用中的實際表現。

實作：  
為了計算PCK，需要對Ultralytics的驗證迴圈（validation loop）進行修改，這通常位於ultralytics/models/v8/pose/val.py或ultralytics/engine/validator.py。在該腳本中，需要添加計算預測點與真實點之間像素距離的邏輯，並根據預設的閾值統計正確率，最後將結果添加到評估指標字典中。

### **5.3 表格：PCK評估指標定義**

下表為PCK指標提供了正式的定義，確保使用者在評估模型時有清晰、統一的標準。

| 指標 | 公式 | 閾值 (d\_thresh) | 解釋 |
| :---- | :---- | :---- | :---- |
| **PCK@T** | PCK=N1​∑i=1N​I(d(predi​,gti​)≤dthresh​) | T 像素 | 被成功定位在真實位置 T 像素半徑內的關鍵點所佔的百分比。 |
| **PCK@5px** | (同上) | 5 像素 | **高精度指標**：適用於需要極高定位準確度的應用，如自動線審系統。 |
| **PCK@10px** | (同上) | 10 像素 | **中精度指標**：一個更具容錯性的指標，適用於一般的場地校準或戰術分析。 |

*註：在公式中，N 是總的真實關鍵點數量，d(predi​,gti​) 是第 i 個預測點與真實點之間的歐幾里得距離，I(⋅) 是指示函數（條件成立時為1，否則為0）。*

### **5.4 推論與解析 Results 物件**

訓練完成後，使用best.pt權重進行推論的過程非常直接。  
**Python推論範例**：

Python

from ultralytics import YOLO

\# 載入訓練好的客製化模型  
model \= YOLO('path/to/runs/pose/train/weights/best.pt')

\# 對單張影像或影像列表進行預測  
results \= model('path/to/your/badminton\_court\_image.jpg')

\# 解析結果  
final\_detections \=  
for result in results:  
    if result.keypoints is not None:  
        \# result.keypoints.xy 是一個 torch.Tensor, 形狀為 \[num\_instances, num\_keypoints, 2\]  
        \# 在我們的案例中，num\_keypoints 是 1  
        keypoints\_xy \= result.keypoints.xy.cpu().numpy()  
          
        \# result.boxes.cls 是一個 torch.Tensor, 形狀為 \[num\_instances\]  
        \# 包含了每個實例的類別索引  
        class\_ids \= result.boxes.cls.cpu().numpy()

        for i in range(len(class\_ids)):  
            class\_id \= int(class\_ids\[i\])  
            class\_name \= model.names\[class\_id\]  
              
            \# 獲取單一關鍵點的座標  
            x, y \= keypoints\_xy\[i, 0, :\]  
              
            final\_detections.append((class\_name, x, y))

print(final\_detections)  
\# 預期輸出:

這段程式碼展示了如何從Results物件 中提取關鍵點座標 (result.keypoints.xy) 和對應的類別 (result.boxes.cls)，並將它們組合成使用者最終需要的 \[(class\_name, x, y),...\] 格式。  
---

## **第六節：綜合、建議與未來工作**

本節對前述所有技術細節進行了高層次的戰略性總結，並為使用者指明了後續的優化方向。

### **6.1 戰略性建議**

分階段實施：  
強烈建議從策略A（直接回歸） 開始著手。該策略是對現有框架的直接擴展，所需的程式碼修改侵入性較小，能夠更快地建立一個性能穩健的基線模型（baseline）。這有助於快速驗證整個流程，包括資料標註和模型訓練是否正確。  
何時採用策略B：  
只有當策略A的定位精度無法滿足最終應用的需求時，才建議投入資源實施策略B（熱圖回歸）。策略B代表了一條高風險、高回報的技術路徑，它雖然有潛力達到更高的精度，但也需要更大量的工程開發與調試工作。  
超參數調校的重要性：  
無論選擇哪種策略，都必須認識到損失權重的調校至關重要。對於策略A，需要仔細調整w\_obj, w\_cls, w\_kpt之間的平衡；對於策略B，則需調整w\_h和w\_off的比例。合理的權重設定是模型收斂至最佳性能的關鍵。

### **6.2 未來的改進方向**

模型架構選擇：  
建議針對不同的模型尺寸（如yolov8n-pose, yolov8s-pose等）進行實驗，以在速度和精度之間找到最佳平衡點。考慮到羽球場線條的幾何特徵相對簡單，一個較小的yolov8n骨幹網路可能已經足夠，並且能提供更快的推論速度。  
引入注意力機制：  
近期的研究表明，在骨幹網路或頸部網路中整合注意力機制（Attention Mechanisms），如CBAM或SimDLKA，可以幫助模型更有效地聚焦於稀疏、微小的目標特徵上。對於本任務中細小的線條交點，引入注意力機制可能帶來顯著的性能提升。  
量化與部署：  
當訓練出一個令人滿意的模型後，可以利用Ultralytics框架內建的匯出功能，將模型轉換為ONNX、TensorRT等高效能的推論格式。這一步是將模型部署到實際應用（無論是邊緣設備還是伺服器）中的關鍵環節，能夠大幅提升模型的運行速度。

### **6.3 結論**

總而言之，透過將羽球場的線條交點重新定義為「無實體的關鍵點」，並據此對YOLOv8的模型預測頭與損失函數進行精心設計與修改，完全可以成功地將這個強大的框架應用於此一新穎且專業化的偵測任務。本報告提供的詳盡計畫，為使用者規劃了兩條清晰且可行的技術路徑，使其能夠根據自身的性能需求與工程資源，選擇最合適的方案。遵循此指南，使用者將能高效地建構、訓練並部署一個能夠精確偵測羽球場線條交點的客製化模型。