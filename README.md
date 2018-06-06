# pyhsafm

このモジュールはHS-AFMで撮影された画像をPython3上で処理するためのモジュールです。
OpenCV v3用の画像を出力することができ、必要に応じてOpenCV v3との連携が可能です。  
使用する際は、afmimprocをインポートしてください。
***

## 使い方
### 目次
#### [依存関係](#依存関係)  
#### [インポート方法](#インポート方法)  

#### クラス
- [AfmImg](#AfmImg)
- [ASD_reader](#ASD_reader)

#### 関数
- [imwrite](#imwrite)
- [imwrite_gray](#imwrite_gray)
- [imwrite](#imwrite)
- [imwrite_gray](#imwrite_gray)
- [imshow](#imshow)
- [imshow_gray](#imshow_gray)
- [implay](#implay)
- [histogram](#histogram)
- [threshold_otsu](#threshold_otsu)
- [binarize](#binarize)
- [apply_mask](#apply_mask)
- [heightCorrection](#heightCorrection)
- [tiltCorrection](#tiltCorrection)
- [heightScaling](#heightScaling)
- [highpass_filter](#highpass_filter)
- [lowpass_filter](#lowpass_filter)
- [bandpass_filter](#bandpass_filter)
- [find_edge](#find_edge)
- [enhance_edge](#enhance_edge)
- [median_filter](#median_filter)
- [convolution_filter](#convolution_filter)
- [average_filter](#average_filter)
- [gaussian_filter](#gaussian_filter)
- [laplacian_filter](#laplacian_filter)
- [sharpen_filter](#sharpen_filter)

### OpenCVの機能拡張
- [movieWriter](#movieWriter)
- [writeTime](#writeTime)

***

## 依存関係
このモジュールは numpy, scipy, opencv-python, numba, sklearn に依存します。
以下のコマンドでインストールを行ってください。
```
$ pip3 install numpy scipy opencv-python numba sklearn
```

## インポート方法
```
import sys
sys.path.append('pyhsafmをgit cloneしたディレクトリ')
import afmimproc as aip
```

## クラス
### AfmImg

` AfmImg(data, XYlength, idx = None, frame_header = None) `

HS-AFM像を扱うためのクラスです。
HS-AFM像が持つXY方向の距離などの情報を扱うことができます。  
直接データを渡してインスタンスを作成することも可能ですが、後述するASD_readerを利用することを推奨します。

引数  
*data* : 高さ情報の2次元リスト  
*XYlength* :　XY方向の長さ(nm)のリスト [x, y]  
*idx* :　画像のインデックス（オプション）  
*frame_header* : 画像のメタ情報（オプション）


#### メソッド
`AfmImg.copy()`  
自身の深いコピーを生成します。

`AfmImg.getOpenCVimageGray()`   
OpenCV互換の8bitグレースケール画像を出力します。

`AfmImg.getOpenCVimage()`   
OpenCV互換の8bitカラー画像を出力します。

`AfmImg[y0:y1,x0:x1]`   
画像をy0からy1, x0からx1の範囲で切り抜きます。  
*元の画像の一部への参照ではなくコピーを返す*
ことに注意してください。  
また、この方法で生成された画像はメンバ変数`idx`および`frame_header`を持っていません。

#### メンバ変数
`AfmImg.data` : 二次元ndarray（numpyリスト）  
高さ情報の生データへのアクセスができます。

`AfmImg.shape` : 要素数2のリスト  
OpenCVと同じフォーマットで画像のサイズを持っています。

`AfmImg.zdata` : 要素数2のリスト  
高さの最小値、最大値を持っています。

`AfmImg.XYlength` : 要素数2のリスト  
XY方向の実際の長さ（測定時のスキャンサイズ）を持っています。

`AfmImg.lenppixel` : 数値  
1ピクセルあたりの実際の長さです。画像上でのピクセル数と積を取ることで実際の長さを求めることができます。

`AfmImg.ns2ppixel` : 数値  
1ピクセルあたりの実際の面積です。画像上で算出されるピクセルでの面積と積を取ることで実際の面積を求めることができます。  
応用として、ns2ppixelと高さの積を取ることでそのピクセルの体積を計算することができます。この計算を他のピクセルに対しても適用し、総和を取ることで領域の体積を求めることができます。

`AfmImg.idx` : 整数  
ASDファイル内でのフレーム番号を持っています。

`AfmImg.frame_header` : 辞書型リスト  
測定時に付加されたヘッダーファイルの生データにアクセスできます。  
次に示すキーを持っています。  
```
['CurrentNum', 'MaxData', 'MiniData', 'XOffset', 'YOffset', 'XTilt', 'YTilt', 'LaserFlag', 'Reserved', 'Reserved', 'Reserved', 'Reserved']
```

### ASD_reader
`ASD_reader(path)`

HS-AFMのASDファイルを読み込むためのクラスです。

引数  
*path* : 読み込むASDファイルのパス

#### メソッド
`ASD_reader[idx]` or `ASD_reader[start:stop:step]`  

ファイル読み書きに使用するopenと同様にwith文に対応しています。
読み込まれた一連の画像は、インデックスを使用してアクセスすることが可能です。*アクセスするごとに新たなAfmImgインスタンスが生成される* ことに注意してください。

#### メンバ変数
`ASD_reader.frame_time` : 数値  
フレームあたりの時間を持っています。

`ASD_reader.comment` : 文字列  
測定時に付加されたコメントを持っています。

`ASD_reader.date` : datetimeオブジェクト  
測定の日時をdatetime型で持っています。

`ASD_reader.header` : 辞書型リスト  
測定時に付加されたヘッダーファイルの生データにアクセスできます。  
次に示すキーを持っています。  
```
['FileType','FileHeaderSizeForSave', 'FrameHeaderSize', 'TextEncoding', 'OpeNameSize', 'CommentSizeForSave', 'DataType1ch', 'DataType2ch', 'FrameNum', 'ImageNum', 'ScanDirection', 'ScanTryNum', 'XPixel', 'YPixel', 'XScanSize', 'YScanSize', 'AveFlag', 'AverageNum', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'XRound', 'YRound', 'FrameTime', 'Sensitivity', 'PhaseSens', 'Offset1', 'Offset2', 'Offset3', 'Offset4', 'MachineNo', 'ADRange', 'ADResolution', 'MaxScanSizeX',　'MaxScanSizeY', 'PiezoConstX', 'PiezoConstY', 'PiezoConstZ', 'DriverGainZ', 'Comment']
```

**使用例**  
```
import afmimproc as aip

with aip.ASD_reader('test.asd') as imgs:  #with文でASDファイルをimgsとして開く
  #測定日を表示
  print(imgs.date)

  #コメントを表示
  print(imgs.comment)

  #1フレーム目を読み込んで表示
  img0 = imgs[0]
  aip.imshow(img0)

  #全フレームを順番に表示
  for img in imgs:
    aip.imshow(img)

  #101フレームから200フレームまで順番に表示
  for img in imgs[100:200]:
    aip.imshow(img)

  #201フレームから300フレームまで2フレームごとに表示
  for img in imgs[200:300:2]:
    aip.imshow(img)
```

***

## 関数
### imwrite
`imwrite(path, img)`

AfmImgをカラーの画像ファイルとして出力する関数です。
拡張子で形式を指定することができます。

引数  
*path* : 出力する画像ファイルのパス  
*img* : 出力するAfmImg形式の画像

戻り値  
なし


### imwrite_gray
`imwrite_gray(path, img)`

AfmImgをグレースケールの画像ファイルとして出力する関数です。
拡張子で形式を指定することができます。

引数  
*path* : 出力する画像ファイルのパス  
*img* : 出力するAfmImg形式の画像

戻り値  
なし


### imshow  
`imshow(img, text='')`

AfmImgをカラーの画像ファイルとして表示する関数です。

引数  
*img* : 表示するAfmImg形式の画像  
*text* : 表示するウィンドウのタイトル（オプション）

戻り値  
なし


### imshow_gray  
`imshow_gray(img, text='')`

AfmImgをグレースケールの画像ファイルとして表示する関数です。

引数  
*img* : 表示するAfmImg形式の画像  
*text* : 表示するウィンドウのタイトル（オプション）

戻り値  
なし


### implay  
`implay(imgs, idx=None, time=True, func=None, args=None)`

ASD_readerの画像を連続で表示する関数です。  
キーボード入力で操作します。  
f: 次の画像(forward)  
b: 前の画像(backward)  
Esc：終了   

引数  
*imgs* : 表示するASD_readerのインスタンス  
*idx* : 表示する範囲 [start, stop]（オプション）  
*time*：時間を表示するかどうか（オプション）  
*func*：画像に対する処理を書いた関数（オプション）  
*args*：funcに渡す引数のリスト（オプション）  

戻り値  
なし  

funcとして画像に対する処理を記述した関数を渡すことができます。  
argsは必ず引数として渡されますが、関数内で使用しなくても構いません。  
funcは以下の条件で作成してください。  
&emsp;&emsp; 引数  
&emsp;&emsp; src : 処理を行うAfmImg形式の画像  
&emsp;&emsp; args：funcに渡す引数のリスト  
&emsp;&emsp;  
&emsp;&emsp; 戻り値  
&emsp;&emsp; dst : OpenCV形式の画像


### histogram  
`histogram(img, range=None, step=0.1, order=None, smoothed=False, smoothing_order=3, mask=None)`

画像のヒストグラムとそのピークリストを生成する関数です。

引数  
*img* : AfmImg形式の画像  
*range* : リスト（オプション）  
&emsp;&emsp; ヒストグラム作成に使用する高さの範囲 [min, max]  
*step* : 数値（オプション）  
&emsp;&emsp; ヒストグラム作成時の高さの幅  
*order* : 整数（オプション）  
&emsp;&emsp; ピーク検出時に比較を行う幅の値  
*smoothed* : bool型（オプション）  
&emsp;&emsp; ヒストグラムを平滑化するかどうか  
*smoothing_order* : 整数（オプション）  
&emsp;&emsp; ヒストグラム平滑化の際に、平均を取る範囲  
*mask* : マスク（オプション）  
&emsp;&emsp; マスクされた部分を除いてヒストグラムを生成する  

戻り値  
*hist* : リスト  
&emsp;&emsp; ヒストグラムのY軸の値  
*hist_bins* : リスト  
&emsp;&emsp; ヒストグラムのラベル（X軸の値）  
*peaks* : リスト  
&emsp;&emsp; ヒストグラム上のピーク。hist_binsのインデックスとして出力されます。  


### threshold_otsu
`threshold_otsu(img, mask=None)`

大津の二値化に用いるしきい値を高さとして出力します。

引数  
*img* : AfmImg形式の画像
*mask* : マスク（オプション）
    マスクされた部分を除いてしきい値を生成する

戻り値  
*threshold* : 数値  
&emsp;&emsp; 大津の二値化に用いるしきい値


### binarize  
`binarize(src, lowest, highest=None)`

二値化した画像を出力します。

引数  
*src* : AfmImg形式の画像  
*lowest* : 数値  
&emsp;&emsp; 二値化のしきい値  
*highest* : 数値（オプション）  
&emsp;&emsp; 最大値の指定。highestが指定されるとそれ以上の高さの部分が0.0nmになります。  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; 二値化（高さ1.0nmまたは0.0nm）された画像  


### apply_mask
`apply_mask(src, mask)`

AfmImg形式の画像にOpenCV形式のマスクを適用する関数です。  

引数   
*src* : AfmImg形式の画像  
*step* : OpenCV形式のマスク  

戻り値  
*dst* : マスクが適用されたAfmImg形式の画像  


### heightCorrection  
`heightCorrection(src, makeItZero=False, peak_num=1, step=0.05)`

高さヒストグラムからマイカ表面を認識し、マイカ表面が0.0nmになるように高さを修正する関数です。

引数  
*src* : AfmImg形式の画像  
*makeItZero* : bool型（オプション）  
&emsp;&emsp; 0.0nm以下をすべて0.0nmで置換するかどうか  
*peak_num* : 整数（オプション）  
&emsp;&emsp; 低い方から数えて何個目のピークをマイカ表面とするか  
*step* : 数値（オプション）  
&emsp;&emsp; ヒストグラム取得の際のパラメーター

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; 高さが修正された画像


### tiltCorrection  
`tiltCorrection(src, th_range=0.25)`

マイカ表面を平面近似し、そのに基づいて傾きを補正する関数です。

引数  
*src* : AfmImg形式の画像  
*th_range* : 数値（オプション）  
&emsp;&emsp; 何nmまでをマイカ表面として扱うかを指定するか

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; 傾きが修正された画像


### heightScaling  
`heightScaling(src, highest)`

highestに指定した以上の高さをすべてhighestに置換する関数です。

引数  
*src* : AfmImg形式の画像  
*highest* : 数値  
&emsp;&emsp; 最大の高さ

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; highest以上の部分がhighestに置換された画像


### highpass_filter  
`highpass_filter(src, size)`

ハイパスフィルター

引数  
*src* : AfmImg形式の画像  
*size* : 整数  
&emsp;&emsp; フィルターのサイズ

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像


### lowpass_filter  
`lowpass_filter(src, size)`

ローパスフィルター

引数  
*src* : AfmImg形式の画像  
*size* : 整数  
&emsp;&emsp; フィルターのサイズ  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### bandpass_filter  
`bandpass_filter(src, size_outer, size_inner)`

ハイパスフィルター

引数  
*src* : AfmImg形式の画像  
*size_outer* : 整数  
&emsp;&emsp; 外側フィルターのサイズ  
*size_inner* : 整数  
&emsp;&emsp; 内側フィルターのサイズ  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### find_edge  
`find_edge(src, inner=True)`

エッジを探すフィルター

引数  
*src* : AfmImg形式の画像  
*inner* : bool型（オプション）  
&emsp;&emsp; エッジを内側に追加するか（Ture）、外側に追加するか（False）  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### enhance_edge  
`enhance_edge(src, inner=True)`

エッジ強調フィルター

引数  
*src* : AfmImg形式の画像  
*inner* : bool型（オプション）  
&emsp;&emsp; エッジを内側に追加するか（Ture）、外側に追加するか（False）  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### median_filter  
`median_filter(src, ksize = 3)`

メディアンフィルター

引数  
*src* : AfmImg形式の画像  
*ksize* : 整数（オプション）  
&emsp;&emsp; フィルターのサイズ  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### convolution_filter  
`convolution_filter(src, kernel)`

畳み込み演算を行う関数

引数  
*src* : AfmImg形式の画像  
*kernel* : 2次元リスト  
&emsp;&emsp; 任意のカーネルを指定できます  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### average_filter  
`average_filter(src, ksize = 3)`

平均値フィルター

引数  
*src* : AfmImg形式の画像  
*ksize* : 整数（オプション）  
&emsp;&emsp; フィルターのサイズ  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### gaussian_filter  
`gaussian_filter(src, ksize = 5, sigma = 0)`

ガウシアンフィルター

引数  
*src* : AfmImg形式の画像  
*ksize* : 整数（オプション）  
&emsp;&emsp; フィルターのサイズ  
*sigma* : 数値（オプション）  
&emsp;&emsp; ガウシアンカーネルの標準偏差。この値を大きくするとぼかしが強くなります。  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### laplacian_filter  
`laplacian_filter(src)`

ラプラシアンフィルター

引数  
*src* : AfmImg形式の画像  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### sharpen_filter  
`sharpen_filter(src)`

鮮鋭化フィルター

引数  
*src* : AfmImg形式の画像  

戻り値  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  

***

## OpenCVの機能拡張
### movieWriter  
`movieWriter(path, frame_time, imgShape)`

cv2.VideoWriterをwith文に対応させ、使いやすくしたラッピングクラスです。

引数  
*path* : 動画の保存先  
*frame_time* : 1フレームあたりの時間、単位は秒  
*imgShape* : 画像のサイズ、shapeをそのまま指定してください  

戻り値  
cv2.VideoWriter : cv2.VideoWriterのインスタンス


### writeTime  
`writeTime(src, time, frame_num = "", font_size = 1.2)`

OpenCVのカラー画像に時間とフレームナンバーを書き込みます。

引数  
*src* : OpenCVのカラー画像  
*time* : 時間、単位はmsec  
*frame_num* : フレームナンバー（オプション）  
*font_size* : 時間表示のフォントサイズ（オプション）

戻り値  
*dst* : 時間とフレームナンバーが書き込まれたOpenCV形式のカラー画像
