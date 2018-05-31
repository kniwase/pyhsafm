# pyhsafm

このモジュールはHS-AFMで撮影された画像をPython上で処理するためのモジュールです。
OpenCV用の画像を出力することができ、必要に応じてOpenCVとの連携が可能です。
***

## クラス
- AfmImg
- ASD_reader

### AfmImg

` AfmImg(data, XYlength, idx = None, frame_header = None) `

HS-AFM像を扱うためのクラスです。
HS-AFM像が持つXY方向の距離などの情報を扱うことができます。  
直接データを渡してインスタンスを作成することも可能ですが、後述するクラスを利用することを推奨します。

**引数**  
*data* : 高さ情報の2次元リスト  
*XYlength* :　XY方向の長さ(nm)のリスト [x, y]  
*idx* :　画像のインデックス（オプション）  
*frame_header* : 画像のメタ情報（オプション）

OpenCVと同じ方法でのインデックスによる画像の切り抜き操作に対応しています。この場合、
*元の画像の一部への参照ではなくコピーを返す*
ことに注意してください。

#### メソッド
`self.copy()`  
自身の深いコピーを生成します。

`self.getOpenCVimageGray() `   
OpenCV互換の8bitグレースケール画像を出力します。

`self.getOpenCVimage() `   
OpenCV互換の8bitカラー画像を出力します。


### ASD_reader
`ASD_reader(path)`

HS-AFMのASDファイルを読み込むためのクラスです。

**引数**  
*path* : 読み込むASDファイルのパス

ファイル読み書きに使用するopenと同様にwith文に対応しています。
読み込まれた一連の画像は、インデックスを使用してアクセスすることが可能です。*アクセスするごとに新たなAfmImgインスタンスが生成されることに注意してください。*



## 関数
- [imwrite](#imwrite)
- [imwrite_gray](#imwrite_gray)
- [imwrite](#imwrite)
- [imwrite_gray](#imwrite_gray)
- [imshow](#imshow)
- [imshow_gray](#imshow_gray)
- [histogram](#histogram)
- [threshold_otsu](#threshold_otsu)
- [binarize](#binarize)
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


### imwrite
`imwrite(path, img)`

AfmImgをカラーの画像ファイルとして出力する関数です。
拡張子で形式を指定することができます。

**引数**  
*path* : 出力する画像ファイルのパス  
*img* : 出力するAfmImg形式の画像

**戻り値**  
なし


### imwrite_gray
`imwrite_gray(path, img)`

AfmImgをグレースケールの画像ファイルとして出力する関数です。
拡張子で形式を指定することができます。

**引数**  
*path* : 出力する画像ファイルのパス  
*img* : 出力するAfmImg形式の画像

**戻り値**  
なし


### imshow  
`imshow(img, text='')`

AfmImgをカラーの画像ファイルとして表示する関数です。

**引数**  
*img* : 表示するAfmImg形式の画像  
*text* : 表示するウィンドウのタイトル（オプション）

**戻り値**  
なし


### imshow_gray  
`imshow_gray(img, text='')`

AfmImgをグレースケールの画像ファイルとして表示する関数です。

**引数**  
*img* : 表示するAfmImg形式の画像  
*text* : 表示するウィンドウのタイトル（オプション）

**戻り値**  
なし


### histogram  
`histogram(img, range=None, step=0.1, order=None, smoothed=False, smoothing_order=3)`

画像のヒストグラムとそのピークリストを生成する関数です。

**引数**  
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

**戻り値**  
*hist* : リスト  
&emsp;&emsp; ヒストグラムのY軸の値  
*hist_bins* : リスト  
&emsp;&emsp; ヒストグラムのラベル（X軸の値）  
*peaks* : リスト  
&emsp;&emsp; ヒストグラム上のピーク。hist_binsのインデックスとして出力されます。  


### threshold_otsu
`threshold_otsu(img)`

大津の二値化に用いるしきい値を高さとして出力します。

**引数**  
*img* : AfmImg形式の画像

**戻り値**  
*threshold* : 数値  
&emsp;&emsp; 大津の二値化に用いるしきい値


### binarize  
`binarize(src, lowest, highest=None)`

二値化した画像を出力します。

**引数**  
*src* : AfmImg形式の画像  
*lowest* : 数値  
&emsp;&emsp; 二値化のしきい値  
*highest* : 数値（オプション）  
&emsp;&emsp; 最大値の指定。highestが指定されるとそれ以上の高さの部分が0.0nmになります。  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; 二値化（高さ1.0nmまたは0.0nm）された画像  


### heightCorrection  
`heightCorrection(src, makeItZero=False, peak_num=1, step=0.05)`

高さヒストグラムからマイカ表面を認識し、マイカ表面が0.0nmになるように高さを修正する関数です。

**引数**  
*src* : AfmImg形式の画像  
*makeItZero* : bool型（オプション）  
&emsp;&emsp; 0.0nm以下をすべて0.0nmで置換するかどうか  
*peak_num* : 整数（オプション）  
&emsp;&emsp; 低い方から数えて何個目のピークをマイカ表面とするか  
*step* : 数値（オプション）  
&emsp;&emsp; ヒストグラム取得の際のパラメーター

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; 高さが修正された画像


### tiltCorrection  
`tiltCorrection(src, th_range=0.25)`

マイカ表面を平面近似し、そのに基づいて傾きを補正する関数です。

**引数**  
*src* : AfmImg形式の画像  
*th_range* : 数値（オプション）  
&emsp;&emsp; 何nmまでをマイカ表面として扱うかを指定するか

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; 傾きが修正された画像


### heightScaling  
`heightScaling(src, highest)`

highestに指定した以上の高さをすべてhighestに置換する関数です。

**引数**  
*src* : AfmImg形式の画像  
*highest* : 数値  
&emsp;&emsp; 最大の高さ

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; highest以上の部分がhighestに置換された画像


### highpass_filter  
`highpass_filter(src, size)`

ハイパスフィルター

**引数**  
*src* : AfmImg形式の画像  
*size* : 整数  
&emsp;&emsp; フィルターのサイズ

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像


### lowpass_filter  
`lowpass_filter(src, size)`

ローパスフィルター

**引数**  
*src* : AfmImg形式の画像  
*size* : 整数  
&emsp;&emsp; フィルターのサイズ  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### bandpass_filter  
`bandpass_filter(src, size_outer, size_inner)`

ハイパスフィルター

**引数**  
*src* : AfmImg形式の画像  
*size_outer* : 整数  
&emsp;&emsp; 外側フィルターのサイズ  
*size_inner* : 整数  
&emsp;&emsp; 内側フィルターのサイズ  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### find_edge  
`find_edge(src, inner=True)`

エッジを探すフィルター

**引数**  
*src* : AfmImg形式の画像  
*inner* : bool型（オプション）  
&emsp;&emsp; エッジを内側に追加するか（Ture）、外側に追加するか（False）  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### enhance_edge  
`enhance_edge(src, inner=True)`

エッジ強調フィルター

**引数**  
*src* : AfmImg形式の画像  
*inner* : bool型（オプション）  
&emsp;&emsp; エッジを内側に追加するか（Ture）、外側に追加するか（False）  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### median_filter  
`median_filter(src, ksize = 3)`

メディアンフィルター

**引数**  
*src* : AfmImg形式の画像  
*ksize* : 整数（オプション）  
&emsp;&emsp; フィルターのサイズ  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### convolution_filter  
`convolution_filter(src, kernel)`

畳み込み演算を行う関数

**引数**  
*src* : AfmImg形式の画像  
*kernel* : 2次元リスト  
&emsp;&emsp; 任意のカーネルを指定できます  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### average_filter  
`average_filter(src, ksize = 3)`

平均値フィルター

**引数**  
*src* : AfmImg形式の画像  
*ksize* : 整数（オプション）  
&emsp;&emsp; フィルターのサイズ  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### gaussian_filter  
`gaussian_filter(src, ksize = 5, sigma = 0)`

ガウシアンフィルター

**引数**  
*src* : AfmImg形式の画像  
*ksize* : 整数（オプション）  
&emsp;&emsp; フィルターのサイズ  
*sigma* : 数値（オプション）  
&emsp;&emsp; ガウシアンカーネルの標準偏差。この値を大きくするとぼかしが強くなります。  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### laplacian_filter  
`laplacian_filter(src)`

ラプラシアンフィルター

**引数**  
*src* : AfmImg形式の画像  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  


### sharpen_filter  
`sharpen_filter(src)`

鮮鋭化フィルター

**引数**  
*src* : AfmImg形式の画像  

**戻り値**  
*dst* : AfmImg形式の画像  
&emsp;&emsp; フィルターがかかった画像  

***

## OpenCVの機能拡張
- movieWriter
- writeTime

### movieWriter  
`movieWriter(path, frame_time, imgShape)`

cv2.VideoWriterをwith文に対応させ、使いやすくしたラッピングクラスです。

**引数**  
*path* : 動画の保存先  
*frame_time* : 1フレームあたりの時間、単位は秒  
*imgShape* : 画像のサイズ、shapeをそのまま指定してください  

**戻り値**  
cv2.VideoWriter : cv2.VideoWriterのインスタンス


### writeTime  
`writeTime(src, time, frame_num = "")`

OpenCVのカラー画像に時間とフレームナンバーを書き込みます。

**引数**  
*src* : OpenCVのカラー画像  
*time* : 時間、単位は秒  
*frame_num* : フレームナンバー（オプション）  

**戻り値**  
*dst* : 時間とフレームナンバーが書き込まれたOpenCV形式のカラー画像
