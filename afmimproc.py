# -*- coding: utf-8 -*-
import cv2, struct, copy, csv, numpy as np, numba, warnings, os
from scipy import signal
from sklearn.linear_model import LinearRegression

#Class
#HS-AFM像を扱うためのクラス
class afmImg():
	def __init__(self, data, XYlength, idx = None, frame_header = None):
		self._XYlength = XYlength
		self._Zdata = np.array([data.min(), data.max()])
		self._shape = np.array(data.shape)
		self._lenppixel = self._XYlength[0] / self._shape[0]
		self._ns2ppixel = (self._XYlength[0]*self._XYlength[1]) / (self._shape[0]*self._shape[1])
		self.__idx = idx
		self.frame_header = frame_header
		self.__data = data.copy()

	#accessors
	def __get_zdata(self):
		return self._Zdata
	def __set_zdata(self, value):
		raise NameError('zdata is read only')
	def __del_zdata(self):
		del self._Zdata
	zdata = property(__get_zdata, __set_zdata, __del_zdata)

	def __get_XYlength(self):
		return self._XYlength
	def __set_XYlength(self, value):
		raise NameError('XYlength is read only')
	def __del_XYlength(self):
		del self._XYlength
	XYlength = property(__get_XYlength, __set_XYlength, __del_XYlength)

	def __get_shape(self):
		return self._shape.copy()
	def __set_shape(self, value):
		raise NameError('shape is read only')
	def __del_shape(self):
		del self._shape
	shape = property(__get_shape, __set_shape, __del_shape)

	def __get_lenppixel(self):
		return self._lenppixel.copy()
	def __set_lenppixel(self, value):
		raise NameError('lenppixel is read only')
	def __del_lenppixel(self):
		del self._lenppixel
	lenppixel = property(__get_lenppixel, __set_lenppixel, __del_lenppixel)

	def __get_ns2ppixel(self):
		return self._ns2ppixel.copy()
	def __set_ns2ppixel(self, value):
		raise NameError('ns2ppixel is read only')
	def __del_ns2ppixel(self):
		del self._ns2ppixel
	ns2ppixel = property(__get_ns2ppixel, __set_ns2ppixel, __del_ns2ppixel)

	def __get_idx(self):
		return self.__idx
	def __set_idx(self, value):
		raise NameError('image index is read only')
	def __del_idx(self):
		del self.__idx
	index = property(__get_idx, __set_idx, __del_idx)

	def __get_data(self):
		return self.__data.copy()
	def __set_data(self, new_data):
		if list(self.shape) == list(new_data.shape):
			self.__data = new_data.copy()
		else:
			self.__data = cv2.resize(new_data, (self.shape[1], self.shape[0]))
		self._Zdata = np.array([self.__data.min(), self.__data.max()])
	def __del_data(self):
		del self.__data
	data = property(__get_data, __set_data, __del_data)

	#OpenCVと同様のインデックスによる画像の一部切り抜きに対応
	def __getitem__(self, slice):
		def getPartialImg(self, x, y):
			cut_data = self.data[y[0]:y[1], x[0]:x[1]]
			XYlength = [int(cut_data.shape[0]*self.lenppixel), int(cut_data.shape[0]*self.lenppixel)]
			return afmImg(cut_data, XYlength)

		if len(slice) != 2: raise IndexError
		y_slice, x_slice = slice
		y0, y1, step = y_slice.indices(self.shape[0])
		if step != 1: raise IndexError
		x0, x1, step = x_slice.indices(self.shape[1])
		if step != 1: raise IndexError
		cut_data = self.data[y0:y1, x0:x1]
		XYlength = [int(cut_data.shape[0]*self.lenppixel), int(cut_data.shape[0]*self.lenppixel)]
		return afmImg(cut_data, XYlength)

	#自身の深いコピーを生成する
	def copy(self):
		return copy.deepcopy(self)

	#OpenCVに対応したグレースケール画像を出力する
	def getOpenCVimageGray(self):
		if (self.zdata[1] - self.zdata[0]) <= 0:
			print(self.zdata[1] - self.zdata[0])
		gray_img = (self.data - self.zdata[0]) / (self.zdata[1] - self.zdata[0])
		gray_img = np.uint8(gray_img * 255)
		return gray_img

	#OpenCVに対応したカラー画像を出力する
	def getOpenCVimage(self):
		img_color = np.zeros((*self.shape, 3), np.uint8)
		img_color[:,:,0] = np.ones(self.shape, np.uint8)*19
		img_color[:,:,1] = self.getOpenCVimageGray()
		img_color[:,:,2] = np.ones(self.shape, np.uint8)*255
		return cv2.cvtColor(img_color, cv2.COLOR_HLS2BGR)


#HS-AFMのASDファイルを読み込むためのクラス
#ファイル読み書きのopenと同様にwith文に対応
class ASD_reader():
	def __init__(self, path):
		extension = os.path.splitext(path)[1]
		try:
			if extension != '.asd':
				raise IOError
		except IOError:
			print('File Type Error')
			print('%s is not a ASD file. Please select a ASD file.' % path)
			exit(-1)
		try:
			self.__file = open(path, 'rb')
		except IOError:
			print('%s cannot be opened.' % path)
			exit(-1)
		self.__header = self.__read_header()
		self.header = self.__header
		self.__FrameNum = self.__header['FrameNum']

	def __enter__(self):
		return self

	def __del__(self):
		if hasattr(self, '_ASD_handler__file'):
			self.__file.close()

	def __exit__(self, type, value, traceback):
		del self

	def __img_generator(self, start, stop, step):
		for idx in range(start, stop, step):
			yield self.__read_frame(idx)

	def __len__(self):
		return self.__FrameNum

	def __iter__(self):
		for idx in range(self.__FrameNum):
			yield self.__read_frame(idx)

	def __getitem__(self, idx):
		if type(idx) == slice:
			start = idx.start if idx.start != None else 0
			stop  = idx.stop  if idx.stop  != None else self.__FrameNum
			step  = idx.step  if idx.step  != None else 1
			if step == 0: raise IndexError
			if start < 0: start = self.__FrameNum + start
			if stop  < 0: stop  = self.__FrameNum + stop
			if start <= self.__FrameNum and stop <= self.__FrameNum:
				return self.__img_generator(start, stop, step)
			else:
				raise IndexError
		else:
			if idx < self.__FrameNum:
				return self.__read_frame(idx if idx >= 0 else self.__FrameNum + idx)
			else:
				raise IndexError

	def __read_header(self):
		self.__file.seek(0)
		header_bin = self.__file.read(165)
		header_format = '=iiiiiiiiiiiiiiiibiiiiiiiiifffiiiiiiiffffff'
		header_keys = ['FileType', 'FileHeaderSizeForSave', 'FrameHeaderSize', 'TextEncoding', 'OpeNameSize', 'CommentSizeForSave', 'DataType1ch', 'DataType2ch', 'FrameNum', 'ImageNum', 'ScanDirection', 'ScanTryNum', 'XPixel', 'YPixel', 'XScanSize', 'YScanSize', 'AveFlag', 'AverageNum', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'XRound', 'YRound', 'FrameTime', 'Sensitivity', 'PhaseSens', 'Offset1', 'Offset2', 'Offset3', 'Offset4', 'MachineNo', 'ADRange', 'ADResolution', 'MaxScanSizeX', 'MaxScanSizeY', 'PiezoConstX', 'PiezoConstY', 'PiezoConstZ', 'DriverGainZ']
		header = {key:d for key, d in zip(header_keys, struct.unpack_from(header_format, header_bin, 0))}

		OpeName_bin = self.__file.read(header['OpeNameSize'])
		OpeName = OpeName_bin.decode('shift_jis')
		header['OpeName'] = OpeName

		Comment_bin = self.__file.read(header['CommentSizeForSave'])
		Comment = Comment_bin.decode('shift_jis').replace('\r', '\n').rstrip('\n')
		header['Comment'] = Comment
		self.__file.seek(0)
		return header

	def __read_frame(self, idx):
		header_point = 165 + self.__header['OpeNameSize'] + self.__header['CommentSizeForSave']
		DriverGainZ, PiezoConstZ, XPixel, YPixel, XScanSize, YScanSize = [self.__header[key] for key in ['DriverGainZ', 'PiezoConstZ', 'XPixel', 'YPixel', 'XScanSize', 'YScanSize']]
		self.__file.seek(header_point + (32 + 2*XPixel*YPixel)*idx)
		frame_header_bin = self.__file.read(32)
		frame_header_format = '=IHHhhffbbhii'
		frame_header_keys = ['CurrentNum', 'MaxData', 'MiniData', 'XOffset', 'YOffset', 'XTilt', 'YTilt', 'LaserFlag', 'Reserved', 'Reserved', 'Reserved', 'Reserved']
		frame_header = {key:d for key, d in zip(frame_header_keys, struct.unpack_from(frame_header_format, frame_header_bin, 0))}
		data = np.fromfile(self.__file, dtype = 'int16', count = XPixel*YPixel, sep = '')
		data = 10.0/4096.0 * DriverGainZ * PiezoConstZ * (-data + 2048.0)
		data = data - np.mean(data)
		data = data.reshape(YPixel, XPixel)[::-1]
		size_times = 3
		new_size = (XPixel*size_times, YPixel*size_times)
		data = cv2.resize(data, new_size)
		img = afmImg(data, (YPixel, XPixel), idx, frame_header)
		self.__file.seek(0)
		return img

#OpenCV画像を動画として書き出す機能を使いやすくしたラッピングクラス
#with文に対応させた、その後の使い方はcv2.VideoWriterと同じ
class movieWriter:
	def __init__(self, path, frame_time, imgShape):
		self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.fps = 1.0 / frame_time
		self.movieWriter = cv2.VideoWriter(path, self.fourcc, self.fps, (imgShape[1], imgShape[0]))
	def __enter__(self):
		return self.movieWriter
	def __exit__(self, type, value, traceback):
		print("   Saving Movie   ")
		self.movieWriter.release()

#functions
#CSVファイルとして出力した画像を読み込む、多分もういらない
def imread(path):
	csv_data = np.genfromtxt(path, delimiter=",", dtype='float')
	XYlength = np.array([csv_data[1][1], csv_data[1][3]], dtype=int)
	data = [row[1:-1] for row in csv_data[4:]]
	data = np.array(data[::-1])
	size_times = 3
	new_size = (data.shape[0]*size_times, data.shape[1]*size_times) #ピクセル数ベース
	data = cv2.resize(data, new_size)
	return afmImg(data, XYlength)

def inforead(path):
	src = readImg(path)
	return afmImgInfo(src.data, src.XYlength)

def imrite(path, img):
	cv2.imwrite(path, img.getOpenCVimage())

def imrite_gray(path, img):
	cv2.imwrite(path, img.getOpenCVimageGray())

def imshow(img, text =''):
	cv2.imshow(text, img.getOpenCVimage())
	cv2.waitKey(0)

def imshow_gray(img, text =''):
	cv2.imshow(text, img.getOpenCVimageGray())
	cv2.waitKey(0)

#戻り値はヒストグラム、X軸、検出されたピーク
def histogram(img, range=None, step=0.1, order=None, smoothed=False, smoothing_order=3):
	#stepに応じて最大値最小値を丸めるラムダ式
	round_min = lambda x, step: round(x-x%step, 3)
	round_max = lambda x, step: round(x-x%step+step, 3)
	#データの1次元化
	data = img.data.flatten()
	#rangeをstepに合うように調整する、もしrangeの指定がなければ最大値最小値から自動で決定
	if range is None:
		range = [round_min(data.min(), step), round_max(data.max(), step)]
	elif len(range) == 2:
		range = [round_min(range[0], step), round_max(range[1], step)]
	else:
		raise ValueError
	#binsはstepに応じて自動で生成
	bins = int((range[1]-range[0])/step)
	#ヒストグラム作成
	hist, hist_bins = np.histogram(data, bins=bins, range=range)
	#smoothedが指定されている場合、グラフの平滑化
	if smoothed:
		kernel = cv2.getGaussianKernel(smoothing_order, 0)[:,0]
		hist = np.convolve(hist, kernel, mode='same')
	#ピーク検出の幅のデフォルト設定はstepの逆数（つまり1nm）
	if order is None:
		order = int(round(1 / step))
	#ピーク検出
	peaks = signal.argrelmax(hist, order=order)[0]

	return [hist, hist_bins[:-1], peaks]

#大津の二値化による閾値取得用関数
def threshold_otsu(img):
	threshold_8bit = cv2.threshold(img.getOpenCVimageGray(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
	threshold = (img.zdata[1]-img.zdata[0])*(threshold_8bit/255.0)+img.zdata[0]
	return threshold

#srcをlowestの高さで二値化する（高さ1.0nmまたは0.0nm）、highestが指定されるとそれ以上の部分も0.0nmになる
def binarize(src, lowest, highest = True):
	def __binarize(src, lowest):
		dst = src.copy()
		black = np.zeros(src.shape, dtype='float')
		white = black + 1.0
		dst.data = np.where(dst.data > lowest, white, black).astype("float")
		return dst

	if highest:
		dst = __binarize(src, lowest)
	else:
		mask1 = __binarize(src, lowest).data
		mask2 = cv2.bitwise_not(__binarize(src, highest).data)
		dst = src.copy()
		dst.data = cv2.bitwise_and(mask1, mask2)
	return dst

#高さヒストグラムからマイカ表面を認識し、マイカ表面が0.0nmになるように高さを修正する
#makeItZeroをTrueにすると0.0nm以下をすべて0.0nmで置換する
#peak_numは低い方から数えて何個目のピークをマイカ表面とするかを指定する
#stepはヒストグラム取得の際のパラメーター
def heightCorrection(src, makeItZero=False, peak_num=1, step=0.05):
	dst = src.copy()
	hist, bins, peak_idx = histogram(dst, step=step)
	peak = bins[peak_idx[peak_num - 1]]
	dst.data = dst.data - peak
	if makeItZero:
		black = np.zeros_like(dst.data, dtype='uint8')
		dst.data = np.where(dst.data >= 0.0, dst.data, black)
	return dst

#マイカ表面を平面近似し、そのパラメーターに基づいて傾きを補正する
#th_rangeで何nmまでをマイカ表面として扱うかを指定できる
def tiltCorrection(src, th_range = 0.25):
	dst = heightCorrection(src)
	explanatory_var = np.where(dst.data <= th_range)
	explanatory_var = list(zip(*explanatory_var))
	data = src.data
	data = np.array([data[y,x] for y, x in explanatory_var])
	warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
	clf = LinearRegression()
	clf.fit(explanatory_var, data)
	src_data = src.data
	var = list(zip(*np.where(src_data)))
	coef = clf.coef_
	data = np.array([value - np.dot(var, coef) for var, value in zip(var, src_data.flatten())])
	dst.data = data.reshape(dst.shape)
	return dst

#highestに指定した以上の高さをすべてhighestに置換する
def heightScaling(src, highest):
	dst = src.copy()
	white = np.zeros(dst.shape, dtype='float') + highest
	dst.data = np.where(dst.data <= highest, dst.data, white)
	return dst

#離散フーリエ変換を行い、フィルターをかける関数
def __dft_filter(src, func, *args):
	#離散フーリエ変換
	def dft(src):
		#高速フーリエ変換
		fimg = np.fft.fft2(src)
		#第1象限と第3象限、第2象限と第4象限を入れ替え
		dst =  np.fft.fftshift(fimg)
		return dst
	#逆離散フーリエ変換
	def idft(src):
		#第1象限と第3象限、第1象限と第4象限を入れ替え
		fimg =  np.fft.fftshift(src)
		#高速逆フーリエ変換
		dst = np.fft.ifft2(fimg)
		return dst

	#離散フーリエ変換
	dst = src.copy()
	dft_img = dft(dst.data)

	#フィルター処理
	#引数に指定した関数を使用する
	dft_img = func(dft_img, *args)

	#逆離散フーリエ変換
	idft_img = idft(dft_img)
	#虚数部を除去
	dst.data = np.abs(idft_img)
	return dst

#フィルタージェネレーター
def __make_filter(mask_src, size):
	mask = mask_src.copy()
	#フィルターのサイズ、位置設定
	size = (int(size/100.0*mask.shape[1]/2), int(size/100.0*mask.shape[0]/2))
	pos = (int(mask.shape[1]/2), int(mask.shape[0]/2))
	#sizeで指定された大きさの楕円を描画
	cv2.ellipse(mask, (pos, size, 0), 0 if mask[pos[1],pos[0]] else 255, -1)
	'''#マスクの形状確認
	cv2.imshow('', mask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	return mask

#ハイパスフィルター、sizeでフィルターのサイズを調整できる
def highpass_filter(src, size):
	def highpass(dft_img_src, *args):
		dft_img = dft_img_src.copy()
		#マスク作成
		mask = __make_filter(np.ones_like(dft_img, dtype = 'uint8')*255, args[0])
		#マスキング
		black = np.zeros_like(dft_img, dtype = 'complex128')
		dft_img = np.where(mask == 255, dft_img, black)
		return dft_img
	dst = __dft_filter(src, highpass, size)
	return dst

#ローパスフィルター、sizeでフィルターのサイズを調整できる
def lowpass_filter(src, size):
	def lowpass(dft_img_src, *args):
		dft_img = dft_img_src.copy()
		#マスク作成
		mask = __make_filter(np.zeros_like(dft_img, dtype = 'uint8'), args[0])
		#マスキング
		black = np.zeros_like(dft_img, dtype = 'complex128')
		dft_img = np.where(mask == 255, dft_img, black)
		return dft_img
	dst = __dft_filter(src, lowpass, size)
	return dst

#バンドパスフィルター、size_outer, size_innerで外側内側のフィルターのサイズを調整する
def bandpass_filter(src, size_outer, size_inner):
	def bandpass(dft_img_src, *args):
		dft_img = dft_img_src.copy()
		#マスク作成
		mask = __make_filter(np.zeros_like(dft_img, dtype = 'uint8'), args[0])
		mask = __make_filter(mask, args[1])
		#マスキング
		black = np.zeros_like(dft_img, dtype = 'complex128')
		dft_img = np.where(mask == 255, dft_img, black)
		return dft_img
	dst = __dft_filter(src, bandpass, size_outer, size_inner)
	return dst

#エッジを探すフィルター、動作が怪しい
def find_edge(src, inner=True):
	def normalize(img):
		return (img - img.min()) / (img.max() - img.min())
	gray = normalize(src.data)
	gray = normalize(cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((5, 5))))
	gray = cv2.GaussianBlur(gray, (9, 9), 0)
	#'''
	if inner:
		#白い部分を収縮させる
		edge = normalize(cv2.erode(gray, np.ones((7, 7)), iterations=1))
	else:
		#白い部分を膨張させる
		edge = normalize(cv2.dilate(gray, np.ones((11, 11)), iterations=1))
	#edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, np.ones((7, 7)))
	edge = normalize(cv2.erode(edge, np.ones((5, 5)), iterations=1))
	#差をとる
	dst = src.copy()
	dst_data = cv2.absdiff(gray, edge)
	dst_data = cv2.GaussianBlur(dst_data, (5, 5), 0)
	dst.data = normalize(dst_data)
	'''
	dst = gaussian_filter(src, 11)
	img_opencv = (normalize(dst.data) * 255).astype(np.uint8)
	img_opencv = cv2.adaptiveThreshold(img_opencv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
	dst.data = normalize(img_opencv)
	#'''
	return dst

#find_edgeで探したエッジを元画像に足し算してエッジを強調させる
def enhance_edge(src, k = 1.0, inner=True):
	def normalize(img):
		return (img - img.min()) / (img.max() - img.min())
	edge = find_edge(src)
	edge = heightCorrection(edge)
	edge = edge.data
	dst = src.copy()
	if inner:
		dst.data += (edge)*(src.data.max() - src.data.min()) * k/5.0
	else:
		dst.data -= (edge)*(src.data.max() - src.data.min()) * k/5.0
	dst = heightCorrection(dst)
	return dst

#メディアンフィルターの実装部分、numbaを使用してjitコンパイラに通すことで高速化
@numba.jit('float64[:,:](float64[:, :], int32)', nopython=True)
def __median_filter(src_data, ksize):
	h, w = src_data.shape[0], src_data.shape[1]
	h_dst, w_dst = h-(ksize-1), w-(ksize-1)
	d, med = (ksize-1)//2, ksize**2//2
	dst_data = np.empty((h_dst, w_dst), dtype = np.float64)
	for y in range(d, h-d):
		for x in range(d, w-d):
			tmp_data = src_data[y-d:y+d+1, x-d:x+d+1].flatten()
			tmp_data.sort()
			dst_data[y-d][x-d] = tmp_data[med]
	return dst_data
#メディアンフィルターを実際に使うときの関数、numbaを使用してjitコンパイラに通すことで高速化
@numba.jit
def median_filter(src, ksize = 3):
	#近傍にある画素値の中央値を出力画像の画素値に設定
	dst = src.copy()
	dst.data = __median_filter(src.data, ksize)
	return dst

#畳み込み演算を行う関数、任意のカーネルを指定できる
def convolution_filter(src, kernel):
	dst = src.copy()
	dst.data = cv2.filter2D(src.data, -1, kernel)
	return dst

#平均値フィルター
def average_filter(src, ksize = 3):
	average = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))/ksize**2
	return convolution_filter(src, average)

#ガウシアンフィルター
def gaussian_filter(src, ksize = 5, sigmaX = 0):
	gaussian = cv2.getGaussianKernel(ksize, sigmaX)
	gaussian = np.array([[x*y for x in gaussian] for y in gaussian])
	return convolution_filter(src, gaussian)

#ラプラシアンフィルター
def laplacian_filter(src):
	laplacian = np.array([
						 [-1, -3, -4, -3, -1],
						 [-3,  0,  6,  0, -3],
						 [-4,  6, 20,  6, -4],
						 [-3,  0,  6,  0, -3],
						 [-1, -3, -4, -3, -1]], np.float32)
	return convolution_filter(src, laplacian)

#鮮鋭化フィルター
def sharpen_filter(src):
	sharp = lambda k = 1: np.matrix('0,{0},0;{0},{1},{0};0,{0},0'.format(-k,1+4*k))
	return convolution_filter(src, sharp)

#OpenCVの画像用
#画像に時間を書き込む関数
def writeTime(src, time, frame_num = ""):
	dst = src.copy()
	round=lambda x:(x*10.0*2+1)//2/10.0
	txt = str(round(time)) + "s"
	font = cv2.FONT_HERSHEY_DUPLEX
	font_size = 1.2
	position = (15, 45)
	dst = cv2.putText(dst, txt, position, font, font_size, (0, 0, 0), 6, cv2.LINE_AA)
	dst = cv2.putText(dst, txt, position, font, font_size, (255, 255, 255), 2, cv2.LINE_AA)
	if frame_num != "":
		txt = frame_num
		font = cv2.FONT_HERSHEY_DUPLEX
		position = (dst.shape[1] - 35, dst.shape[0]-4)
		font_size = 0.4
		dst = cv2.putText(dst, txt, position, font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
		dst = cv2.putText(dst, txt, position, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)
	return dst