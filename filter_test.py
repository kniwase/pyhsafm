# -*- coding: utf-8 -*-
import sys, cv2, csv, numpy as np
sys.path.remove('/Users/kniwase/Projects/niwaCV')
from niwaCV import niwaCV

file_name = sys.argv[1]
img = niwaCV.readImg(file_name)

'''
res_img = niwaCV.convolution_filter(img, niwaCV.Kernels.high_pass)
res_img = niwaCV.convolution_filter(res_img, niwaCV.Kernels.gaussian)
#res_img = niwaCV.heightCorrection(res_img)
normalize = lambda d: d/(d.max() - d.min())
res_img.data = normalize(res_img.data)*10


res_img.data += img.data
res_img = niwaCV.convolution_filter(res_img, niwaCV.Kernels.gaussian)
dst = cv2.hconcat([img.getOpenCVimage(), res_img.getOpenCVimage()])

#dst = cv2.hconcat([img.getOpenCVimage(), cv2.cvtColor(res_img.getOpenCVimageGray(), cv2.COLOR_GRAY2BGR)])
'''

'''
res_img = niwaCV.find_edge(img)
dst = cv2.hconcat([img.getOpenCVimage(), cv2.cvtColor(res_img.getOpenCVimageGray(), cv2.COLOR_GRAY2BGR)])
'''
res_img = niwaCV.enhance_edge(img, 25)
dst = cv2.hconcat([img.getOpenCVimage(), res_img.getOpenCVimage()])
#'''

cv2.imshow('test', dst)
cv2.waitKey(0)
