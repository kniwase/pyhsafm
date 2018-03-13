# -*- coding: utf-8 -*-
import sys, cv2, csv, numpy as np, time
sys.path.remove('/Users/kniwase/Projects/niwaCV')
from niwaCV import niwaCV

file_name = sys.argv[1]
img = niwaCV.readImg(file_name)

#'''
#res_img = niwaCV.find_edge(img)
#dst = cv2.hconcat([img.getOpenCVimage(), cv2.cvtColor(niwaCV.gaussian_filter(img).getOpenCVimageGray(), cv2.COLOR_GRAY2BGR)])
'''
res_img = niwaCV.enhance_edge(img, 10)
dst = cv2.hconcat([img.getOpenCVimage(), res_img.getOpenCVimage()])
#'''

start = time.time()
res_img = niwaCV.median_filter(img, 5)
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

dst = cv2.hconcat([img.getOpenCVimage(), res_img.getOpenCVimage()])
cv2.imshow('test', dst)
cv2.waitKey(0)
