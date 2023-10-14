import pandas as pd
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample, shuffle
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import os

import time
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV

import os
import sys
import subprocess
import os.path




env = {'MallocNanoZone': '1', 'USER': 'vivianhuang', 'COMMAND_MODE': 'unix2003', '__CFBundleIdentifier': 'org.qgis.qgis3', 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LOGNAME': 'vivianhuang', 'SSH_AUTH_SOCK': '/private/tmp/com.apple.launchd.sGOU34JwDH/Listeners', 'PYQGIS_STARTUP': 'pyqgis-startup.py', 'HOME': '/Users/vivianhuang', 'MallocSpaceEfficient': '1', 'SHELL': '/bin/zsh', 'EXTENSION_KIT_EXTENSION_TYPE': '2', 'TMPDIR': '/var/folders/rl/b_rhkl_n5bn9m_27ts9r5rjc0000gn/T/', '__CF_USER_TEXT_ENCODING': '0x1F5:0x0:0x0', 'QT_AUTO_SCREEN_SCALE_FACTOR': '1', 'XPC_SERVICE_NAME': 'application.org.qgis.qgis3.2168055.2169234', 'XPC_FLAGS': '0x0', 'QT3D_RENDERER': 'opengl', 'GDAL_DRIVER_PATH': '/Applications/QGIS-LTR.app/Contents/MacOS/lib/gdalplugins', 'GDAL_DATA': '/Applications/QGIS-LTR.app/Contents/Resources/gdal', 'PYTHONHOME': '/Applications/QGIS-LTR.app/Contents/MacOS', 'GDAL_PAM_PROXY_DIR': '/Users/vivianhuang/Library/Application Support/QGIS/QGIS3/profiles/default/gdal_pam/', 'GISBASE': '/Applications/QGIS-LTR.app/Contents/MacOS/grass', 'GRASS_PAGER': 'cat', 'LC_CTYPE': 'UTF-8', 'SSL_CERT_DIR': '/Applications/QGIS-LTR.app/Contents/Resources/certs', 'SSL_CERT_FILE': '/Applications/QGIS-LTR.app/Contents/Resources/certs/certs.pem'}
paths = ['/Applications/QGIS-LTR.app/Contents/MacOS/../Resources/python', '/Users/vivianhuang/Library/Application Support/QGIS/QGIS3/profiles/default/python', '/Users/vivianhuang/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins', '/Applications/QGIS-LTR.app/Contents/MacOS/../Resources/python/plugins', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/scipy-1.5.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/Fiona-1.8.13.post1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/pyproj-3.2.0-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/Pillow-7.2.0-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/opencv_contrib_python-4.3.0.36-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/statsmodels-0.11.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/netCDF4-1.5.4-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/numba-0.50.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/lib-dynload', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/matplotlib-3.3.0-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/geopandas-0.8.1-py3.9.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/Rtree-0.9.7-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/cftime-1.2.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/rasterio-1.1.5-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python39.zip', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/numpy-1.20.1-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/GDAL-3.3.2-py3.9-macosx-10.13.0-x86_64.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/patsy-0.5.1-py3.9.egg', '/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.9/site-packages/pandas-1.3.3-py3.9-macosx-10.13.0-x86_64.egg', '/Users/vivianhuang/Library/Application Support/QGIS/QGIS3/profiles/default/python']

for k,v in env.items():
    os.environ[k] = v

for p in paths:
    sys.path.insert(0,p) #insert the p at the front of list of the path


os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/Applications/QGIS-LTR.app/Contents/PlugIns'

os.environ['DYLD_INSERT_LIBRARIES'] = '/Applications/QGIS-LTR.app/Contents/MacOS/lib/libsqlite3.dylib'

os.environ['PYTHONPATH'] = '/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3.9'

from qgis.core import *
from qgis.utils import *
from qgis.gui import *
from qgis.PyQt import QtGui

qgs = QgsApplication([], False)
QgsApplication.setPrefixPath("/Applications/QGIS-LTR.app/Contents/MacOS",True)
print("Ready")
qgs.initQgis()
import processing

from processing.core.Processing import Processing

from qgis.analysis import QgsNativeAlgorithms

from qgis.analysis import QgsRasterCalculatorEntry,QgsRasterCalculator

QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
print("Processing")

from os import listdir
from os.path import isfile, join
from PyQt5.QtCore import QFileInfo

total_number_band = 5
mypath='/Volumes/Liting_HD/output-maxent/output/historical_predict_Dim.tif'
raw_tif_path = '/Volumes/Liting_HD/output-maxent/output/'
"""
load 
"""
input_files = ['historical_predict_Dim']
for this_file in input_files:

    output_raster_path = raw_tif_path + this_file+'full.tif'
    feedback = QgsProcessingFeedback()
    Processing.initialize()
    fileInfo = QFileInfo(mypath)
    this_rpath = fileInfo.filePath()
    baseName = fileInfo.baseName()

    this_raster = QgsRasterLayer(this_rpath,baseName)

    entries = []

    for band_num in range(total_number_band):
        ras = QgsRasterCalculatorEntry()
        ras.ref = this_file+'@'+str(band_num+1)
        ras.raster = this_raster
        ras.bandNumber = band_num+5
        entries.append(ras)
    
    print(entries)
    # calculate the five band average
    that_item = '('+this_file+'@1+'+this_file+'@2+'+this_file+'@3+'+this_file+'@4+'+this_file+'@5)/5'
    print(that_item)
    calc = QgsRasterCalculator(that_item, output_raster_path, 'GTiff', this_raster.extent(), int(this_raster.width()), int(this_raster.height()), entries)
    calc.processCalculation()



print("success!")

"""
Exit QGIS at the end.
"""

qgs.exitQgis()