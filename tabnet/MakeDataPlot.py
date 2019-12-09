import tensorflow as tf
import MakePlots as plot
import numpy as np
import pandas as pd

def getMaximum(y,y_predict):
    maximum = 0.
    for i in range(len(y[:,0])):
        if(y[i,0] == 1): continue
        if(y_predict[i,1] > maximum):
            maximum = y_predict[i,1]
    return maximum

PDFbasename = "test"
uncVZi = 0
minVZ = -50
maxVZ = 80
fpr_max = 0.25
threshold_min = 0.9
nBins = 150

df = pd.read_csv('tflog/tabnet_forest_covertype_model_combined_2mm_data/combined_2mm_data_out.csv')
df_vz = df['vz']
df_m = df['uncM']
df_yhat = df['yhat']
df_y = df['y']
X_test = np.tile(np.asarray(df_vz).reshape(-1,1),(1,2))
Y_test = np.tile(np.asarray(df_y).reshape(-1,1),(1,2))
Y_test_proba =  np.tile(np.asarray(df_yhat).reshape(-1,1),(1,2))

X_train = np.tile(np.asarray(vztrain).reshape(-1,1),(1,2))
X_val = np.tile(np.asarray(vzval).reshape(-1,1),(1,2))
#X_test = np.tile(np.asarray(vztest).reshape(-1,1),(1,2))
Y_train = np.tile(np.asarray(ytrain).reshape(-1,1),(1,2))
Y_val = np.tile(np.asarray(yval).reshape(-1,1),(1,2))
#Y_test = np.tile(np.asarray(ytest).reshape(-1,1),(1,2))
Y_train_proba =  np.tile(np.asarray(yvalhat).reshape(-1,1),(1,2))
Y_val_proba =  np.tile(np.asarray(yvalhat).reshape(-1,1),(1,2))
#Y_test_proba =  np.tile(np.asarray(ytesthat).reshape(-1,1),(1,2))

eps = 10e-6
clf_cut = getMaximum(Y_test,Y_test_proba) + eps
print(clf_cut)

#plot.MakeClassifierOutputPlots(X_train, Y_train,Y_train_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, clf_cut=clf_cut, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_train")
#plot.MakeRocCurves(ytrain, Y_train_proba, fpr_max=fpr_max, threshold_min=threshold_min, PDFbasename=PDFbasename+"_train")
#plot.MakeZPlots(X_train, Y_train, Y_train_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_train")

#plot.MakeClassifierOutputPlots(X_val, Y_val,Y_val_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, clf_cut=clf_cut, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_val")
#plot.MakeRocCurves(yval, Y_val_proba, fpr_max=fpr_max, threshold_min=threshold_min, PDFbasename=PDFbasename+"_val")
#plot.MakeZPlots(X_val, Y_val, Y_val_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_val")

plot.MakeClassifierOutputPlots(X_test, Y_test,Y_test_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, clf_cut=clf_cut, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_test")
#plot.MakeRocCurves(ytest, Y_test_proba, fpr_max=fpr_max, threshold_min=threshold_min, PDFbasename=PDFbasename+"_test")
plot.MakeRocCurves(df_y, Y_test_proba, fpr_max=fpr_max, threshold_min=threshold_min, PDFbasename=PDFbasename+"_test")
plot.MakeZPlots(X_test, Y_test, Y_test_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_test")
