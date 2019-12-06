import tensorflow as tf
import MakePlots as plot
import numpy as np
import pandas as pd

path_to_events_file = "tflog/tabnet_forest_covertype_model_combined_08mm/events.out.tfevents.1575660180.cardinalmoose"
PDFbasename = "test"
uncVZi = 0
#minVZ = -0.5
#maxVZ = 0.5
minVZ = -50
maxVZ = 80
fpr_max = 0.25
threshold_min = 0.9
nBins = 150

def getMaximum(y,y_predict):
    maximum = 0.
    for i in range(len(y[:,0])):
        if(y[i,0] == 1): continue
        if(y_predict[i,1] > maximum):
            maximum = y_predict[i,1]
    return maximum

trainLoss = []
trainLoss = []
valLoss = []
valAcc = []
testAcc = []
ytrainhat = []
ytrain = []
yvalhat = []
yval = []
ytesthat = []
ytest = []
vztrain = []
vzval = []
vztest = []
uncmtrain = []
uncmval = []
uncmtest = []

countLoss = 0
countAcc = 0
for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if('Total_loss' in v.tag):
            if('val' in v.tag):
                countLoss = countLoss + 1
                valLoss.append(v.simple_value)
            else:
                trainLoss.append(v.simple_value)
        if(v.tag == 'Val_accuracy'):
            countAcc = countAcc + 1
            valAcc.append(v.simple_value)
        if(v.tag == 'Test_accuracy'):
            testAcc.append(v.simple_value)
        if('ytrain' in v.tag):
            if('hat' in v.tag):
                ytrainhat.append(v.simple_value)
            else:
                ytrain.append(v.simple_value)
        if('yval' in v.tag):
            if('hat' in v.tag):
                yvalhat.append(v.simple_value)
            else:
                yval.append(v.simple_value)
        if('ytest' in v.tag):
            if('hat' in v.tag):
                ytesthat.append(v.simple_value)
            else:
                ytest.append(v.simple_value)
        if('vz' in v.tag):
            if('train' in v.tag):
                vztrain.append(v.simple_value)
            elif('val' in v.tag):
                vzval.append(v.simple_value)
            else:
                vztest.append(v.simple_value)
        if('uncm' in v.tag):
            if('train' in v.tag):
                uncmtrain.append(v.simple_value)
            elif('val' in v.tag):
                uncmval.append(v.simple_value)
            else:
                uncmtest.append(v.simple_value)

stepLoss = []
for i in range(1,countLoss+1):
    stepLoss.append(i)

stepAcc = []
stepSize = int(countLoss/countAcc)
for i in range(1,countAcc+1):
    stepAcc.append(i*stepSize)

df = pd.read_csv('tflog/tabnet_forest_covertype_model_combined_08mm/combined_08mm_out.csv')
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

plot.PlotLoss(stepLoss,trainLoss,valLoss)
plot.PlotAcc(stepAcc,valAcc,testAcc)

eps = 10e-6
clf_cut = getMaximum(Y_test,Y_test_proba) + eps
clf_cut = 0.9999584
clf_cut = 0.9999999
print clf_cut

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