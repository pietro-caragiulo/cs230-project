import tensorflow as tf
import MakePlots as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
plt.style.use('ggplot')

length = "08"
directory_000001 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,000001)
directory_00000316 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,00000316)
directory_00001 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,00001)
directory_0000316 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,0000316)
directory_0001 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,0001)
directory_000316 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,000316)
directory_001 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,001)
directory_00316 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,0316)
directory_01 = "tflog/tabnet_forest_covertype_model_combined_{0}mm_{1}/".format(length,01)

sum_file_000001 = directory_000001 + "events.out.tfevents.{0}.cardinalmoose".format(1575701683)
sum_file_00000316 = directory_00000316 + "events.out.tfevents.{0}.cardinalmoose".format(1575700465)
sum_file_00001 = directory_00001 + "events.out.tfevents.{0}.cardinalmoose".format(1575692724)
sum_file_0000316 = directory_0000316 + "events.out.tfevents.{0}.cardinalmoose".format(1575691498)
sum_file_0001 = directory_0001 + "events.out.tfevents.{0}.cardinalmoose".format(1575690258)
sum_file_000316 = directory_000316 + "events.out.tfevents.{0}.cardinalmoose".format(1575688752)
sum_file_001 = directory_001 + "events.out.tfevents.{0}.cardinalmoose".format(1575687499)
sum_file_00316 = directory_00316 + "events.out.tfevents.{0}.cardinalmoose".format(1575686175)
sum_file_01 = directory_01 + "events.out.tfevents.{0}.cardinalmoose".format(1575684956)

df_000001 = pd.read_csv(directory_000001 + 'combined_{0}mm_{1}_out.csv'.format(length,000001))
df_00000316 = pd.read_csv(directory_00000316 + 'combined_{0}mm_{1}_out.csv'.format(length,00000316))
df_00001 = pd.read_csv(directory_00001 + 'combined_{0}mm_{1}_out.csv'.format(length,00001))
df_0000316 = pd.read_csv(directory_0000316 + 'combined_{0}mm_{1}_out.csv'.format(length,0000316))
df_0001 = pd.read_csv(directory_0001 + 'combined_{0}mm_{1}_out.csv'.format(length,0001))
df_000316 = pd.read_csv(directory_000316 + 'combined_{0}mm_{1}_out.csv'.format(length,000316))
df_001 = pd.read_csv(directory_001 + 'combined_{0}mm_{1}_out.csv'.format(length,001))
df_00316 = pd.read_csv(directory_00316 + 'combined_{0}mm_{1}_out.csv'.format(length,0316))
df_01 = pd.read_csv(directory_01 + 'combined_{0}mm_{1}_out.csv'.format(length,01))

PDFbasename = "plots/{0}mm_plots".format(length)
uncVZi = 0
minVZ = -50
maxVZ = 80
fpr_max = 0.25
threshold_min = 0.9
nBins = 150
eps = 10e-6

sparsArr = []
SparsArr.append(0.000001)
SparsArr.append(0.00000316)
SparsArr.append(0.00001)
SparsArr.append(0.0000316)
SparsArr.append(0.0001)
SparsArr.append(0.000316)
SparsArr.append(0.001)
SparsArr.append(0.00316)
SparsArr.append(0.01)

def getMaximum(y,y_predict):
    maximum = 0.
    for i in range(len(y[:,0])):
        if(y[i,0] == 1): continue
        if(y_predict[i,1] > maximum):
            maximum = y_predict[i,1]
    return maximum

def ReadSum(path_file):
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

    stepLoss = []
    for i in range(1,countLoss+1):
        stepLoss.append(i)

    stepAcc = []
    stepSize = int(countLoss/countAcc)
    for i in range(1,countAcc+1):
        stepAcc.append(i*stepSize)
    return stepLoss, trainLoss, valLoss, stepAcc, valAcc, testAcc 

def PlotLoss(steps,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9,title="",PNGbasename="test"):
    plt.clf()
    plt.plot(steps,plot1,label="Loss Weight = 0.000001")
    plt.plot(steps,plot2,label="Loss Weight = 0.000003")
    plt.plot(steps,plot3,label="Loss Weight = 0.00001")
    plt.plot(steps,plot4,label="Loss Weight = 0.00003")
    plt.plot(steps,plot5,label="Loss Weight = 0.0001")
    plt.plot(steps,plot6,label="Loss Weight = 0.0003")
    plt.plot(steps,plot7,label="Loss Weight = 0.001")
    plt.plot(steps,plot8,label="Loss Weight = 0.003")
    plt.plot(steps,plot9,label="Loss Weight = 0.01")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('{0} Loss'format(title))
    plt.legend(loc=1)
    PNGname = PNGbasename + "_loss.png"
    plt.savefig(PNGname)

def PlotAcc(steps,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9,title="",PNGbasename="test"):
    plt.clf()
    plt.plot(steps,plot1,label="Loss Weight = 0.000001")
    plt.plot(steps,plot2,label="Loss Weight = 0.000003")
    plt.plot(steps,plot3,label="Loss Weight = 0.00001")
    plt.plot(steps,plot4,label="Loss Weight = 0.00003")
    plt.plot(steps,plot5,label="Loss Weight = 0.0001")
    plt.plot(steps,plot6,label="Loss Weight = 0.0003")
    plt.plot(steps,plot7,label="Loss Weight = 0.001")
    plt.plot(steps,plot8,label="Loss Weight = 0.003")
    plt.plot(steps,plot9,label="Loss Weight = 0.01")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('{0} Accuracy'.format(title))
    plt.legend(loc=4)
    PNGname = PNGbasename + "_acc.png"
    plt.savefig(PNGname)

def ReadCsv(df):
    df_vz = df['vz']
    df_m = df['uncM']
    df_yhat = df['yhat']
    df_y = df['y']
    return df_vz, df_m, df_yhat, df_y

def ReshapeArr(df_vz, df_y, df_yhat):
    X_test = np.tile(np.asarray(df_vz).reshape(-1,1),(1,2))
    Y_test = np.tile(np.asarray(df_y).reshape(-1,1),(1,2))
    Y_test_proba =  np.tile(np.asarray(df_yhat).reshape(-1,1),(1,2))
    return X_test, Y_test, Y_test_proba

def SignalDetection(clf_cut,eps):
    return

def GetRocCurves(Y, y_predictions_proba):
    fpr, tpr, threshold = roc_curve(Y, y_predictions_proba[:,1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def MakeRocCurves(fpr1,tpr1,fpr2,tpr2,fpr3,tpr3,fpr4,tpr4,fpr5,tpr5,fpr6,tpr6,fpr7,tpr7,fpr8,tpr8,fpr9,tpr9,PDFbasename=""):
    PDFbasename = PDFbasename + "_roc"
    PDFname = PDFbasename + ".png"
    plt.clf()
    fig, ((ax0, ax1),(ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(20,24))
    plt.plot(fpr1, tpr1, label = 'Loss Weight = 0.000001')
    plt.plot(fpr2, tpr2, label = 'Loss Weight = 0.00000316')
    plt.plot(fpr3, tpr3, label = 'Loss Weight = 0.00001')
    plt.plot(fpr4, tpr4, label = 'Loss Weight = 0.0000316')
    plt.plot(fpr5, tpr5, label = 'Loss Weight = 0.0001')
    plt.plot(fpr6, tpr6, label = 'Loss Weight = 0.000316')
    plt.plot(fpr7, tpr7, label = 'Loss Weight = 0.001')
    plt.plot(fpr8, tpr8, label = 'Loss Weight = 0.00316')
    plt.plot(fpr9, tpr9, label = 'Loss Weight = 0.01')
    plt.title('ROC Curve',fontsize=20)
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.legend(loc=1)
    plt.savefig(PDFname)

def plotAUC(spars,auc,PDFbasename=""):
    plt.clf()
    plt.plot(spars,auc,label="")
    plt.xlabel('Sparsity Loss Weight')
    plt.ylabel('AUC')
    plt.title('Area Under ROC Curve')
    plt.xscale("log")
    PNGname = PDFbasename + ".png"
    plt.savefig(PNGname)
    return

def plotSigAcc(spars,acc,PDFbasename=""):
    plt.clf()
    plt.plot(spars,auc,label="")
    plt.xlabel('Sparsity Loss Weight')
    plt.ylabel('Acc')
    plt.title('Signal Acc')
    plt.xscale("log")
    PNGname = PDFbasename + ".png"
    plt.savefig(PNGname)
    return

stepLoss, trainLoss_000001, valLoss_000001, stepAcc, valAcc_000001, testAcc_000001 = ReadSum(sum_file_000001)
stepLoss, trainLoss_00000316, valLoss_00000316, stepAcc, valAcc_00000316, testAcc_00000316 = ReadSum(sum_file_00000316)
stepLoss, trainLoss_00001, valLoss_00001, stepAcc, valAcc_00001, testAcc_00001 = ReadSum(sum_file_00001)
stepLoss, trainLoss_0000316, valLoss_0000316, stepAcc, valAcc_0000316, testAcc_0000316 = ReadSum(sum_file_0000316)
stepLoss, trainLoss_0001, valLoss_0001, stepAcc, valAcc_0001, testAcc_0001 = ReadSum(sum_file_0001)
stepLoss, trainLoss_000316, valLoss_000316, stepAcc, valAcc_000316, testAcc_000316 = ReadSum(sum_file_000316)
stepLoss, trainLoss_001, valLoss_001, stepAcc, valAcc_001, testAcc_001 = ReadSum(sum_file_001)
stepLoss, trainLoss_00316, valLoss_00316, stepAcc, valAcc_00316, testAcc_00316 = ReadSum(sum_file_00316)
stepLoss, trainLoss_01, valLoss_01, stepAcc, valAcc_01, testAcc_01 = ReadSum(sum_file_01)

plot.PlotLoss(stepLoss,trainLoss_000001,trainLoss_00000316,trainLoss_00001,trainLoss_0000316,trainLoss_0001,trainLoss_000316,trainLoss_001,trainLoss_00316,trainLoss_01,title="Train",PNGbasename=PDFbasename+"_trainLoss")
plot.PlotLoss(stepLoss,valLoss_000001,valLoss_00000316,valLoss_00001,valLoss_0000316,valLoss_0001,valLoss_000316,valLoss_001,valLoss_00316,valLoss_01,title="Validation",PNGbasename=PDFbasename+"_valLoss")
plot.PlotLoss(stepAcc,trainAcc_000001,trainAcc_00000316,trainAcc_00001,trainAcc_0000316,trainAcc_0001,trainAcc_000316,trainAcc_001,trainAcc_00316,trainAcc_01,title="Train",PNGbasename=PDFbasename+"_trainAcc")
plot.PlotLoss(stepAcc,valAcc_000001,valAcc_00000316,valAcc_00001,valAcc_0000316,valAcc_0001,valAcc_000316,valAcc_001,valAcc_00316,valAcc_01,title="Validation",PNGbasename=PDFbasename+"_valAcc")

df_vz_000001, df_m_000001, df_yhat_000001, df_y_000001 = ReadCsv(df_000001)
df_vz_00000316, df_m_00000316, df_yhat_00000316, df_y_00000316 = ReadCsv(df_00000316)
df_vz_00001, df_m_00001, df_yhat_00001, df_y_00001 = ReadCsv(df_00001)
df_vz_0000316, df_m_0000316, df_yhat_0000316, df_y_0000316 = ReadCsv(df_0000316)
df_vz_0001, df_m_0001, df_yhat_0001, df_y_0001 = ReadCsv(df_0001)
df_vz_000316, df_m_000316, df_yhat_000316, df_y_000316 = ReadCsv(df_000316)
df_vz_001, df_m_001, df_yhat_001, df_y_001 = ReadCsv(df_001)
df_vz_00316, df_m_00316, df_yhat_00316, df_y_00316 = ReadCsv(df_00316)
df_vz_01, df_m_01, df_yhat_01, df_y_01 = ReadCsv(df_01)

X_test_000001, Y_test_000001, Y_test_proba_000001 = ReshapeArr(df_vz_000001, df_y_000001, df_yhat_000001)
X_test_00000316, Y_test_00000316, Y_test_proba_00000316 = ReshapeArr(df_vz_00000316, df_y_00000316, df_yhat_00000316)
X_test_00001, Y_test_00001, Y_test_proba_00001 = ReshapeArr(df_vz_00001, df_y_00001, df_yhat_00001)
X_test_0000316, Y_test_0000316, Y_test_proba_0000316 = ReshapeArr(df_vz_0000316, df_y_0000316, df_yhat_0000316)
X_test_0001, Y_test_0001, Y_test_proba_0001 = ReshapeArr(df_vz_0001, df_y_0001, df_yhat_0001)
X_test_000316, Y_test_000316, Y_test_proba_000316 = ReshapeArr(df_vz_000316, df_y_000316, df_yhat_000316)
X_test_001, Y_test_001, Y_test_proba_001 = ReshapeArr(df_vz_001, df_y_001, df_yhat_001)
X_test_00316, Y_test_00316, Y_test_proba_00316 = ReshapeArr(df_vz_00316, df_y_00316, df_yhat_00316)
X_test_01, Y_test_01, Y_test_proba_01 = ReshapeArr(df_vz_01, df_y_01, df_yhat_01)

clf_cut_arr = []
clf_cut_arr.append(getMaximum(Y_test_000001,Y_test_proba_000001) + eps)
clf_cut_arr.append(getMaximum(Y_test_00000316,Y_test_proba_00000316) + eps)
clf_cut_arr.append(getMaximum(Y_test_00001,Y_test_proba_00001) + eps)
clf_cut_arr.append(getMaximum(Y_test_0000316,Y_test_proba_0000316) + eps)
clf_cut_arr.append(getMaximum(Y_test_0001,Y_test_proba_0001) + eps)
clf_cut_arr.append(getMaximum(Y_test_000316,Y_test_proba_000316) + eps)
clf_cut_arr.append(getMaximum(Y_test_001,Y_test_proba_001) + eps)
clf_cut_arr.append(getMaximum(Y_test_00316,Y_test_proba_00316) + eps)
clf_cut_arr.append(getMaximum(Y_test_01,Y_test_proba_01) + eps)

for i in range(len(clf_cut_arr)):
    print(clf_cut_arr[i])

fpr_000001, tpr_000001, roc_auc_000001 = GetRocCurves(df_Y_000001, y_test_proba_000001)
fpr_00000316, tpr_00000316, roc_auc_00000316 = GetRocCurves(df_Y_00000316, y_test_proba_00000316)
fpr_00001, tpr_00001, roc_auc_00001 = GetRocCurves(df_Y_00001, y_test_proba_00001)
fpr_0000316, tpr_0000316, roc_auc_0000316 = GetRocCurves(df_Y_0000316, y_test_proba_0000316)
fpr_0001, tpr_0001, roc_auc_0001 = GetRocCurves(df_Y_0001, y_test_proba_0001)
fpr_000316, tpr_000316, roc_auc_000316 = GetRocCurves(df_Y_000316, y_test_proba_000316)
fpr_001, tpr_001, roc_auc_001 = GetRocCurves(df_Y_001, y_test_proba_001)
fpr_00316, tpr_00316, roc_auc_00316 = GetRocCurves(df_Y_00316, y_test_proba_00316)
fpr_01, tpr_01, roc_auc_01 = GetRocCurves(df_Y_01, y_test_proba_01)

AUC_arr = []
AUC_arr.append(roc_auc_000001)
AUC_arr.append(roc_auc_00000316)
AUC_arr.append(roc_auc_00001)
AUC_arr.append(roc_auc_0000316)
AUC_arr.append(roc_auc_0001)
AUC_arr.append(roc_auc_000316)
AUC_arr.append(roc_auc_001)
AUC_arr.append(roc_auc_00316)
AUC_arr.append(roc_auc_01)

MakeRocCurves(fpr_000001, tpr_000001, fpr_00000316, tpr_00000316, fpr_00001, tpr_00001, fpr_0000316, tpr_0000316 fpr_0001, tpr_0001, fpr_000316, tpr_000316 fpr_001, tpr_001, fpr_00316, tpr_00316 fpr_01, tpr_01, PDFbasename=PDFbasename+"_ROC")
plotAUC(SparsArr,AUC_arr,PDFbasename=PDFbasename+"_AUC")
#plotSigAcc(SparsArr,SigAcc,PDFbasename=PDFbasename+"_sig")

#plot.MakeClassifierOutputPlots(X_test, Y_test,Y_test_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, clf_cut=clf_cut, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_test")
#plot.MakeRocCurves(df_y, Y_test_proba, fpr_max=fpr_max, threshold_min=threshold_min, PDFbasename=PDFbasename+"_test")
#plot.MakeZPlots(X_test, Y_test, Y_test_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_test")
