import tensorflow as tf
import MakePlots as plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getMaximum(y,y_predict,m):
    maximum = 0.
    for i in range(len(y[:,0])):
        if(y[i,0] == 1): continue
        if(m[i,0] < 0.090 or m[i,0] > 0.110): continue
        if(y_predict[i,1] > maximum):
            maximum = y_predict[i,1]
    return maximum

def MakeZPlots(X, Y, X2, Y_test_proba, uncVZi=2, uncMi = 1, minVZ=-30, maxVZ=70, threshold=0, nBins=150, PDFbasename=""):
	plt.clf()
	massSig = []
	massBck = []
	zSig = []
	zBck = []
	colorSig = []
	colorBck = []
	for i in range(len(Y[:,1])):
		if(Y_test_proba[i,1] > threshold and Y[i][1] == 1):
			if(np.random.random() < 0.005):
				massSig.append(X2[i,uncMi])
				zSig.append(X[i,uncVZi])
				colorSig.append(Y[i:1])
		elif (Y[i][1] == 0):
			massBck.append(X2[i,uncMi])
			zBck.append(X[i,uncVZi])
			colorBck.append(Y[i:1])
	#plt.scatter(X2[:,uncMi], X[:,uncVZi], c=Y[:,0][Y[:,0] == 0], alpha=0.6, label='Background')
	#plt.scatter(X2[:,uncMi], X[:,uncVZi], c=Y_test_proba[:,1]>threshold, alpha=0.6, label='Signal Identified')
	#plt.scatter(X2[:,uncMi], X[:,uncVZi], c=Y[:,1], alpha=0.6, label='Signal')
	plt.scatter(massBck, zBck, color='red', alpha=0.6, label='Data (Background)')
	plt.scatter(massSig, zSig, alpha=0.6, label='TabNet Signal Identified')
	plt.xlim(0,0.2)
	plt.ylim(minVZ, maxVZ)
	plt.legend(loc=1)
	plt.xlabel("Mass (GeV)", fontsize=25)
	plt.ylabel("Measured Decay Length (mm)", fontsize=25)
	plt.title("Data with Enhanced Signal Overlaid", fontsize=25)
    
	PNGname = PDFbasename + ".png"
	plt.savefig(PNGname)

PDFbasename = "test"
uncVZi = 0
uncMi = 0
minVZ = -30
maxVZ = 70
threshold_min = 0.9
nBins = 150

df = pd.read_csv('tflog/tabnet_forest_covertype_model_combined_2mm_data/combined_2mm_data_out.csv')
df_vz = df['vz']
df_m = df['uncM']
df_yhat = df['yhat']
df_y = df['y']
X_test = np.tile(np.asarray(df_vz).reshape(-1,1),(1,2))
X_testm = np.tile(np.asarray(df_m).reshape(-1,1),(1,2))
Y_test = np.tile(np.asarray(df_y).reshape(-1,1),(1,2))
Y_test_proba =  np.tile(np.asarray(df_yhat).reshape(-1,1),(1,2))

eps = 10e-12
clf_cut = getMaximum(Y_test,Y_test_proba,X_testm) + eps
print(clf_cut)

MakeZPlots(X_test, Y_test, X_testm, Y_test_proba, uncVZi=uncVZi, uncMi=uncMi, minVZ=minVZ, maxVZ=maxVZ, threshold=clf_cut, nBins=nBins, PDFbasename=PDFbasename+"_test")
#plot.MakeClassifierOutputPlots(X_test, Y_test,Y_test_proba, uncVZi=uncVZi, minVZ=minVZ, maxVZ=maxVZ, clf_cut=clf_cut, threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_test")