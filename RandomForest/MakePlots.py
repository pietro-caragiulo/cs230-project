import numpy as np
#import root_numpy as rnp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc

def MakePlots(clf, param_list, param_min, param_max, 
	X_train=None, Y_train=None, X_val=None, Y_val=None, X_test=None, Y_test=None, 
	uncVZi=2, clf_cut=0.5, threshold_min=0.9, 
	nBins=150, fpr_max=0.25, PDFbasename=""):
	#if(X_train != None and Y_train != None):
	y_predictions = clf.predict(X_train)
	y_predictions_proba = clf.predict_proba(X_train)
	MakeClassifierOutputPlots(X_train, Y_train, y_predictions_proba, uncVZi=uncVZi, 
		minVZ=param_min[uncVZi], maxVZ=param_max[uncVZi], clf_cut=clf_cut, 
		threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_train")
	MakeRocCurves(Y_train, y_predictions_proba, fpr_max=fpr_max, threshold_min=threshold_min, 
		PDFbasename=PDFbasename+"_train")
	MakeZPlots(X_train, Y_train, y_predictions_proba, uncVZi=uncVZi, minVZ=param_min[uncVZi], 
		maxVZ=param_max[uncVZi], threshold_min=threshold_min, nBins=nBins, 
		PDFbasename=PDFbasename+"_train")
	MakePhysicsPlots(X_train, Y_train, y_predictions, y_predictions_proba, param_list, 
		param_min, param_max, uncVZi=uncVZi, clf_cut=clf_cut, threshold_min=threshold_min, 
		nBins=nBins, PDFbasename=PDFbasename+"_train")

	if(X_val != None and Y_val != None):
		y_predictions = clf.predict(X_val)
		y_predictions_proba = clf.predict_proba(X_val)
		MakeClassifierOutputPlots(X_val, Y_val, y_predictions_proba, uncVZi=uncVZi, 
			minVZ=param_min[uncVZi], maxVZ=param_max[uncVZi], clf_cut=clf_cut, 
			threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_val")
		MakeRocCurves(Y_val, y_predictions_proba, fpr_max=fpr_max, threshold_min=threshold_min, 
			PDFbasename=PDFbasename+"_val")
		MakeZPlots(X_val, Y_val, y_predictions_proba, uncVZi=uncVZi, minVZ=param_min[uncVZi], 
			maxVZ=param_max[uncVZi], threshold_min=threshold_min, nBins=nBins, 
			PDFbasename=PDFbasename+"_val")
		MakePhysicsPlots(X_val, Y_val, y_predictions, y_predictions_proba, param_list, 
			param_min, param_max, uncVZi=uncVZi, clf_cut=clf_cut, threshold_min=threshold_min, 
			nBins=nBins, PDFbasename=PDFbasename+"_val")

	if(X_test != None and Y_test != None):
		y_predictions = clf.predict(X_test)
		y_predictions_proba = clf.predict_proba(X_test)
		MakeClassifierOutputPlots(X_test, Y_test, y_predictions_proba, uncVZi=uncVZi, 
			minVZ=param_min[uncVZi], maxVZ=param_max[uncVZi], clf_cut=clf_cut, 
			threshold_min=threshold_min, nBins=nBins, PDFbasename=PDFbasename+"_test")
		MakeRocCurves(Y_test, y_predictions_proba, fpr_max=fpr_max, threshold_min=threshold_min, 
			PDFbasename=PDFbasename+"_test")
		MakeZPlots(X_test, Y_test, y_predictions_proba, uncVZi=uncVZi, minVZ=param_min[uncVZi], 
			maxVZ=param_max[uncVZi], threshold_min=threshold_min, nBins=nBins, 
			PDFbasename=PDFbasename+"_test")
		MakePhysicsPlots(X_test, Y_test, y_predictions, y_predictions_proba, param_list, 
			param_min, param_max, uncVZi=uncVZi, clf_cut=clf_cut, threshold_min=threshold_min, 
			nBins=nBins, PDFbasename=PDFbasename+"_test")

def MakeClassifierOutputPlots(X, Y, y_predictions_proba, uncVZi=2, minVZ=0, maxVZ=100, 
	clf_cut=0.5, threshold_min=0.9, nBins=150, PDFbasename=""):
	PDFbasename = PDFbasename + "_classoutput"
	PDFname = PDFbasename + ".pdf"
	pp = PdfPages(PDFname)

	fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(20,24))
	zcut = 17

	ax0.hist(X[:, uncVZi][Y[:,0] == 0], bins=150, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Background")
	ax0.hist(X[:, uncVZi][np.logical_and(y_predictions_proba[:,1] < clf_cut, Y[:,0] == 1)], bins=150, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Signal Identified as Background")
	ax1.hist(X[:, uncVZi][Y[:,0] == 1], bins=150, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Signal")
	ax1.hist(X[:, uncVZi][np.logical_and(y_predictions_proba[:,1] > clf_cut, Y[:,0] == 0)], bins=150, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Background Identified as Signal")

	ax2.hist(X[:, uncVZi][Y[:,0] == 0], bins=150, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Background")
	ax2.hist(X[:, uncVZi][np.logical_and(y_predictions_proba[:,1] < clf_cut, Y[:,0] == 0)], bins=150, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Background Identified as Background")
	ax3.hist(X[:, uncVZi][Y[:,0] == 1], bins=150, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Signal")
	n, bins, _ = ax3.hist(X[:, uncVZi][np.logical_and(y_predictions_proba[:,1] > clf_cut, Y[:,0] == 1)], nBins, range=(minVZ, maxVZ), alpha=0.8, histtype="stepfilled", label="Signal Identified as Signal")

	ax4.hist(y_predictions_proba[:,1][Y[:,0] == 0], bins=150, range=(0, 1), alpha=0.8, histtype="stepfilled", label="Background")
	ax4.hist(y_predictions_proba[:,1][Y[:,0] == 1], bins=150, range=(0, 1), alpha=0.8, histtype="stepfilled", label="Signal")

	ax5.hist(y_predictions_proba[:,1][Y[:,0] == 0], bins=150, range=(threshold_min, 1), alpha=0.8, histtype="stepfilled", label="Background")
	ax5.hist(y_predictions_proba[:,1][Y[:,0] == 1], bins=150, range=(threshold_min, 1), alpha=0.8, histtype="stepfilled", label="Signal")

	ax0.set_yscale("log")
	ax1.set_yscale("log")
	ax2.set_yscale("log")
	ax3.set_yscale("log")
	ax4.set_yscale("log")
	ax5.set_yscale("log")

	ax0.set_ylim(0.5)
	ax1.set_ylim(0.5)
	ax2.set_ylim(0.5)
	ax3.set_ylim(0.5)
	ax4.set_ylim(0.5)
	ax5.set_ylim(0.5)

	ax0.legend(loc=1)
	ax1.legend(loc=1)
	ax2.legend(loc=1)
	ax3.legend(loc=1)
	ax4.legend(loc=1)
	ax5.legend(loc=1)

	ax0.set_xlabel("Measured Decay Length (mm)", fontsize=20)
	ax1.set_xlabel("Measured Decay Length (mm)", fontsize=20)
	ax2.set_xlabel("Measured Decay Length (mm)", fontsize=20)
	ax3.set_xlabel("Measured Decay Length (mm)", fontsize=25)
	ax4.set_xlabel("Classifier Output", fontsize=25)
	ax5.set_xlabel("Classifier Output", fontsize=20)

	pp.savefig(fig)
	pp.close()

	bin_width = bins[1] - bins[0]
	zCut_bin = int((zcut-minVZ)/(maxVZ-minVZ)*nBins)
	signal_yield_old = bin_width * sum(n[zCut_bin:nBins])
	signal_yield_new = bin_width * sum(n[0:nBins])
	print (zCut_bin)
	print (signal_yield_old)
	print (signal_yield_new)
	print (signal_yield_new/signal_yield_old)


def MakeRocCurves(Y, y_predictions_proba, fpr_max=0.25, threshold_min=0.9,PDFbasename=""):
	PDFbasename = PDFbasename + "_roc"
	PDFname = PDFbasename + ".pdf"
	pp = PdfPages(PDFname)

	fpr, tpr, threshold = roc_curve(Y, y_predictions_proba[:,1])
	roc_auc = auc(fpr, tpr)
	roc_auc_max = auc(fpr, tpr)

	fig, ((ax0, ax1),(ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(20,24))
	ax0.set_title('ROC Curve; AUC = ' + str(roc_auc),fontsize=20)
	ax0.plot(fpr, tpr, label = '')
	ax0.set_ylim([0, 1])
	ax0.set_ylabel('True Positive Rate',fontsize=20)
	ax0.set_xlabel('False Positive Rate',fontsize=20)

	ax1.set_title('ROC Curve Zoom',fontsize=20)
	ax1.plot(fpr, tpr, label = '')
	ax1.set_xlim([0, fpr_max])
	ax1.set_ylim([0, 1])
	ax1.set_ylabel('True Positive Rate',fontsize=20)
	ax1.set_xlabel('False Positive Rate',fontsize=20)

	ax2.set_title('False Positive Rate vs Threshold',fontsize=20)
	ax2.plot(threshold, fpr, label = '')
	ax2.set_xlabel('Classifier Threshold',fontsize=20)
	ax2.set_ylabel('False Positive Rate',fontsize=20)

	ax3.set_title('False Positive Rate vs Threshold',fontsize=20)
	ax3.plot(threshold, fpr, label = '')
	ax3.set_xlim([threshold_min, 1])
	ax3.set_ylim([0, fpr_max])
	ax3.set_xlabel('Classifier Threshold',fontsize=20)
	ax3.set_ylabel('False Positive Rate',fontsize=20)

	ax4.set_title('True Positive Rate vs Threshold',fontsize=20)
	ax4.plot(threshold, tpr, label = '')
	ax4.set_xlim([0, 1])
	ax4.set_ylim([0, 1])
	ax4.set_ylabel('True Positive Rate',fontsize=20)
	ax4.set_xlabel('Classifier Threshold',fontsize=20)

	ax5.set_title('True Positive Rate vs Threshold',fontsize=20)
	ax5.plot(threshold, tpr, label = '')
	ax5.set_xlim([threshold_min, 1])
	ax5.set_ylim([0, 1])
	ax5.set_ylabel('True Positive Rate',fontsize=20)
	ax5.set_xlabel('Classifier Threshold',fontsize=20)

	#ax0.set_yscale("log")
	#ax1.set_yscale("log")
	#ax2.set_yscale("log")
	#ax3.set_yscale("log")
	#ax4.set_yscale("log")
	#ax5.set_yscale("log")

	#ax0.set_xscale("log")
	#ax1.set_xscale("log")
	#ax2.set_xscale("log")
	#ax3.set_xscale("log")
	#ax4.set_xscale("log")
	#ax5.set_xscale("log")

	pp.savefig(fig)
	pp.close()

def MakeZPlots(X, Y, y_predictions_proba, uncVZi=2, minVZ=0, maxVZ=100, 
	threshold_min=0.9, nBins=150, PDFbasename=""):
	PDFbasename = PDFbasename + "_zplots"
	PDFname = PDFbasename + ".pdf"
	pp = PdfPages(PDFname)

	fig, ((ax0, ax1),(ax2, ax3),(ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(20,24))
	ax0.scatter(y_predictions_proba[:,1], X[:,uncVZi], c=Y[:,0], alpha=0.6)
	ax1.scatter(y_predictions_proba[:,1], X[:,uncVZi], c=Y[:,0], alpha=0.6)
	ax0.set_xlim(0,1)
	ax0.set_ylim(minVZ, maxVZ)
	ax0.set_xlabel("Classifier Output", fontsize=20)
	ax0.set_ylabel("Measured Decay Length (mm)", fontsize=20)
	ax1.set_xlabel("Classifier Output", fontsize=20)
	ax1.set_ylabel("Measured Decay Length (mm)", fontsize=20)
	ax1.set_xlim(threshold_min,1)
	ax1.set_ylim(minVZ, maxVZ)

	ax2.hist2d(y_predictions_proba[:,1][Y[:,0] == 1], X[:,uncVZi][Y[:,0] == 1], bins=100, range=[[0,1],[minVZ, maxVZ]], alpha=0.6, cmin=0.5)
	ax3.hist2d(y_predictions_proba[:,1][Y[:,0] == 1], X[:,uncVZi][Y[:,0] == 1], bins=100, range=[[threshold_min,1],[minVZ, maxVZ]], alpha=0.6, cmin=0.5)
	ax2.set_xlabel("Classifier Output", fontsize=20)
	ax2.set_ylabel("Measured Decay Length (mm)", fontsize=20)
	ax2.set_title("Signal", fontsize=20)
	ax3.set_xlabel("Classifier Output", fontsize=20)
	ax3.set_ylabel("Measured Decay Length (mm)", fontsize=20)
	ax3.set_title("Signal", fontsize=20)

	ax4.hist2d(y_predictions_proba[:,1][Y[:,0] == 0], X[:,uncVZi][Y[:,0] == 0], bins=100, range=[[0,1],[minVZ, maxVZ]], alpha=0.6, cmin=0.5, norm=LogNorm())
	ax5.hist2d(y_predictions_proba[:,1][Y[:,0] == 0], X[:,uncVZi][Y[:,0] == 0], bins=100, range=[[threshold_min,1],[minVZ, maxVZ]], alpha=0.6, cmin=0.5)
	ax4.set_xlabel("Classifier Output", fontsize=20)
	ax4.set_ylabel("Measured Decay Length (mm)", fontsize=20)
	ax4.set_title("Background", fontsize=20)
	ax5.set_xlabel("Classifier Output", fontsize=20)
	ax5.set_ylabel("Measured Decay Length (mm)", fontsize=20)
	ax5.set_title("Background", fontsize=20)

	#ax4.set_zscale("log")
	ax4.set_xlabel("Classifier Output", fontsize=25)
	ax4.set_ylabel("Measured Decay Length (mm)", fontsize=25)
	ax4.set_title("Background", fontsize=30)

	pp.savefig(fig)
	pp.close()

def MakePhysicsPlots(X, Y, y_predictions, y_predictions_proba, param_list, 
	param_min, param_max, uncVZi=2, clf_cut=0.5, threshold_min=0.9, nBins=150, PDFbasename=""):
	PDFbasename = PDFbasename + "_physicsplots"
	PDFname = PDFbasename + ".pdf"
	pp = PdfPages(PDFname)

	i = 0
	for name in param_list:
		fig, ((ax0, ax1),(ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(20,24))
		fig2, ((ax6, ax7), (ax8, ax9), (ax10, ax11)) = plt.subplots(nrows=3, ncols=2, figsize=(20,24))

		ax0.hist(X[:, i][Y[:,0] == 0], bins=150, alpha=0.8, range=(param_min[i], param_max[i]), histtype="stepfilled", label="Background")
		ax0.hist(X[:, i][Y[:,0] == 1], bins=150, alpha=0.8, range=(param_min[i], param_max[i]), histtype="stepfilled", label="Signal")
		ax1.hist(X[:, i][y_predictions_proba[:,1] < clf_cut], bins=150, alpha=0.8, range=(param_min[i], param_max[i]), histtype="stepfilled", label="Identified as Background")
		ax1.hist(X[:, i][y_predictions_proba[:,1] > clf_cut], bins=150, alpha=0.8, range=(param_min[i], param_max[i]), histtype="stepfilled", label="Identified as Signal")

		ax0.set_title(name, fontsize=20)
		ax0.set_xlabel(name, fontsize=20)
		ax0.legend(loc=2)
		ax1.set_title(name, fontsize=20)
		ax1.set_xlabel(name, fontsize=20)
		ax1.legend(loc=2)
    
		ax2.scatter(X[:, i], X[:, uncVZi], c=y_predictions,label = name)
		ax3.scatter(X[:, i], X[:, uncVZi], c=y_predictions_proba[:,1] > clf_cut,label = name)
    
		ax2.set_title('Signal and Background', fontsize=20)
		ax2.set_xlabel(name, fontsize=20)
		ax2.set_ylabel('Measured Decay Length (mm)', fontsize=20)  
		ax2.set_xlim(param_min[i], param_max[i])
		ax2.set_ylim(param_min[uncVZi], param_max[uncVZi])
		ax3.set_title('Identified Signal and Background', fontsize=20)
		ax3.set_xlabel(name, fontsize=20)
		ax3.set_ylabel('Measured Decay Length (mm)', fontsize=20)
		ax3.set_xlim(param_min[i], param_max[i])
		ax3.set_ylim(param_min[uncVZi], param_max[uncVZi])
    
		ax4.hist2d(X[:, i][Y[:,0] == 0], X[:, uncVZi][Y[:,0] == 0], range=[[param_min[i], param_max[i]],[param_min[uncVZi], param_max[uncVZi]]], bins=150, alpha=0.6, label="Background", cmin=0.5)
		ax5.hist2d(X[:, i][Y[:,0] == 1], X[:, uncVZi][Y[:,0] == 1], range=[[param_min[i], param_max[i]],[param_min[uncVZi], param_max[uncVZi]]], bins=150, alpha=0.6, label="Signal", cmin=0.5)
    
		ax4.set_title('Background', fontsize=20)
		ax4.set_xlabel(name, fontsize=20)
		ax4.set_ylabel('Measured Decay Length (mm)', fontsize=20)
		ax5.set_title('Signal', fontsize=20)
		ax5.set_xlabel(name, fontsize=20)
		ax5.set_ylabel('Measured Decay Length (mm)', fontsize=20)
    
		ax6.hist2d(X[:, i][y_predictions_proba[:,1] < clf_cut], X[:, uncVZi][y_predictions_proba[:,1] < clf_cut], range=[[param_min[i], param_max[i]],[param_min[uncVZi], param_max[uncVZi]]], bins=150, alpha=0.6, label="Identified as Background", cmin=0.5)
		ax7.hist2d(X[:, i][y_predictions_proba[:,1] > clf_cut], X[:, uncVZi][y_predictions_proba[:,1] > clf_cut], range=[[param_min[i], param_max[i]],[param_min[uncVZi], param_max[uncVZi]]], bins=150, alpha=0.6, label="Identified as Signal", cmin=0.5)
    
		ax6.set_title('Identified Background', fontsize=20)
		ax6.set_xlabel(name, fontsize=20)
		ax6.set_ylabel('Measured Decay Length (mm)', fontsize=20)
		ax7.set_title('Identified Signal', fontsize=20)
		ax7.set_xlabel(name, fontsize=20)
		ax7.set_ylabel('Measured Decay Length (mm)', fontsize=20)
    
		ax8.hist2d(y_predictions_proba[:,1][Y[:,0] == 0], X[:, i][Y[:,0] == 0], range=[[0,1],[param_min[i], param_max[i]]], bins=150, alpha=0.6, label="Background", cmin=0.5)
		ax9.hist2d(y_predictions_proba[:,1][Y[:,0] == 1], X[:, i][Y[:,0] == 1], range=[[0,1],[param_min[i], param_max[i]]], bins=150, alpha=0.6, label="Signal", cmin=0.5)
    
		ax8.set_title('Background', fontsize=20)
		ax8.set_xlabel('Classifier Output', fontsize=20)
		ax8.set_ylabel(name, fontsize=20)
		ax9.set_title('Signal', fontsize=20)
		ax9.set_xlabel('Classifier Output', fontsize=20)
		ax9.set_ylabel(name, fontsize=20)
    
		ax10.hist2d(y_predictions_proba[:,1][Y[:,0] == 0], X[:, i][Y[:,0] == 0], range=[[threshold_min,1],[param_min[i], param_max[i]]], bins=150, alpha=0.6, label="Background", cmin=0.5)
		ax11.hist2d(y_predictions_proba[:,1][Y[:,0] == 1], X[:, i][Y[:,0] == 1], range=[[threshold_min,1],[param_min[i], param_max[i]]], bins=150, alpha=0.6, label="Signal", cmin=0.5)
    
		ax10.set_title('Background', fontsize=20)
		ax10.set_xlabel('Classifier Output', fontsize=20)
		ax10.set_ylabel(name, fontsize=20)
		ax11.set_title('Signal', fontsize=20)
		ax11.set_xlabel('Classifier Output', fontsize=20)
		ax11.set_ylabel(name, fontsize=20)
    
		i = i + 1
		pp.savefig(fig)
		pp.savefig(fig2)
	pp.close()