import numpy as np
from array import array
import ROOT
from ROOT import gROOT, TCanvas, TF1, TFile, gStyle, TFormula, TGraph, TGraphErrors, TH1F, TCutG, TH2D, gDirectory, TLegend, TNtuple, TTree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm


def openPDF(outfile,canvas):
	canvas.Print(outfile+".pdf[")

def closePDF(outfile,canvas):
	canvas.Print(outfile+".pdf]")

def ctau(mass,eps):
	hbar_c = 1.973e-13
	return  3 * hbar_c/ (1 * mass * 1/137. * eps**2)

def DrawZHisto(events,histo,nBins,minX,maxX):
	events.Draw("{0}>>{1}({2},{3},{4})".format("triEndZ",histo,nBins,minX,maxX))
	histo = ROOT.gROOT.FindObject(histo)
	return histo

def shapeHisto(gammact,nBins,targetz,maxZ):
	histo = TH1F("histo","histo",nBins,targetz,maxZ)
	zbin = (maxZ - targetz) / nBins
	for i in range(nBins):
		ibin = i + 1
		z = targetz + (i + 0.5) * zbin
		histo.Fill(z,np.exp((targetz - z)/gammact) / gammact)
	histo.Scale(1./histo.Integral())
	return histo

def multiplyHisto(histo1, histo2, n):
	histo = histo1.Clone("histo")
	histo.Multiply(histo2)
	histo.Scale(float(n)/histo.Integral())
	return histo

def divideHisto(histo1, histo2):
	histo = histo1.Clone("histo")
	histo.Divide(histo2)
	return histo

input_truth = TFile("ap_100MeV_truth.root")
input_sig = TFile("ap_100MeV_L1L1_cleanup_nPos_1.root")

events = input_sig.Get("ntuple")
events_truth = input_truth.Get("ntuple")

gStyle.SetOptStat(0)
c = TCanvas("c","c",800,600);
outfile = "test"

nBins = 70
zTarg = -4.3
maxZ = 70 + zTarg

eps = np.sqrt(10**-9.)
np.random.seed(1)
#eps = np.sqrt(4*10**-10.)
#np.random.seed(2)
#eps = np.sqrt(4*10**-9.)
#np.random.seed(3)

mass = 0.100
eBeam = 2.3
avg_gamma = 0.95
n_samples = 100000

gamma = avg_gamma * eBeam / mass
ct = ctau(mass,eps)
gammact = gamma * ct

#openPDF(outfile,c)
recon_shape = DrawZHisto(events,"recon_shape",nBins,zTarg,maxZ)
#recon_shape.Draw("histp")
#c.Print(outfile+".pdf")
input_shape = DrawZHisto(events_truth,"input_shape",nBins,zTarg,maxZ)
#input_shape.Draw("histp")
#c.Print(outfile+".pdf")
truth_shape = shapeHisto(gammact,nBins,zTarg,maxZ)
#truth_shape.Draw("histp")
#c.Print(outfile+".pdf")
acceptance_shape = divideHisto(recon_shape,input_shape)
acceptance_shape.Scale(1./acceptance_shape.Integral())
#acceptance_shape.Draw("histp")
#c.Print(outfile+".pdf")
signal_shape = multiplyHisto(truth_shape,acceptance_shape,n_samples)
#signal_shape.Draw("histp")
#c.Print(outfile+".pdf")
reweight = divideHisto(signal_shape,recon_shape)
#reweight = divideHisto(truth_shape,input_shape)
#reweight.Draw("histp")
#c.Print(outfile+".pdf")

#closePDF(outfile,c)


gROOT.Reset()

rootfile = TFile(outfile + ".root","RECREATE")

tree = TTree( 'tree', 'tree' )

triEndZ = array('d',[0])
uncVX = array('d',[0])
uncVY = array('d',[0])
uncVZ = array('d',[0])
uncP = array('d',[0])
uncChisq = array('d',[0])
uncM = array('d',[0])
uncTargProjX = array('d',[0])
uncTargProjY = array('d',[0])
uncTargProjXErr = array('d',[0])
uncTargProjYErr = array('d',[0])
uncCovXX = array('d',[0])
uncCovYY = array('d',[0])
uncCovZZ = array('d',[0])
bscChisq = array('d',[0])
tarChisq = array('d',[0])
eleP = array('d',[0])
eleTrkChisq = array('d',[0])
eleTrkHits = array('d',[0])
eleTrkD0 = array('d',[0])
eleTrkLambda = array('d',[0])
eleTrkZ0 = array('d',[0])
eleTrkD0Err = array('d',[0])
eleTrkLambdaErr = array('d',[0])
eleTrkZ0Err = array('d',[0])
posP = array('d',[0])
posTrkChisq = array('d',[0])
posTrkHits = array('d',[0])
posTrkD0 = array('d',[0])
posTrkLambda = array('d',[0])
posTrkZ0 = array('d',[0])
posTrkD0Err = array('d',[0])
posTrkLambdaErr = array('d',[0])
posTrkZ0Err = array('d',[0])

tree.Branch("triEndZ",triEndZ,"triEndZ/D")
tree.Branch("uncVX",uncVX,"uncVX/D")
tree.Branch("uncVX",uncVX,"uncVX/D")
tree.Branch("uncVY",uncVY,"uncVY/D")
tree.Branch("uncVZ",uncVZ,"uncVZ/D")
tree.Branch("uncP",uncP,"uncP/D")
tree.Branch("uncChisq",uncChisq,"uncChisq/D")
tree.Branch("uncM",uncM,"uncM/D")
tree.Branch("uncTargProjX",uncTargProjX,"uncTargProjX/D")
tree.Branch("uncTargProjY",uncTargProjY,"uncTargProjY/D")
tree.Branch("uncTargProjXErr",uncTargProjXErr,"uncTargProjXErr/D")
tree.Branch("uncTargProjYErr",uncTargProjYErr,"uncTargProjYErr/D")
tree.Branch("uncCovXX",uncCovXX,"uncCovXX/D")
tree.Branch("uncCovYY",uncCovYY,"uncCovYY/D")
tree.Branch("uncCovZZ",uncCovZZ,"uncCovZZ/D")
tree.Branch("bscChisq",bscChisq,"bscChisq/D")
tree.Branch("tarChisq",tarChisq,"tarChisq/D")
tree.Branch("eleP",eleP,"eleP/D")
tree.Branch("eleTrkChisq",eleTrkChisq,"eleTrkChisq/D")
tree.Branch("eleTrkHits",eleTrkHits,"eleTrkHits/D")
tree.Branch("eleTrkD0",eleTrkD0,"eleTrkD0/D")
tree.Branch("eleTrkLambda",eleTrkLambda,"eleTrkLambda/D")
tree.Branch("eleTrkZ0",eleTrkZ0,"eleTrkZ0/D")
tree.Branch("eleTrkD0Err",eleTrkD0Err,"eleTrkD0Err/D")
tree.Branch("eleTrkLambdaErr",eleTrkLambdaErr,"eleTrkLambdaErr/D")
tree.Branch("eleTrkZ0Err",eleTrkZ0Err,"eleTrkZ0Err/D")
tree.Branch("posP",posP,"posP/D")
tree.Branch("posTrkChisq",posTrkChisq,"posTrkChisq/D")
tree.Branch("posTrkHits",posTrkHits,"posTrkHits/D")
tree.Branch("posTrkD0",posTrkD0,"posTrkD0/D")
tree.Branch("posTrkLambda",posTrkLambda,"posTrkLambda/D")
tree.Branch("posTrkZ0",posTrkZ0,"posTrkZ0/D")
tree.Branch("posTrkD0Err",posTrkD0Err,"posTrkD0Err/D")
tree.Branch("posTrkLambdaErr",posTrkLambdaErr,"posTrkLambdaErr/D")
tree.Branch("posTrkZ0Err",posTrkZ0Err,"posTrkZ0Err/D")

for entry in xrange(events.GetEntries()):
	events.GetEntry(entry)
	triEndZ[0] = events.triEndZ
	uncVX[0] = events.uncVX
	uncVY[0] = events.uncVY
	uncVZ[0] = events.uncVZ
	uncP[0] = events.uncP
	uncChisq[0] = events.uncChisq
	uncM[0] = events.uncM
	uncTargProjX[0] = events.uncTargProjX
	uncTargProjY[0] = events.uncTargProjY
	uncTargProjXErr[0] = events.uncTargProjXErr
	uncTargProjYErr[0] = events.uncTargProjYErr
	uncCovXX[0] = events.uncCovXX
	uncCovYY[0] = events.uncCovYY
	uncCovZZ[0] = events.uncCovZZ
	bscChisq[0] = events.bscChisq
	tarChisq[0] = events.tarChisq
	eleP[0] = events.eleP
	eleTrkChisq[0] = events.eleTrkChisq
	eleTrkHits[0] = events.eleTrkHits
	eleTrkD0[0] = events.eleTrkD0
	eleTrkLambda[0] = events.eleTrkLambda
	eleTrkZ0[0] = events.eleTrkZ0
	eleTrkD0Err[0] = events.eleTrkD0Err
	eleTrkLambdaErr[0] = events.eleTrkLambdaErr
	eleTrkZ0Err[0] = events.eleTrkZ0Err
	posP[0] = events.posP
	posTrkChisq[0] = events.posTrkChisq
	posTrkD0[0] = events.posTrkD0
	posTrkLambda[0] = events.posTrkLambda
	posTrkZ0[0] = events.posTrkZ0
	posTrkD0Err[0] = events.posTrkD0Err
	posTrkLambdaErr[0] = events.posTrkLambdaErr
	posTrkZ0Err[0] = events.posTrkZ0Err
	zbin = reweight.FindBin(events.triEndZ)
	copy = reweight.GetBinContent(zbin)
	for i in range(int(copy)):
		tree.Fill()
	if((copy - int(copy)) > np.random.random()):
		tree.Fill()

rootfile.Write()
rootfile.Close()