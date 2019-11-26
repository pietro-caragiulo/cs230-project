def getPlot(string):
    arr = string.split(" ")
    return arr[0]

def getMin(string):
    arr = string.split(" ")
    if(len(arr) < 2): return -9999
    else: return float(arr[1])

def getMax(string):
    arr = string.split(" ")
    if(len(arr) < 3): return -9999
    else: return float(arr[2])

def getParameters():    
	param_names = []
	param_names.append("uncVX -1 1")
	param_names.append("uncVY -1 1")
	param_names.append("uncVZ 0 50")
	param_names.append("uncVX -1 1")
	param_names.append("uncVY -1 1")
	param_names.append("uncVZ 0 50")
	param_names.append("eleTrkZ0 -5 5")
	param_names.append("posTrkZ0 -5 5")
	param_names.append("eleTrkZ0Err 0 1")
	param_names.append("posTrkZ0Err 0 1")
	param_names.append("sqrt(uncCovXX) 0 0.3")
	param_names.append("sqrt(uncCovYY) 0 0.3")
	param_names.append("sqrt(uncCovZZ) 0 10")
	param_names.append("eleTrkD0 -5 5")
	param_names.append("posTrkD0 -5 5")
	param_names.append("eleTrkD0Err 0 1")
	param_names.append("posTrkD0Err 0 1")
	param_names.append("eleTrkLambda -0.1 0.1")
	param_names.append("posTrkLambda -0.1 0.1")
	param_names.append("eleTrkLambdaErr 0 0.01")
	param_names.append("posTrkLambdaErr 0 0.01")
	param_names.append("uncM 0.045 0.055")
	param_names.append("eleTrkChisq 0 40")
	param_names.append("posTrkChisq 0 40")
	param_names.append("bscChisq 0 10")
	param_names.append("uncChisq 0 10")
	param_names.append("tarChisq 0 10")
	param_names.append("uncTargProjX -1 1")
	param_names.append("uncTargProjY -1 1")
	param_names.append("truthZ 0 70")

	branch_names = []
	param_min = []
	param_max = []
	for i in range(len(param_names)):
		branch_names.append(getPlot(param_names[i]))
		param_min.append(getMin(param_names[i]))
		param_max.append(getMax(param_names[i]))
	param_list = branch_names

	return param_list, param_min, param_max