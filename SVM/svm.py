"""
Support Vector Machine(SVM) 
Without using sklearn package, the performance of this version is a little bit worse and the process time is quite long
"""

from numpy import *
import common
"""
Variable Definition

train_num: training set size
test_num: testing set size
fig_w: figure width
epochs: number of epoches
common_path = common path of input data
"""


train_num = 60000
test_num = 10000
fig_w = 28
epochs = 50
common_path = "../mnist"

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn  					#feature
        self.labelMat = classLabels 			#class
        self.C = C                              #C
        self.tol = toler 						#thresthold
        self.m = shape(dataMatIn)[0] 			#data row
        self.alphas = mat(zeros((self.m,1)))    #alpha
        self.b = 0 				                #b
        self.eCache = mat(zeros((self.m,2))) 	#cache
        self.K = mat(zeros((self.m,self.m))) 	#kernel result
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup): 
    """ kernel fucntion
    X: feature tree of support vector 
    A:feature value of each row
    kTup:('lin',k1) kernel type and parameter
    """
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': #linear
        K = X * A.T
    elif kTup[0]=='rbf': # radial bias function
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Kernel is not recognized')
    return K


def SMO(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): 
    """
    use SMO algorithm to compute alpha
    """
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): #trival all data 
                alphaPairsChanged += innerL(i,oS)
                #print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) 		#show info during iteration
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: #trival no bound data
                alphaPairsChanged += innerL(i,oS)
                #print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))	#show info during iteration
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas


def innerL(i, oS): 
    """
    test whether ai satisfies KKT condition, if not, randomly choose aj to optimize, update ai,aj and b 
    """
    Ei = calcEk(oS, i) 		#compute E
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):	#test whether statisfy KKT
        j,Ej = selectJ(i, oS, Ei) 				#randomly choose aj
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):			 #p126
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            #print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] 	#p127 fomula  7.107
        if eta >= 0:
            #print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta 	#p127 fomula  7.106
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)	 	#p127 fomula  7.108
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): 	#alpha threshold
            #print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])	#p127 fomula  7.109
        updateEk(oS, i) #update 

        #p129 fomula  7.114-7.116
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def calcEk(oS, k): 
    """
    compute Ek (p127 fomula  7.105)
    """
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei): 
    """
    randomly choose Ei
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): #return max step Ej
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k): 
    """
    update Ek
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def selectJrand(i,m): 
    """
    randomly choose a j in (0-m) not equal to i
    """
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  
    """
    keep (L <= a <= H)
    """
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj


def SVM(train_x, test_x, train_y, test_y, kTup):
    """ SVM
    :param path: data file
    :return: data(np.array)
    """
    #get support vector
	b,alphas = SMO(train_x, train_y, 200, 0.0001, epochs, kTup)
	datMat=mat(dataArr)
	labelMat = mat(train_y).transpose()
	svInd=nonzero(alphas)[0] 
	sVs=datMat[svInd]
	labelSV = labelMat[svInd] 
	print("there are %d Support Vectors" % shape(sVs)[0]) 
	m,n = shape(datMat) 
	errorCount = 0
	for i in range(m):
	    kernelEval = kernelTrans(sVs,datMat[i,:],kTup) #change support vectors into kernel 
	    predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b  #p133
	    if sign(predict)!=sign(labelArr[i]): #-1 if x < 0, 0 if x==0, 1 if x > 0
	        errorCount += 1
	print("the training error rate is: %f" % (float(errorCount)/m))

    #testing
	errorCount_test = 0
	datMat_test=mat(test_x)
	labelMat = mat(test_y).transpose()
	m,n = shape(datMat_test)
	for i in range(m): 				#test error
		kernelEval = kernelTrans(sVs,datMat_test[i,:],kTup)
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict)!=sign(test_y[i]):
			errorCount_test += 1
	print("the test error rate is: %f" % (float(errorCount_test)/m))

def main():
    data = common.Data(common_path+"/mnist_train/train_data.npy", common_path+"/mnist_train/mnist_train_label",
                    common_path+"/mnist_test/test_data.npy", common_path+"/mnist_test/mnist_test_label", fig)
    train_x = data.train_x
    test_x = data.test_x
    train_y = data.train_y
    test_y = data.test_y
    SVM(train_x, test_x, train_y, test_y,('rbf',1.3))

main()
