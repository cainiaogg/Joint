#coding=utf-8
import sys
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
import cPickle as pickle
from sklearn.externals import joblib

def Verify(A, G, x1, x2):

    x1.shape = (-1,1)
    x2.shape = (-1,1)
    ratio = np.dot(np.dot(np.transpose(x1),A),x1) + np.dot(np.dot(np.transpose(x2),A),x2) - 2*np.dot(np.dot(np.transpose(x1),G),x2)
    return float(ratio)

def data_to_pkl(data, file_path):
    # 序列化 保存数据
    print "Saving data to file(%s). "%(file_path)

    with open(file_path, "w") as f:
        pickle.dump(data,f)
        return True

    print "Occur Error while saving..."
    return False

def JointBayesian_Train(trainingset, label, fold = "./"):
    if fold[-1] != '/':
        fold += '/'
    print trainingset.shape
    # the total num of image
    n_image = len(label)
    # the dim of features
    n_dim   = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    # the total people num
    n_class = len(classes)
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image + 1)
    maxNumberInOneClass = 0
    for i in range(n_class):
        # get the item of i
        cur[i] = trainingset[labels==i]
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]

        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    print "prepare done, maxNumberInOneClass=", maxNumberInOneClass

    u  = np.zeros([n_dim, n_class])
    ep = np.zeros([n_dim, withinCount])
    nowp=0
    for i in range(n_class):
        # the mean of cur[i]
        u[:,i] = np.mean(cur[i], 0)
        b = u[:,i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep[:, nowp:nowp+n_same_label] = cur[i].T-b
            nowp += n_same_label

    Su = np.cov(u.T,  rowvar=0)
    Sw = np.cov(ep.T, rowvar=0)
    oldSw = Sw
    SuFG  = {}
    SwG   = {}
    convergence = 1
    min_convergence = 1
    for l in range(500):
        F  = np.linalg.pinv(Sw)
        u  = np.zeros([n_dim, n_class])
        ep = np.zeros([n_dim, n_image])
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            if numberBuff[mi] == 1:
		#G = −(mS μ + S ε )−1*Su*Sw−1
                G = -np.dot(np.dot(np.linalg.pinv(mi*Su+Sw), Su), F)
		#Su*(F+mi*G) for u
                SuFG[mi] = np.dot(Su, (F+mi*G))
		#Sw*G for e
                SwG[mi]  = np.dot(Sw, G)
        for i in range(n_class):
            ##print l, i
            nn_class = cur[i].shape[0]
	    #formula 7 in suppl_760
            u[:,i] = np.sum(np.dot(SuFG[nn_class],cur[i].T), 1)
	    #formula 8 in suppl_760
            ep[:,nowp:nowp+nn_class] = cur[i].T+np.sum(np.dot(SwG[nn_class],cur[i].T),1).reshape(n_dim,1)
            nowp = nowp+nn_class

        Su = np.cov(u.T,  rowvar=0)
        Sw = np.cov(ep.T, rowvar=0)
        convergence = np.linalg.norm(Sw-oldSw)/np.linalg.norm(Sw)
        # print_info("Iterations-" + str(l) + ": "+ str(convergence))
        if convergence<1e-6:
            print "Convergence: ", l, convergence
            break;
        oldSw=Sw

        if convergence < min_convergence:
       	    min_convergence = convergence
            F = np.linalg.pinv(Sw)
            G = -np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su), F)
            A = np.linalg.pinv(Su+Sw)-(F+G)
            data_to_pkl(G, fold + "G.pkl")
            data_to_pkl(A, fold + "A.pkl")

    F = np.linalg.pinv(Sw)
    G = -np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su), F)
    A = np.linalg.pinv(Su+Sw) - (F+G)
    data_to_pkl(G, fold + "G_con.pkl")
    data_to_pkl(A, fold + "A_con.pkl")

    return A, G
