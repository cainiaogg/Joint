# coding=utf8

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np
import os
import cPickle as pickle
import algorithm

if __name__ == '__main__':
    A = pickle.load(open("A.pkl", "w"))
    G = pickle.load(open("G.pkl", "w"))
    image1 = "../lfw/Zico/Zico_0001.jpg"
    image2 = "../lfw/Zico/Zico_0002.jpg"
    print algorithm.Verify(A,G,pca.transform(deal_image(image1)), pca.transform(deal_image(image2)))
