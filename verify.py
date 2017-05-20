# coding=utf8

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np
import os
import cPickle as pickle
import algorithm

if __name__ == '__main__':
    with open("A_con.pkl", "rb") as fa:
        A = pickle.load(fa)
    with open("G_con.pkl", "rb") as fg:
        G = pickle.load(fg)
    image1 = "../lfw/Zico/Zico_0001.jpg"
    image2 = "../lfw/Zico/Zico_0002.jpg"
    pca = joblib.load("pca_model.m")
    print algorithm.Verify(A,G,pca.transform(deal_image(image1)), pca.transform(deal_image(image2)))
