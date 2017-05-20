#coding=utf8

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np
import os
import cPickle as pickle

def data_to_pkl(data, file_path):
    print "Saving data to file(%s). "%(file_path)

    with open(file_path, "w") as f:
        pickle.dump(data,f)
        return True

    print "Occur Error while saving..."
    return False

def load_PCA(data, path):
    pca = joblib.load(path)
    print pca.transform(data)

def test_PCA(data):
    print "PCA Start"
    pca = PCA(n_components = 2000)
    # data = [[1,2,3,4],[1,2,3,2],[1,3,4,1]]
    pca.fit(data)
    joblib.dump(pca, "pca_model.m")
    data_to_pkl(pca.transform(data), "lfw_pac_data")
    print "PCA End"


def deal_image(photo):
    img_main = Image.open(photo)
    # print "image size", img_main.size
    w, h = img_main.size
    img_matrix = np.matrix(img_main.convert("P").getdata())
    # img_matrix.resize(w, h)
    # print "photo size: ", img_matrix.shape
    # img_matrix = np.array(img_matrix)
    return np.array(img_matrix)

def pre_do_data():
    aim_dir = "../lfw/"
    ans_list = []
    cnt = 0
    flag = 0
    for filename_pre in os.listdir(aim_dir):
	if filename_pre[0] == '.':
		continue
        filename_pre_path = os.path.join(aim_dir, filename_pre)
    #    if filename_pre_path != "../lfw/Zico":
    #        continue
        for filename in os.listdir(filename_pre_path):
	    if filename[0] == '.':
		continue
            filename_path = os.path.join(filename_pre_path, filename)
            image_array = deal_image(filename_path)[0]
            cnt += 1
	    if cnt > 8000:
		flag = 1
		break
            if cnt % 10 == 0:
                print cnt
            ans_list.append(list(image_array))
	if flag == 1:
	    break
    print "All", cnt
    ans_list = ans_list
    test_PCA(ans_list)

if __name__ == '__main__':
    # test_PCA()
    ans_list = []
    ans_list.append(deal_image("../lfw/Zico/Zico_0003.jpg"))
    load_PCA(deal_image("../lfw/Zico/Zico_0003.jpg"), "./pca_model.m")
