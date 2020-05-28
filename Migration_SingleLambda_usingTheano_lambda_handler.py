import time
import numpy as np
from ml_inference.modeling import *
import pickle, csv, json
from PIL import Image
from PIL import ImageFile
import urllib3
from io import BytesIO
import base64


http = urllib3.PoolManager()
ImageFile.LOAD_TRUNCATED_IMAGES = True
def image_decoder(baseImg):
    x = base64.b64decode(baseImg) # base64 to bytes
    splited_x = x.split(b'\r\n') # split bytes
    data_decoded_bytes = splited_x[4]
    print(splited_x)
    try:
        data_decoded_bytes.decode('utf-8')
        data_decoded_bytes = base64.b64decode(data_decoded_bytes)
        print(data_decoded_bytes)
    except:
        print("need not decode base64.")
        print(data_decoded_bytes)
    return data_decoded_bytes

def reduce_v(pair):
    cache = {}
    for k, v in pair.items():
        if v not in cache.values():
            cache[k] = v
        if len(cache.values()) == 3:
            break
    return cache

def normalize_minmax(input_x):
    input_x = np.abs((input_x - np.min(input_x)) / (np.min(input_x) - np.max(input_x)))
    return input_x


model = NeuralNet('weights_dict.h5')

knn_model = pickle.load(open('knn_model.pkl','rb'))
def lambda_handler(event, context):
    img = event['body-json'] # mapp
    print(img)
    print(type(img))
    img_shape = event['params']['querystring']
    print(img_shape)
    start_t = time.time()
    img = image_decoder(img) # extract bytes image from a encoded base64.
    img = Image.open(BytesIO(img))
    img = img.resize((64, 64))
    img = np.array(img, np.float32)
    img = normalize_minmax(img)
    #img = np.transpose(img, (2,0,1))
    img = img.transpose(2,0,1)
    #img = np.array([img[2],img[1],img[0]]) # RGB -> BGR
    img = np.array([img])

    our = model.predict(img) #Inference using Deep Learning
    dist, ks = knn_model.kneighbors(our[0]) #Ranks using KNN

    trans_closest = []
    patchs = np.load("patch_trained.npy")
    for a in range(0, len(ks)):
        nearest_result = [k for k in ks[a]]
        buf = [int(patchs[i]) for i in nearest_result]
        #k1 = patchs[k1]
        #k2 = patchs[k2]
        #k3 = patchs[k3]
        trans_closest.append(buf)

    trans_closest = np.array(trans_closest)

    '''
    Search Clothes
    '''
    # using KNN
    clothes = test_funct(trans_closest) # no impl.
    clothes = reduce_v(clothes)
    pairs = {"clothes" : clothes}

    return pairs
