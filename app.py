from vm_print import console, b_c, vm_proc_print
import sys
import os
from PIL import Image
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from  keras.models import Sequential
from  keras.layers import Dense
import numpy as np
from vm_print import console, b_c, vm_proc_print
# console()
def make_train_matr(p_:str):
  try:
    matr=np.zeros(shape=(4, 10000))
    data=None
    img=None
    for i in os.listdir(p_):
        ful_p=os.path.join(p_,i)
        img=Image.open(ful_p)
        print("img", ful_p)
        data=list(img.getdata())
        for row in range(4):
            for elem in range(10000):
                matr[row][elem] =\
                    data[elem]
  except Exception as e:
      print(e.with_traceback(None))
      vm_proc_print(b_c, locals(), globals())

  return matr

def create_nn():
    model = Sequential()
    model.add(Dense(70, input_dim=10000, activation='sigmoid'))
    model.add(Dense(70, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def fit_nn(X,Y):
    new_img=None
    nn=create_nn()
    es=EarlyStopping(monitor='val_accuracy')
    opt=SGD(lr=0.07)
    # opt='adam'
    nn.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    nn.fit(X, Y, epochs=512, validation_split=0.25)
           #, callbacks=[es])
    # vm_proc_print(b_c, locals(), globals())
    return nn


def calc_out_nn(l_:list):
    l_tested=[0] * 10000
    for i in range(len(l_)):
        val=round(l_[i],1)
        if val > 0.5 :
            l_tested[i] = 0
        else:
            l_tested[i] = 1
    return l_tested

def make_2d_arr(_1d_arr:list):
    matr_make=np.zeros(shape=(100,100))
    for i in range(100):
        for j in range(100):
            matr_make[i][j] = _1d_arr[i * 100 + j]
    return matr_make
def pred(X, nn):
      p_matr=nn.predict(np.array([X]))
      p_vec=p_matr[0]
      p_tested=calc_out_nn(p_vec)
      p_2d_img=make_2d_arr(p_tested)
      new_img=Image.fromarray(np.uint8(p_2d_img))
      new_img.save("ker.png")


def main():
    X = make_train_matr('b:/out')
    Y = np.zeros(shape=(4,10000))
    # Y[0][0]=1
    # Y[1][0]=1
    # Y[2][0]=1
    # Y[3][0]=1
    Y=np.array([[1], [1], [1], [1]])
    nn=fit_nn(X, Y)
    scores=nn.evaluate(X, Y, verbose=2)
    print("scores",scores)
    # single_vec=np.array(Y[0])
    single_vec = np.random.randn(10000)
    pred(single_vec, nn)
    nn_pred=nn.predict(np.array([X[0]]))

main()

