import numpy as np
import matplotlib.pyplot as plt
import math
import os
import obspy
from obspy.taup import TauPyModel
model_s=TauPyModel(model='iasp91')
import csv
from obspy import read
from pathlib import Path
import keras
from keras import Model
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import UpSampling1D, Conv1D, ZeroPadding1D, Reshape, Concatenate, Dropout , MaxPooling1D, Flatten
from keras.layers import Dense, Input, LeakyReLU, InputLayer
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint


in_data, out_data, nev, info = [], [], [], []
with open('../data/test.list') as l:
        for row in l:
                nst=str(row.split()[0])
                neq=str(row.split()[1])
                stlat=float(row.split()[2])
                stlon=float(row.split()[3])
                evlat=float(row.split()[9])
                evlon=float(row.split()[10])
                evdep=float(row.split()[11])
                AB=str(row.split()[12])

                print(nst, neq, stlat, stlon, evlat, evlon)

                S_arrival=-1

                shift=0

                tr_z, tr_n, tr_e, tr_out = [], [], [], []

                nz='../data/SWS_trace/'+nst+'/'+neq+'/'+nst+'.z'
                nn='../data/SWS_trace/'+nst+'/'+neq+'/'+nst+'.n'
                ne='../data/SWS_trace/'+nst+'/'+neq+'/'+nst+'.e'

                st=read(nz)

                #time of f, end of time window
                f=st[0].stats.sac['f']
                f_tr=int((f-st[0].stats.sac['b'])*100)

                #theorical S arrival time
                if S_arrival==-1: S_arrival=model_s.get_travel_times(source_depth_in_km=st[0].stats.sac['evdp'],
                                                                distance_in_degree=st[0].stats.sac['gcarc'],
                                                                phase_list=["s"])[0].time
                S_arr=int((st[0].stats.sac['o']+S_arrival-st[0].stats.sac['b'])*100)+shift

                #read 200 data points (2 s) before and after theorical S arrival time
                nb=0
                if S_arr<200:   #if data point is not enough before theorical S arrival time
                        nb=200-S_arr
                        tr_z=np.zeros(nb)
                        tr_z=np.append(tr_z,st[0].data[S_arr-200+nb:S_arr+200])
                else:
                        tr_z=st[0].data[S_arr-200:S_arr+200]

                for i in range(len(tr_z),400):  #if have enough data points
                        tr_z=np.append(tr_z,[0])

                #NS component
                st=read(nn)
                if S_arr<200:
                        nb=200-S_arr
                        tr_n=np.zeros(nb)
                        tr_n=np.append(tr_n,st[0].data[S_arr-200+nb:S_arr+200])
                else:
                        tr_n=st[0].data[S_arr-200:S_arr+200]

                for i in range(len(tr_n),400):
                        tr_n=np.append(tr_n,[0])

                #EW component
                st=read(ne)
                if S_arr<200:
                        nb=200-S_arr
                        tr_e=np.zeros(nb)
                        tr_e=np.append(tr_e,st[0].data[S_arr-200+nb:S_arr+200])
                else:
                        tr_e=st[0].data[S_arr-200:S_arr+200]

                for i in range(len(tr_e),400):
                        tr_e=np.append(tr_e,[0])

                #normalize
                max_A=max(max(abs(tr_z)), max(abs(tr_n)), max(abs(tr_e)))
                tr_z=tr_z/max_A
                tr_n=tr_n/max_A
                tr_e=tr_e/max_A

                #target
                tr_out=np.zeros(400)
                for i in range(21):
                        tr_out[f_tr+i-10-S_arr+200]=math.exp(-(i-10)**2/(2*(10/3)**2))

                tr, tr_o = [], []
                for i in range(len(tr_z)):
                        tr.append(np.array([tr_z[i],tr_n[i],tr_e[i]]))
                        tr_o.append([tr_out[i]])

                in_data.append(tr)
                out_data.append(tr_o)
                nev.append(nst+'_'+neq)
                info.append([st[0].stats.sac['b'], st[0].stats.sac['o'], st[0].stats.sac['f'], S_arrival, shift])

print(np.array(in_data).shape, np.array(out_data).shape, np.array(nev).shape, np.array(info).shape)

in_data=np.array(in_data)
out_data=np.array(out_data)


#CNN model
input_shape=(400,3)

model=Sequential()
#1 400 to 200
model.add(Conv1D(kernel_size=(3), filters=64,
                 input_shape=input_shape,
                 strides=(2),
                 padding='same',))
model.add(LeakyReLU(alpha=0.05))

#2 200 to 100
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(2),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#3 100 to 50
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(2),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#4 50 to 25
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(2),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#5 25 to 13
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(2),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#6 13 to 7
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(2),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#7 7 to 13
model.add(UpSampling1D(size=2))
model.add(ZeroPadding1D(padding = (0,1)))
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(1)))
model.add(LeakyReLU(alpha=0.05))

#8 13 to 25
model.add(UpSampling1D(size=2))
model.add(ZeroPadding1D(padding = (0,1)))
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(1)))
model.add(LeakyReLU(alpha=0.05))

#9 25 to 50
model.add(UpSampling1D(size=2))
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(1),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#10 50 to 100
model.add(UpSampling1D(size=2))
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(1),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#11 100 to 200
model.add(UpSampling1D(size=2))
model.add(Conv1D(kernel_size=(3), filters=64,
                 strides=(1),
                 padding='same'))
model.add(LeakyReLU(alpha=0.05))

#12 200 to 400
model.add(UpSampling1D(size=2))
model.add(Conv1D(kernel_size=(3), filters=1,
                 strides=(1),
                 padding='same',
                 activation='sigmoid'))

model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

nmodel='../model/model_paper.h5'
model.load_weights(nmodel)

results=model.predict(in_data)

for i in range(len(results)):
        if i%100==0: print(i)
        a=str(i)
        nrout_in='Outp/'+nev[i]+'_'+a.zfill(6)+'.in'
        nrout_out='Outp/'+nev[i]+'_'+a.zfill(6)+'.out'
        nrout_info='Outp/'+nev[i]+'_'+a.zfill(6)+'.info'
        nrout_res='Outp/'+nev[i]+'_'+a.zfill(6)+'.res'

        np.savetxt(nrout_in, in_data[i])
        np.savetxt(nrout_out, out_data[i])
        np.savetxt(nrout_info, info[i])
        np.savetxt(nrout_res, results[i])
