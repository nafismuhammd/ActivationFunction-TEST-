import csv
import numpy as np
import math
from sklearn.preprocessing import minmax_scale

def onehot_enc(lbl, min_val=0):
    mi = int(min(lbl))
    mx = int(max(lbl))
    enc = np.full((len(lbl), (mx - mi + 1)), min_val, np.int8)
    for i, x in enumerate(lbl):
        enc[i, int(x) - mi] = 1
    return enc

def onehot_dec(enc, mi=1):
    return [np.argmax(e) + mi for e in enc]

#Fungsi Aktivasi
def customSigm(X, alpha):
    return [1 / (1 + np.exp(-(alpha * x))) for x in X]

def customSigmDer(X, alpha):
    output = []
    for i, x in enumerate(X):
        s = sig([x])[0]
        output.append(alpha * s * (1 - s))
    return output

def arctan(X):
    return [2./math.pi * math.atan(x) for x in X]

def arctander(X):
    output = []
    for i, x in enumerate(X):
        hasil = (2/math.pi) * (1/(1+x**2))
        output.append(hasil)
    return output

def sig(X):
    return [1 / (1 + np.exp(-x)) for x in X]

def sigder(X):
    output = []
    for i, x in enumerate(X):
        s = sig([x])[0]
        output.append(s * (1 - s))
    return output

def adaptSlope(X, slope):
    return [1 / (1 + np.exp(-(slope * x))) for x in X]

def adaptiveSlopeder(X,slope):
    output = []
    for i, x in enumerate(X):
        a = adaptSlope(([x])[0],slope)
        output.append(a * (1 - a))
    return output

#Fungsi Pelatihan Backpropagation
def bp_train(X_new, target, layer_conf, aktivasi, max_epoch=1000, max_error=.1, learn_rate=.5, learn_rate_momentum = .5, print_per_epoch=100):
    #1. Inisialiasi Bobot dengan nguyen widrow
    scaleFactor = 0.7 * (len(X_new[0])**(1/len(X_new[0])))
    bobot = abs((np.array([np.random.rand(layer_conf[i] + 1, layer_conf[i + 1]) for i in range(len(layer_conf) - 1)]))-0.5)
    bobotawal1 = []
    bobotawal2 = []
    b = 0
    for i in range(len(bobot[0])):
        for j in range(len(bobot[0][i])):
            bobotawal1.append((bobot[0][i][j]/np.sqrt(sum(bobot[0][i]**2))))
        bobot[0][i]=[bobotawal1[b],bobotawal1[b+1],bobotawal1[b+2]]
        b+=3
    b=0
    for i in range(len(bobot[1])):
        for j in range(len(bobot[0][i])):
            bobotawal2.append((bobot[1][i][j]/np.sqrt(sum(bobot[1][i]**2))))
        bobot[1][i]=[bobotawal2[b],bobotawal2[b+1],bobotawal2[b+2]]
        b += 3

    d_bobot = [np.empty((layer_conf[i] + 1, layer_conf[i + 1])) for i in range(len(layer_conf) - 1)]
    d_prev_bobot = [np.empty((layer_conf[i] + 1, layer_conf[i + 1])) for i in range(len(layer_conf) - 1)]

    network_layer = [np.empty(j + 1) if i < len(layer_conf) - 1 else np.empty(j) for i, j in enumerate(layer_conf)]
    network_in = [np.empty(i) for i in layer_conf]
    d = [np.empty(s) for s in layer_conf[1:]]
    din = [np.empty(s) for s in layer_conf[1:-1]]

    epoch = 0
    mse = 1

    #Inisialisasi bias dengan nilai 1
    for i in range(0, len(network_layer)-1):
        network_layer[i][-1] = abs(np.random.random_sample()-scaleFactor)

   #2. selama kondisi berhenti belum tercapai, lakukan langkah 3-10
    while (max_epoch == -1 or epoch < max_epoch) and max_error < mse:
        epoch+=1
        mse=0

        #3. Untuk setiap data latih, lakukan langkah 4-9
        for data in range(len(X_new)):
            #Feedforward
            #4. Setiap neuron input menerima data latih x, dan meneruskan ke hidden layer
            network_layer[0][:-1] = X_new[data]

            #5. Setiap neuron hidden menjumlahkan nilai yang diterima, menghitung nilai aktivasinya, dan meneruskan ke output layer
            #6. Setiap neuron output menjumlahkan nilai yang diterima dan menghitung nilai aktivasinya
            for L in range(1, len(layer_conf)):
                network_in[L] = np.dot(network_layer[L-1], bobot[L-1])
                if (aktivasi == 'customSigm'):
                    network_layer[L][:len(network_in[L])] = customSigm(network_in[L], .1)
                elif(aktivasi == 'arctan'):
                    network_layer[L][:len(network_in[L])] = arctan(network_in[L])
                else:
                    network_layer[L][:len(network_in[L])] = sig(network_in[L])

            #Algoritma Backpropagation
            #7. Setiap neuron output menghitung nilai error (selisih) antara output y dengan target t:
            #Ket: network_layer[-1] adalah output layer
            e = target[data] - network_layer[-1]
            if(aktivasi == 'customSigm'):
                d[-1] = e * customSigmDer(network_in[-1], .1)
            elif (aktivasi == 'arctan'):
                d[-1] = e * arctander(network_in[-1])
            else:
                d[-1] = e * sigder(network_in[-1])
            if(epoch==1):
                d_bobot[-1] = learn_rate * d[-1] * network_layer[-2].reshape((-1,1))
            else:
                d_bobot[-1] = learn_rate * d[-1] * network_layer[-2].reshape((-1, 1)) + learn_rate_momentum * d_prev_bobot[-1]

            #8. Setiap neuron hidden menjumlahkan nilai d dari output layer
            for L in range(len(layer_conf) - 1, 1, -1):
                din[L - 2] = np.dot(d[L - 1], np.transpose(bobot[L - 1][:-1]))
                if(aktivasi=='customSigm'):
                    d[L-2] = din[L-2] * np.array(customSigmDer(network_in[L-1], learn_rate))
                elif(aktivasi=='arctan'):
                    d[L-2] = din[L-2] * np.array(arctander(network_in[L-1]))
                else:
                    d[L - 2] = din[L - 2] * np.array(sigder(network_in[L - 1]))
                if(epoch==1):
                    d_bobot[L - 2] = (learn_rate * d[L - 2]) * network_layer[L - 2].reshape((-1, 1))
                else:
                    d_bobot[L - 2] = (learn_rate * d[L - 2]) * network_layer[L - 2].reshape((-1, 1)) + learn_rate_momentum * d_prev_bobot[L-2]
            d_prev_bobot = d_bobot
            bobot += d_bobot
            mse += sum(e ** 2)

        mse /= len(X_new)
        if epoch % print_per_epoch == 0:
            print(f'Epoch {epoch}, MSE: {mse}')

    return bobot

def bp_slope_train(X, target, layer_conf, max_epoch=1000, max_error=.1, learn_rate=.1, learn_rate_momentum = .2, print_per_epoch=100):
    #1. Inisialiasi Bobot dengan Nguyen-Widrow
    scaleFactor = 0.7 * (len(X[0]) ** (1 / len(X[0])))
    bobot = abs((np.array([np.random.rand(layer_conf[i] + 1, layer_conf[i + 1]) for i in range(len(layer_conf) - 1)])) - 0.5)
    bobotawal1 = []
    bobotawal2 = []
    b = 0
    for i in range(len(bobot[0])):
        for j in range(len(bobot[0][i])):
            bobotawal1.append((bobot[0][i][j] / np.sqrt(sum(bobot[0][i] ** 2))))
        bobot[0][i] = [bobotawal1[b], bobotawal1[b + 1], bobotawal1[b + 2]]
        b += 3
    b = 0
    for i in range(len(bobot[1])):
        for j in range(len(bobot[0][i])):
            bobotawal2.append((bobot[1][i][j] / np.sqrt(sum(bobot[1][i] ** 2))))
        bobot[1][i] = [bobotawal2[b], bobotawal2[b + 1], bobotawal2[b + 2]]
        b += 3

    d_bobot = [np.empty((layer_conf[i] + 1, layer_conf[i + 1])) for i in range(len(layer_conf) - 1)]

    slope = [np.empty(i) for i in layer_conf[1:]]
    d_slope = [np.empty(i) for i in layer_conf[1:]]
    d_prev_bobot = [np.empty((layer_conf[i] + 1, layer_conf[i + 1])) for i in range(len(layer_conf) - 1)]
    network_layer = [np.empty(j + 1) if i < len(layer_conf) - 1 else np.empty(j) for i, j in enumerate(layer_conf)]
    network_in = [np.empty(i) for i in layer_conf]
    d = [np.empty(s) for s in layer_conf[1:]]
    epoch = 0
    mse = 1

    #Inisialisasi bias
    for i in range(0, len(network_layer)-1):
        network_layer[i][-1] = abs(np.random.random_sample()-scaleFactor)

   #2. selama kondisi berhenti belum tercapai, lakukan langkah 3-10
    while (max_epoch == -1 or epoch < max_epoch) and max_error < mse:
        epoch+=1
        mse=0
        #3. Untuk setiap data latih, lakukan langkah 4-9
        for data in range(len(X)):
            #Feedforward
            #4. Setiap neuron input menerima data latih x, dan meneruskan ke hidden layer
            network_layer[0][:-1] = X[data]

            #5. Setiap neuron hidden menjumlahkan nilai yang diterima, menghitung nilai aktivasinya, dan meneruskan ke output layer
            #6. Setiap neuron output menjumlahkan nilai yang diterima dan menghitung nilai aktivasinya
            for L in range(1, len(layer_conf)):
                network_in[L] = np.dot(network_layer[L - 1], bobot[L - 1])
                X1 = network_in[L] * slope[L-1]
                network_layer[L][:len(network_in[L])] = sig(X1)

            #Algoritma Backpropagation
            #7. Setiap neuron output menghitung nilai error (selisih) antara output y dengan target t:
            #Ket: network_layer[-1] adalah output layer
            e = target[data] - network_layer[-1]
            d[-1] = e * sigder(network_layer[-1])
            if(epoch==1):
                d_bobot[-1] = learn_rate * d[-1] * slope[L-1] * network_layer[-2].reshape((-1,1))
            else:
                d_bobot[-1] = learn_rate * d[-1] * slope[L-1] * network_layer[-2].reshape((-1, 1)) + learn_rate_momentum * d_prev_bobot[-1]
            
            #8. Setiap neuron hidden menjumlahkan nilai d dari output layer
            for L in range(len(layer_conf) - 1, 1, -1):
                d[L - 2] = np.dot(d[L - 1], np.transpose(bobot[L - 1][:-1]))*slope[L-2]*np.array(sigder(network_in[L-1]))
                if(epoch==1):
                    d_bobot[L - 2] = (learn_rate * d[L - 2]) * slope[L-2] * network_layer[L - 2].reshape((-1, 1))
                else:
                    d_bobot[L - 2] = (learn_rate * d[L - 2]) * slope[L-2] * network_layer[L - 2].reshape((-1, 1)) + learn_rate_momentum * d_prev_bobot[L-2]

            #update slope
            for L in range(1, len(layer_conf)):
                d_slope[L-1] = learn_rate * d[L - 1] * network_in[L]
            for L in range(len(layer_conf)-1,1,-1):
                d_slope[L-2] = learn_rate * d[L - 2] * network_in[L-1]
            d_prev_bobot = d_bobot
            slope += d_slope
            bobot += d_bobot
            mse += sum(e ** 2)

        mse /= len(X)
        if epoch % print_per_epoch == 0:
            print(f'Epoch {epoch}, MSE: {mse}')

    return bobot

#Fungsi Pengujian Backpropagation
def bp_test(X, w, aktivasi):
    #w[0] adalah Bobot w, w[1] adalah Bobot v
    #Inisialisasi neuron dan neuron input
    n = [np.empty(len(i)) for i in w]
    nin = [np.empty(len(i[0])) for i in w]

    #Hasil
    predict = []

    #Menambahkan bias
    n.append(np.empty(len(w[-1][0])))

    #Looping sebanyak data input
    for x in X:
        n[0][:-1] = x
        #setiap neuron output
        #hitung nilai input dan nilai aktivasi
        for L in range(0, len(w)):
            nin[L] = np.dot(n[L], w[L])
            if(aktivasi=='customSigm'):
                n[L+1][:len(nin[L])] = customSigm(nin[L], .1)
            elif(aktivasi=='arctan'):
                n[L + 1][:len(nin[L])] = arctan(nin[L])
            else:
                n[L + 1][:len(nin[L])] = sig(nin[L])

        predict.append(n[-1].copy())
    return predict

def bp_slope_test(X,w,slope):
    n = [np.empty(len(i)) for i in w]
    nin = [np.empty(len(i[0])) for i in w]
    predict = []
    n.append(np.empty(len(w[-1][0])))
    for x in X:
        n[0][:-1] = x
        for L in range(0, len(w)):
            nin[L] = np.dot(n[L], w[L])
            n[L+1][:len(nin[L])] = adaptSlope(nin[L],slope)
            
        predict.append(n[-1].copy())
    return predict

def akurasi(predict, target):
    sum = 0
    for i in range(len(predict)):
        if predict[i] == target[i]:
            sum +=1
    acc = sum / len(predict)
    return acc

#Main Method
def main():
    dataset = []
    target = []
    with open('dataset.csv', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in data:
            dataset.append(row)

    with open('target.csv') as csvfile:
        tar = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in tar:
            row1 = [str(r) for r in row]
            row2 = "".join(row1)
            target.append(int(row2))

    X = minmax_scale(dataset)
    X_new = np.array(X)
    datatrain = np.delete(X_new,[0,2,15,12,23,25],0)
    targettrain = np.delete(target,[0,2,15,12,23,25])
    datates = X[0],X[2],X[15],X[12],X[23],X[25]
    targettes = target[0],target[2],target[15],target[12],target[23],target[25]
    Y = onehot_enc(targettrain)
    layer_conf = (6,3,3)

    #Pilihan Aktivasi
    #Customized Sigmoid Function = customSigm
    #Adaptive Slope for Sigmoid = adaptSlope
    #Arctangent function = arctan
    #Sigmoid biner = nilai bebas
    alpha = 0
    for i in range(9):
        print('----------------Custom Sigmoid Function----------------')
        print('Pengujian ke-',i+1)
        alpha += .1
        print('Alpha :',alpha)
        bobotCustom = bp_train(datatrain, Y, layer_conf, 'customSigm', max_epoch=1000, max_error=.1, learn_rate=alpha, learn_rate_momentum=.3, print_per_epoch=50)
        predict_custom = bp_test(datates, bobotCustom, 'customSigm')
        predict_custom_train = bp_test(datatrain, bobotCustom, 'customSigm')
        predict = onehot_dec(predict_custom)
        predict_train = onehot_dec(predict_custom_train)
        acc = akurasi(predict, targettes)
        acc_train = akurasi(predict_train,targettrain)
        print('Prediksi', predict)
        print('Target', targettes)
        print('Akurasi', acc)
        print('Akurasi Data Latih', acc_train)
#    
    alpha = 0
    for i in range(9):
        print('\n----------------Adaptive Slope for Sigmoid Function----------------')
        print('Pengujian ke-',i+1)
        alpha +=.1
        print('Alpha :',alpha)
        bobotAdapt = bp_slope_train(datatrain, Y, layer_conf, max_epoch=1000, max_error=.1, learn_rate=alpha, learn_rate_momentum=.3, print_per_epoch=50)
        predict_Adapt = bp_slope_test(datates, bobotAdapt, .5)
        predict_Adapt_train = bp_slope_test(datatrain, bobotAdapt, .5)
        predict = onehot_dec(predict_Adapt)
        predict_train = onehot_dec(predict_Adapt_train)
        acc = akurasi(predict, targettes)
        acctrain = akurasi(predict_train, targettrain)
        print('Prediksi', predict)
        print('Target', targettes)
        print('Akurasi', acc)
        print('Akurasi Data Latih ',acctrain)
        
    alpha = 0
    for i in range(1):
        print('\n----------------Arctangent Function----------------')
        print('Pengujian ke-',i+1)
        alpha +=.1
        print('Alpha :',alpha)
        bobotArctan = bp_train(datatrain, Y, layer_conf, 'arctan', max_epoch=1000, max_error=.1, learn_rate=alpha, learn_rate_momentum=.3, print_per_epoch=50)
        predict_arctan = bp_test(datates, bobotArctan, 'arctan')
        predict_arctan_train = bp_test(datatrain, bobotArctan, 'arctan')
        predict = onehot_dec(predict_arctan)
        predict_train = onehot_dec(predict_arctan_train)
        acc = akurasi(predict, targettes)
        acc_train = akurasi(predict_train,targettrain)
        print('Prediksi', predict)
        print('Target', target)
        print('Akurasi', acc)
        print('Akurasi Data Latih', acc_train)
       
    alpha = 0
    for i in range(1):
        print('\n----------------BPNN Momentum Regular----------------')
        print('Pengujian ke-',i+1)
        alpha +=.1
        print('Alpha :',alpha)
        bobot = bp_train(datatrain, Y, layer_conf, 'sigm', max_epoch=1000, max_error=.1, learn_rate=alpha, learn_rate_momentum=.3, print_per_epoch=50)
        predict_test = bp_test(datates, bobot, "sigm")
        predict_latih = bp_test(datatrain, bobot, "sigm")
        predict = onehot_dec(predict_test)
        predicttrain = onehot_dec(predict_latih)
        acc = akurasi(predict, targettes)
        acctrain = akurasi(predicttrain, targettrain)
        print('Prediksi', predict)
        print('Target', targettes)
        print('Akurasi', acc)
        print('Akurasi Data Latih ',acctrain)

main()
