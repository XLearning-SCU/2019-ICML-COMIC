import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize

def load_data(data_name, is_normalize=True):
    X_list = []  # view set
    if data_name in ['noisemnist_3k','Caltech101-20', 'Caltech101-7', 'yale_mtv', 'LandUse_21', 'Texture_25', 'Scene_15']:
        data_dir = '../data/' + data_name + '.mat'
        mat = sio.loadmat(data_dir)
        Y = np.squeeze(mat['Y'])
        X = np.squeeze(mat['X'])
    elif data_name in ['MSRA', 'ORL_mtv', 'still_db', 'bbc_sport']:
        data_dir = '../data/zcq_data/' + data_name + '.mat'
        mat = sio.loadmat(data_dir)
        Y = np.squeeze(mat['gt'])
        X = np.squeeze(mat['X'])
    elif data_name in ['MNIST-USPS']:
        data_dir = '../data/' + 'MNIST' + '.mat'
        mat = sio.loadmat(data_dir)
        Y = np.squeeze(mat['label'])
        X = np.squeeze(mat['data'])
        MNIST_data = []
        for i in range(10):
            count = 0
            for j in range(Y.shape[0]):
                if i == Y[j]:
                    MNIST_data.append(X[j])
                    count += 1
                if count == 100:
                    break
        MNIST_data = np.array(MNIST_data).astype('float')
        print(MNIST_data.shape)
        data_dir = '../data/' + 'USPS' + '.mat'
        mat = sio.loadmat(data_dir)
        Y = np.squeeze(mat['labels'])
        X = np.squeeze(mat['DAT']).T
        USPS_data = []
        for i in range(0, 10):
            count = 0
            for j in range(Y.shape[0]):
                if i ==0 and 10 == Y[j]:
                    USPS_data.append(X[j])
                    count += 1
                elif i == Y[j]:
                    USPS_data.append(X[j])
                    count += 1
                if count == 100:
                    break
        USPS_data = np.array(USPS_data).astype('float')
        print(USPS_data.shape)
        X = [MNIST_data, USPS_data]

        Y = []
        for i in range(0,10):
            count = 0
            for j in range(100):
                Y.append(i)
                count +=1
        Y = np.array(Y)

        sio.savemat('MNIST-USPS.mat', {'X1': MNIST_data, 'X2': USPS_data, 'Y': Y})
    else:
        raise Exception('Wrong data name!')

    if is_normalize:
        if type(X) == list:
            view_size = len(X)
        else:
            view_size = X.shape[0]
        if Y.shape[0] == X[0].shape[0]:
            for view in range(view_size):
                X_list.append(normalize(X[view], norm='l2'))
        else:
            for view in range(view_size):
                X_list.append(normalize(X[view].T, norm='l2'))
    else:
        view_size = X.shape[0]
        if Y.shape[0] == X[0].shape[0]:
            for view in range(view_size):
                X_list.append(X[view])
        else:
            for view in range(view_size):
                X_list.append(X[view].T)

    if data_name in ['bbc_sport']:
        for view in range(view_size):
             X_list[view] = X_list[view].toarray()
             
    return X_list, Y

