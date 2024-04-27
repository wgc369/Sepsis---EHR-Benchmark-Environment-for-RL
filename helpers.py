import numpy as np
import tensorflow as tf
import csv 
import pickle
import pandas as pd

def read_data(filename):
    df = pd.read_csv(filename, dtype='float64')
    fieldnames = list(df.columns)
    data = np.array(df.to_dict(orient='records'))
    return data, fieldnames

def get_features_and_labels(filename):
    data, fieldnames = read_data(filename)
    all_features = fieldnames[3:7] + fieldnames[11:50] + [fieldnames[56]] + fieldnames[57:59] + fieldnames[50:52] + fieldnames[59:61]
    labels = fieldnames[3:7] + fieldnames[11:50] + [fieldnames[56]] + fieldnames[57:59]
    _x = []
    _y = []
    for i in range(len(data)-1):
        if data[i]['icustayid'] != data[i+1]['icustayid']:
            continue
 
        _x.append([data[i][k] for k in all_features])
        _y.append([data[i+1][k] for k in labels])
    return np.array(_x), np.array(_y)

class EHRGenerator:
    def __init__(self, model='lstm', sofa=5):
        # models available : lstm, rnn, gru, stacked_dense_layers, transformer
        # sofa scores available: 1 to 19
        if model not in ['lstm', 'gru', 'rnn', 'stacked_dense_layers', 'transformer']:
            raise ValueError(f'Unknown model "{model}"!')
        if sofa < 1 or sofa > 22:
            raise ValueError(f'Sofa score out of range "{sofa}"! Must be within [1, 22].')
        
        self.model_name = model
        self.initial_states = self._initialize_initial_states()
        self.sofa = sofa
        with open('scalers/scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open('scalers/scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)


        
        if model == 'tabnet':
            pass
            # self.model = torch.load('saved_models/tabnet.pt')
        else:
            self.model = tf.keras.models.load_model(f'saved_models/{model}')

    def _initialize_initial_states(self):
        pass

    def _pre_process(self, sap):
        # sap_scaled = np.array(sap)
        # sap_scaled = sap_scaled.reshape(-1, 1)
        sap_scaled = [sap]
        sap_scaled = self.scaler_X.transform(sap_scaled)
        # sap_scaled = sap_scaled.reshape(-1)
        return sap_scaled

    def _post_process(self, next_state, curr_state):
        ns = self.scaler_y.inverse_transform(next_state)[0]
        ns[44] = self._get_sofa_score(ns)
        ns[:4] = curr_state[:4]
        return ns

    def _get_sofa_score(self, next_state):
        pao2_fio2 = next_state[42]
        mechvent = next_state[40]
        # fio2 = next_state[13]
        platelets = next_state[30]
        gcs = next_state[5]
        bili = next_state[26]
        map = (2 * next_state[9] + next_state[7]) / 3
        creatinine = next_state[19]
        score = 0

        if gcs < 6:
            score += 4
        elif gcs < 10:
            score += 3
        elif gcs < 13:
            score += 2
        elif gcs < 15:
            score += 1

        if map < 70:
            score += 1
        
        if pao2_fio2 < 100 and mechvent == 1:
            score += 4
        elif pao2_fio2 < 200 and mechvent == 1:
            score += 3
        elif pao2_fio2 < 200 and mechvent == 0:
            score += 2
        elif pao2_fio2 < 300:
            score += 2
        elif pao2_fio2 < 400:
            score += 1

        if platelets < 20:
            score += 4
        elif platelets < 50:
            score += 3
        elif platelets < 100:
            score += 2
        elif platelets < 150:
            score += 1

        if bili < 1.2:
            score += 0
        elif bili < 2:
            score += 1
        elif bili < 6:
            score += 2
        elif bili < 12:
            score += 3
        elif bili >= 12:
            score += 4

        if creatinine < 1.2:
            score += 0
        elif creatinine < 2:
            score += 1
        elif creatinine < 3.5:
            score += 2
        elif creatinine < 5:
            score += 3
        elif creatinine >= 5:
            score += 4
        return score

    def get_next_state(self, curr_state, action):

        # raise Exception(curr_state, action)
        # print(curr_state)
        # print()
        # print(action)
        # raise(curr_state, action)
        sap = np.concatenate((curr_state, action), axis=None)
        sap = self._pre_process(sap)
        _temp_vals_reshaped = sap.reshape((sap.shape[0], 1, sap.shape[1]))
        if self.model_name in ('lstm', 'gru', 'rnn'):
            next_state = self.model.predict([sap[:, :4], _temp_vals_reshaped])
        elif self.model_name == 'stacked_dense_layers':
            next_state = self.model.predict([sap[:, :4], sap])
        elif self.model_name == 'transformer':
            next_state = self.model.predict(_temp_vals_reshaped)
        else:
            next_state = self.model.predict(sap)
        next_state = self._post_process(next_state, curr_state)
        next_state = np.array(next_state)
        # print(next_state)
        return next_state
        
    def get_initial_state(self):
        # initial_state = ['0.0000000000', '17639.8264351852', '0.0000000000', '0.0000000000', '78.6999969482', '15.0000000000', '74.5714285714', '104.7142857143', '72.8571428571', '56.0000000000', '22.8571428571', '97.4285714286', '36.3333317590', '0.5000000000', '3.7000000000', '138.0000000000', '97.0000000000', '84.0000000000', '15.0000000000', '0.5000000000', '1.8000000000', '8.1000000000', '1.0500000000', '31.0000000000', '25.0000000000', '12.0000000000', '3.7000000000', '2.8000000000', '9.5331935709', '8.0000000000', '186.0000000000', '48.3000000000', '14.5000000000', '1.3000000000', '7.5000000000', '84.0000000000', '38.0000000000', '5.0000000000', '0.8000000000', '33.0000000000', '0.0000000000', '0.7121418827', '168.0000000000', '-7090.0000000000', '5.0000000000', '1.0000000000']
        # initial_state = [float(i) for i in initial_state]
        # initial_state = np.array(initial_state)
        initial_state = None
        data, fieldnames = read_data('patients.csv')
        initial_state = np.array([data[self.sofa-1][k] for k in fieldnames])
        return initial_state

    

if __name__ == "__main__":

    # e = EHRGenerator(sofa=1)
    # print(e.get_initial_state())
    # print(type(e.get_initial_state()[0]))

    # a = [1.0 for i in range(46)]
    # b = [2.0 for i in range(4)]
    # a = np.array(a)
    # b = np.array(b)
    # raise Exception(a, b)
    # c = np.concatenate((a,b), axis=0)
    # e = EHRGenerator()
    # print(e.get_next_state(a, b))
    # # print(np.concatenate((a,b), axis=0))
    # pass
    # a = "asdasd"
    # raise Exception(a, a)

    # data, fieldnames = read_data('dataset/rl_data_final_cont.csv')
    # all_features = fieldnames[3:7] + fieldnames[11:50] + [fieldnames[56]] + fieldnames[57:59] + fieldnames[50:52] + fieldnames[59:61]
    # labels = fieldnames[3:7] + fieldnames[11:50] + [fieldnames[56]] + fieldnames[57:59]
    # X, y = get_features_and_labels(data)
    # max_sofa = 0
    # min_sofa = 999
    # for i in y:
    #     if i[44] > max_sofa:
    #         max_sofa = i[44]
    #     if i[44] < min_sofa:
    #         min_sofa = i[44]
    # print(max_sofa)
    # print(min_sofa)

    # temp = EHRGenerator()
    # print(temp.get_next_state(['0.0000000000', '17639.8264351852', '0.0000000000', '0.0000000000'], ['0.0000000000', '17639.8264351852', '0.0000000000', '0.0000000000', '78.6999969482', '15.0000000000', '74.5714285714', '104.7142857143', '72.8571428571', '56.0000000000', '22.8571428571', '97.4285714286', '36.3333317590', '0.5000000000', '3.7000000000', '138.0000000000', '97.0000000000', '84.0000000000', '15.0000000000', '0.5000000000', '1.8000000000', '8.1000000000', '1.0500000000', '31.0000000000', '25.0000000000', '12.0000000000', '3.7000000000', '2.8000000000', '9.5331935709', '8.0000000000', '186.0000000000', '48.3000000000', '14.5000000000', '1.3000000000', '7.5000000000', '84.0000000000', '38.0000000000', '5.0000000000', '0.8000000000', '33.0000000000', '0.0000000000', '0.7121418827', '168.0000000000', '-7090.0000000000', '5.0000000000', '1.0000000000', 0.0, 0.0,]))


    # print(['{0:.10f}'.format(i) for i in X[0][:46]])

    # min_arr = [i for i in y[0]]
    # max_arr = [i for i in y[0]]
    # for i in range(len(y)):
    #     for j in range(len(y[i])):
    #         if y[i][j] < min_arr[j]:
    #             min_arr[j] = y[i][j]
    #         if y[i][j] > max_arr[j]:
    #             max_arr[j] = y[i][j]
    # min_arr = ['{0:.10f}'.format(i) for i in min_arr]
    # max_arr = ['{0:.10f}'.format(i) for i in max_arr]
    # print(min_arr)
    # print("spacer")
    # print(max_arr)


    # # print(X.shape, y.shape)
    # count = 0
    # eg = EHRGenerator()
    # for i in range(len(y)):
    #     if eg._get_sofa_score(y[i]) == y[i][44]:
    #         count += 1
    # print(count)
    # print(len(y))
    # pass


    # X, y = get_features_and_labels('dataset/rl_data_final_cont.csv')
    # indexes = [i for i in range(1, 24)]
    # patient_dict = {}
    # for i in X:
    #     if i[44] > 0 and i[44] < 24 and i[44] not in patient_dict:
    #         patient_dict[int(i[44])] = i[:46]
    #         indexes.remove(int(i[44]))
    #         if indexes == []:
    #             break
    # print(patient_dict.keys())
    # with open('patients.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(all_features[:46])
    #     for i in sorted(patient_dict.keys()):
    #         writer.writerow(patient_dict[i])

    pass