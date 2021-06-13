import matplotlib.pyplot as plt
import os
import numpy as np
import random as rnd
from classify_emotions import TrainConfig
from data_interact import add_zeros


INPUT_DATA = '$PROJ_DIR/emotions.csv'

def generate_artifact(mx, mn):
    nof_spikes = rnd.randint(4, 6)
    artifact = np.ones(598)*np.array(list(0.001*rnd.randint(1, 100) for _ in range(598)))
    for _ in range(nof_spikes):
        nof_samples = rnd.randint(5, 10)
        x_data = np.linspace(0, 0.01* rnd.randint(340, 628), nof_samples)
        s = np.sin(x_data)
        pos = rnd.randint(0, 597-len(s))
        artifact[pos: pos+len(s)] += s
        artifact[pos: pos+len(s)] += np.array(list(0.005*rnd.randint(1, 100) for _ in range(len(s))))
    artifact = 4*mx/max(artifact) * artifact
    return artifact


def customize_data(*matrices):
    full_m = matrices[0]
    for mat in matrices[1:]:
        full_m += mat

    for k, img in enumerate(full_m):
        idx = k
        channels = list()
        channels.append(np.array(img[8][:598]))
        channels.append(np.array(img[9][:598])* np.array(list(0.1 * rnd.randint(1,10) for i in range(598))))
        channels.append(np.array(img[8][152:])* np.array(list(0.1 * rnd.randint(1,10) for i in range(598))))
        channels.append(np.array(img[9][152:])* np.array(list(0.1 * rnd.randint(1,10) for i in range(598))))

        ampl0 = max(channels[0])
        ampl1 = min(channels[0])

        artifact1 = generate_artifact(ampl0, ampl1)
        artifact2 = -1* artifact1 + np.array(list(0.1 * rnd.randint(-10,10) for i in range(598)))
        channels.append(artifact1)
        channels.append(artifact2)

        # Filtered channels
        csv_lines = []
        for i in range(598):
            csv_lines.append(','.join([
                str(round(channels[0][i], 1)), str(round(channels[1][i], 1)),
                str(round(channels[2][i], 1)), str(round(channels[3][i], 1)),
                str(round(channels[4][i], 1)), str(round(channels[5][i], 1))]))
        csv_text = '\n'.join(csv_lines)
        with open(os.path.join('filtered_data', f'filtered_sample_{add_zeros(idx, 4)}.csv'), 'w') as f:
            f.write(csv_text)

        plt.figure()
        fig, axs = plt.subplots(6, 1)
        for i, channel in enumerate(channels):
            axs[i].plot(channel)
            if i < 4:
                axs[i].set_title(f"EEG {i+1}", fontsize=10)
            else:
                axs[i].set_title(f"EOG {i-3}", fontsize=10)
        plt.draw()
        plt.savefig(os.path.join('filtered_data_plotted', f'filtered_sample_{add_zeros(idx, 4)}.png'))

        # UNfiltered channels
        init_channels = list(channels)
        init_channels[0] = init_channels[0]* rnd.randint(70,80)/100 + init_channels[4]*rnd.randint(20, 30)/100
        init_channels[1] = init_channels[1]* rnd.randint(70,80)/100 + init_channels[4]*rnd.randint(20, 30)/100
        init_channels[2] = init_channels[2]* rnd.randint(70,80)/100 + init_channels[4]*rnd.randint(20, 30)/100
        init_channels[3] = init_channels[3]* rnd.randint(70,80)/100 + init_channels[4]*rnd.randint(20, 30)/100
        init_channels[4] = init_channels[4]* rnd.randint(70,80)/100 + init_channels[0]*rnd.randint(20, 30)/100
        init_channels[5] = init_channels[5]* rnd.randint(70,80)/100 + init_channels[1]*rnd.randint(20, 30)/100
        csv_lines = []
        for i in range(598):
            csv_lines.append(','.join([
                str(round(init_channels[0][i], 1)), str(round(init_channels[1][i], 1)),
                str(round(init_channels[2][i], 1)), str(round(init_channels[3][i], 1)),
                str(round(init_channels[4][i], 1)), str(round(init_channels[5][i], 1))]))
        csv_text = '\n'.join(csv_lines)
        with open(os.path.join('data', f'sample_{add_zeros(idx, 4)}.csv'), 'w') as f:
            f.write(csv_text)

        plt.figure()
        fig, axs = plt.subplots(6, 1)
        for i, channel in enumerate(init_channels):
            axs[i].plot(channel)
            if i < 4:
                axs[i].set_title(f"EEG {i+1}", fontsize=10)
            else:
                axs[i].set_title(f"EOG {i-3}", fontsize=10)
        plt.draw()
        plt.savefig(os.path.join('data_plotted', f'sample_{add_zeros(idx, 4)}.png'))


if __name__ == "__main__":
    config = TrainConfig(dataset_path=INPUT_DATA, dnn_type='CNN', training_weight=0.8, nof_dimensions=2)
    numeric_data, labels, features_list = config._extract_data()
    X_train, y_train, X_test, y_test, channels = config.extract_network_data()
    out = ['sample_index,patient_index,patient_sex,emotional_state']
    for i, label in enumerate(labels):
        if i >= 1260:
            sex = 'MALE'
        else:
            sex = 'FEMALE'
        if i < 420:
            index = 1
        elif i < 840:
            index = 2
        elif i < 1260:
            index = 3
        elif i < 1696:
            index = 4
        else:
            index = 5
        label = label.replace('HAPPY','HAPPINESS').replace('SAD', 'SADNESS')
        out.append(f'{i},{index},{sex},{label}')
    with open('labels_map.csv', 'w') as f:
        f.write('\n'.join(out))
    '''

    customize_data(X_train, X_test)

    X_train, y_train, X_test, y_test = config.flatten_data(
        X_train, y_train, X_test, y_test)
    patient1 = numeric_data[: 420]
    patient2 = numeric_data[420: 840]
    patient3 = numeric_data[840: 1260]
    patient4 = numeric_data[1260: 1696]
    patient5 = numeric_data[1696: ]
    '''
    #2394
