# Sepsis - EHR Benchmark Environment for RL

This repository is an online reinforcement learning (RL) environment that allows its users to benchmark different RL algorithms in treating sepsis patients given their electronic health record (EHR) data.

## Environment Description
To use this environment, your agent is expected to observe a patient's current EHR and assign some dosage aiming to improve the patient's condition. The generation of patient EHR at the next state is achieved through a machine learning model trained on roughly 250000 data points from the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/). The reward is calculated based on whether the patient's [SOFA Score](https://en.wikipedia.org/wiki/SOFA_score) has improved, which can be derived from the EHR figures. Termination of an episode is achieved through either the patient's mortality rate reaches the minimum (SOFA Score being 0) or the patient's mortality rate reaches the maximum (SOFA score being 24).

### Environment Parameters
- model(str): ['lstm', 'gru', 'rnn', 'stacked_dense_layers', 'transformer']
    - Indicate the backend ML model you wish to use.
    - Models ranked from lowest testing mean squared error to highest: lstm(0.196), gru(0.196), rnn(0.198), transformer(0.202), stacked_dense_layers(0.206).
    - Models generate in different speeds based on your machine builds.
- sofa(int): [1, 2, 3, ..., 19]
    - Indicate the patient's SOFA Score at the initial state. (mortality rate increases as SOFA Score gets higher)

### Observation Space
46 continuous floating point numbers indicating the patient's current EHR: 

[(0, 'gender'), (1, 'age'), (2, 'elixhauser'), (3, 're_admission'), (4, 'Weight_kg'), (5, 'GCS'), (6, 'HR'), (7, 'SysBP'), (8, 'MeanBP'), (9, 'DiaBP'), (10, 'RR'), (11, 'SpO2'), (12, 'Temp_C'), (13, 'FiO2_1'), (14, 'Potassium'), (15, 'Sodium'), (16, 'Chloride'), (17, 'Glucose'), (18, 'BUN'), (19, 'Creatinine'), (20, 'Magnesium'), (21, 'Calcium'), (22, 'Ionised_Ca'), (23, 'CO2_mEqL'), (24, 'SGOT'), (25, 'SGPT'), (26, 'Total_bili'), (27, 'Albumin'), (28, 'Hb'), (29, 'WBC_count'), (30, 'Platelets_count'), (31, 'PTT'), (32, 'PT'), (33, 'INR'), (34, 'Arterial_pH'), (35, 'paO2'), (36, 'paCO2'), (37, 'Arterial_BE'), (38, 'Arterial_lactate'), (39, 'HCO3'), (40, 'mechvent'), (41, 'Shock_Index'), (42, 'PaO2_FiO2'), (43, 'cumulated_balance'), (44, 'SOFA'), (45, 'SIRS')]

### Action Space
4 continuous floating point numbers indicating the dosage assigned to the patient: 

[(0, 'median_dose_vaso'), (1, 'max_dose_vaso'), (2, 'vaso_input'), (3, 'iv_input')]

### Reward
If the patient's SOFA Score increases, the reward is -1.0. (Patient gets worse)

If the patient's SOFA Score decreases, the reward is 1.0. (Patient gets better)

If the patient's SOFA Score remains the same, the reward is 0.0.

###


## Dependencies
To install dependencies, use the following command:
```
pip install -r requirements.txt
```


## Getting Started
The environment follows the standard [Gymnasium](https://gymnasium.farama.org/index.html) API. 

To use the environment, create a notebook in the current directory and include the following code in your notebook:
```py
import gymnasium as gym
from EHR_env import EHREnv

gym.register(
    id='EHREnv-v0',
    entry_point='EHR_env:EHREnv',
    max_episode_steps=100, # Recommended, otherwise likely stuck in an episode.
    kwargs={'model': 'lstm',
            'sofa': 5}
)

env = gym.make('EHREnv-v0', model='lstm', sofa=5)
```
## Example
A more detailed example notebook that benchmarks a few RL algorithms can be found in `example_benchmarking_notebook.ipynb`.

You may view the learning curves of the example benchmark via [TensorBoard](https://www.tensorflow.org/tensorboard) by using the following command in terminal:
```
tensorboard --logdir ./tensorboard/EHREnv-v0/
```
![Screenshot of the learning curves of the example benchmark](/assets/benchmark_result.png)


## About

This project is my learning outcome of the course STATS 199 Individual Study supervised by Professor Hengrui Cai @ University of California, Irvine.