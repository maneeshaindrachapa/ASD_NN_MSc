# Autism Spectrum Disorder Classification on Electroencephalogram data using Neural Networks
## File tree
```commandline
│   capsnet.py
│   eed_data_asd.py
│   hyperparameter_select.py
│   info.py
│   models.py
│   predict.py
│   README.md
│   requirements.txt
│   train_test.py
│   vizualize.py

```
## How to Train Models
This project is compatible with ***Python 3.7.x*** version and all the libraries which need to run this project is included in ***requirements.txt*** file. To install the libraries please run the below command.<br>

```
pip install -r requirements.txt
```

After installing the necessary libraries; make sure that the EEG data is in the ***data*** folder as ***eeg/{ID}/{ID}_{EPOCH}.csv*** files. Then run the below command to read and combine all data.

```
python eeg_data_asd.py R
```

After appending all the data to ***data-original.ftr*** file run the below command to preprocess EEG data

```
python eeg_data_asd.py P
```

After  that preprocessing pipeline will start and preprocessed data will save as ***data-clean.ftr*** file. Then to create the frequency bands dataset run the below command.

```
python eeg_data_asd.py G
```

To Train, Test and get model info run the below command.

```
python train_test.py [parameter] [model_name]
```

In the above command ***[ parameter ][ train | test | info ]*** and ***[ model_name ][ conv | lstm | bilstm | caps | gru |convlstm ]***

## Run Front-end
Go inside ***AutismSpectrumDisorderDetectUI*** and run the below command to install libraries and start the server

```
npm install
ng serve
```
## Run Back-end
Go insiide ***AutismSpectrumDisorderAPI*** and run the below command to start the Flask server
```
flask --app server run
```
## Hyperparameter tuning
```
python hyperparameter_select.py [model_name]
```
## Vizualize Frequency Bands of Each Participant
```
python vizualize.py
```