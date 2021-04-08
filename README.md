# ECG_classify
This python project is an implelmentation of paper "Attention Based Joint Learning for Supervised Electrocardiogram Arrhythmia 
Differentiation with Unsupervised Abnormal Beat Segmentation".

## Setup
Just create a new virtual environment with python3. 
```
python>=3.5
torch==1.8.1
```
And install other packages according to the requirements.txt, using command: pip install -r requirements.txt

## How to use it
### Data preprocessing
put the ecg data under the directory data/ecg_data, with the following structure:

```
├── classA
│   ├── patient1
│   │   ├── lead1.txt
│   │   ├── lead2.txt
...
│   │   ├── lead12.txt
│   ├── patient2
├── classB
│   ├── patient1
│   ├── patient2

```
and run the datasets/preprocess.py to remove noise, remove baseline shift, do the normalization save 12 leads to numpy file. The preprocessed data will be saved
in the data/preprocess directory

### Training
After finishing the above process, simple run the run_fold_seg_pipleine.py to do training, and testing is included after the training process. 
```
python run_fold_seg_pipeline
```

