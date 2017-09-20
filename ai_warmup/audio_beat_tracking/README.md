# Evaluate performance of audio beat tracking

## ToDo Tasks:

- [x] Learn the usage of librosa STFT - [stft_practice.py](stft_practice.py)
- [x] Check onset strength process - [onset_strength_practice.py](onset_strength_practice.py)
- [x] Realize MIREX 2016 training data - [train_data_quicklook.py](train_data_quicklook.py)
- [x] Preprocess training data - [mirex2016_dataset.py](mirex2016_dataset.py)
- [x] Single LSTM Layer - [lstm_beat_tracking.py](lstm_beat_tracking.py)
- [x] Design Model with stacked LSTM
- [x] Taking Mel-freq bin as input_x
- [ ] Bi-directional LSTM (option)
- [ ] Demo

## Input Data
- 2016 MIREX Data
- Total 20 audio excerpts, each 30 sec
- Sample rate = 44100 Hz
- 1 frame = 2048 samples = 46.44ms
- Each excerpt can be divided into 862 frames (because overlap 1/4 frame)
- Each excerpt was labeled beats by 40 listeners
- Take a quick look at the beat distribution. [source](train_data_quicklook.py)
- The frame was labeled by anyone was treated as 1 (called beat frame)
- Beat frame : Blank frame = 9201 : 8039 = 53.37 : 46.63 (original, 11.27 : 88.73)
- Data preprocessing including
    * Mel-freq [source](onset_strength_practice.py)
    * The positive first order difference
    ```
        D(n,m) = H (M(n,m) − M(n − 1,m))
        H(x) = (x+|x|)/2
    ```

## Model Architecture
- Look back 20 frames (time_steps = 20)
- Features : 128 diffs of Mel-freq bins
- Stacked LSTM
    * 1st LSTM with 128 units
    * 2nd LSTM with 40 units
    * Activation : tanh
- Optimizer : RMSprop
- Loss function : Binary crossentropy
- 500 epochs
- Validation split : 0.2 (4 excerpts)
- Shuffle = False
![](https://lh6.googleusercontent.com/P18xDaB3MeJWb-TG5mQolMiJxUNOJafBkTBDu7RpfDiPwYNeRWS8Iwzymb15sR0ItN-OhaQUP_v7iKQ=w2560-h1406-rw)

## Result
- [Tensorboard](http://localhost:6006)
- Learning rate = 0.000001
- Traing dataset
    * loss = 0.69
    * accuracy = 0.52
    * recall = 0.89
    * precision = 0.51
- Validation dataset
    * loss = 0.69
    * accuracy = 0.63
    * recall = 0.99
    * precision = 0.63
- Universal Onset Detection with BLSTM (Dataset: Bello-set)
    * recall = 0.925
    * precision = 0.945
- [Result notebook](http://localhost:8888/notebooks/sound/beat_tracking_demo.ipynb)

## Public Dataset

- [MIREX Database](http://www.music-ir.org/mirex/wiki/2016:Audio_Beat_Tracking)
    * Beat locations have been annotated in each excerpt by 40 different listeners (39 listeners for a few excerpts) 
    * The length of each excerpt is 30 seconds
    * This dataset contains 217 excerpts around 40s each, of which 19 are "easy" and the remaining 198 are "hard"
    * The harder excerpts were drawn from the following musical styles: Romantic music, ﬁlm soundtracks, blues, chanson and solo guitar
    * This dataset has been designed for radically new techniques which can contend with challenging beat tracking situations like: quiet accompaniment, expressive timing, changes in time signature, slow tempo, poor sound quality etc


## Reference
- Paper: [Improved Musical Onset Detection With Convolutional Neural Networks](http://ieeexplore.ieee.org/document/6854953/)
