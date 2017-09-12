import librosa
import numpy as np

# Verify the number of frames
# - The number of frames (n_of_frames)
#  if 0 < hop_length <= n_fft,
#  (n_of_frames - 1) * hop_length + n_fft = sr * the total time in second
#  n_of_frames * hop_length - hop_length + n_fft = sr * the total time in second
# 
#  - Formula : n_of_frames = (sr * the total time in second + hop_length - n_fft) / hop_length
def verify_number_of_frames(n_of_frames, y, sr, n_fft, hop_length):
    __total_time = len(y) / sr
    __n_of_frames = int(np.ceil((sr * __total_time + hop_length - n_fft) / hop_length))
    if __n_of_frames != n_of_frames:
        print("%d != %d, total_time=%d" % (n_of_frames, __n_of_frames, __total_time))
        raise ValueError


if __name__ == '__main__':
    # #### MIREX Audio Formats
    # - CD-quality (PCM, 16-bit, 44100 Hz)
    # - single channel (mono)
    # - file length between 2 and 36 seconds (total time: 14 minutes)
    # 
    # From experience,
    # - n_fft: # samples of a frame = 2048 samples@22050Hz, 4096 samples@44100Hz = 92.879818594104ms = frame duration
    # - hop_length = n_fft * 3 // 4, that means overlap n_fft / 4 = 512 samples@22050Hz, 1024 samples@44100Hz = 23.219954648526 = overlap duration
    sr = 44100
    n_fft = 4096
    hop_length = n_fft * 3 // 4
    frame_duration = n_fft / sr
    print("frame_duration = %f ms" % (frame_duration))

    # Default 40 listeners for each audio
    # Default 431 frames for each 30-minute audio
    # frames_beats = [  [0, 0, 0, 1, ...., 1, 0], => row is listener
    #                   [0, 0, 0, 1, ...., 1, 0], => column is frame
    #                   ......                  , => the value 0/1 represents which frame with beat
    #                   [0, 0, 0, 1, ...., 1, 0] ]
    frames_beats = np.ndarray(shape=(40, 431), dtype=int)

    for i in range(1, 21):
        train_fprefix = 'dataset/mirex_beat_tracking_2016/train/train%d' % (i)
        print(train_fprefix)

        audio_raw_data, sr = librosa.load(train_fprefix +'.wav', sr=sr, mono=True) # default mono=True
        linear_freq = librosa.stft(y=audio_raw_data, n_fft=n_fft, hop_length=hop_length)
        #print("n_of_freq_bins = %d" % (len(linear_freq)))
        n_of_frames = len(linear_freq[0, :])
        verify_number_of_frames(n_of_frames, y=audio_raw_data, sr=sr, n_fft=n_fft, hop_length=hop_length)

        beat_sequences = np.genfromtxt(train_fprefix + '.txt', dtype='str', delimiter='\r')
        # beat_sequences.shape = (40,) # each element is one beat sequence in seconds
        n_of_listeners = len(beat_sequences)

        frames_beats.resize((n_of_listeners, n_of_frames), refcheck=False)
        frames_beats.fill(0)
        for ith_listener in range(n_of_listeners):
            beat_seq = beat_sequences[ith_listener] # the ith listener's beat sequence

            # step1. split into beats list. each element is a beat in seconds.
            beats_sec = [float(b) for b in beat_seq.split('\t')] # beats_sec = [0.625, 1.235, 1.740, ...]
            
            # step2. use beats_sec to indicate which frame has a beat.
            frames_idx = range(n_of_frames) # frames_idx = [0, 1, 2, ..., n_of_frames-1]
            frames_time = librosa.frames_to_time(frames_idx, sr=sr, hop_length=hop_length) # frames_time = [0, t1, t2, ...]
            start_fi = 0
            for beat_sec in beats_sec:
                for fi in frames_idx[start_fi:]:
                    ti = frames_time[fi]
                    if beat_sec < ti:
                        frames_beats[ith_listener][fi - 1] = 1
                        start_fi = fi
                        break # found, to do next beat
                else:
                    if beat_sec < (frames_time[-1] + frame_duration): # check if the beat is within the last frame
                        frames_beats[ith_listener][-1] = 1
                    else:
                        # not found, there must be someting wrong here
                        raise ValueError

            # then next listener's beat_sequence

        # [Option1] Linear frequency
        # linear_freq.shape = (n_of_freq_bins, n_of_frames) = (2049, 431)
        #linear_freq = librosa.amplitude_to_db(linear_freq, ref=np.max)

        # [OPtion2] Mel-frequency
        # mel_freq.shape = (n_of_freq_bins, n_of_frames) = (128, 431)
        linear_ps = np.abs(linear_freq)**2 # Linear power spectram
        mel_freq = librosa.feature.melspectrogram(S=linear_ps)
        mel_freq = librosa.power_to_db(mel_freq, ref=np.max)

        # train1_x.npy, train2_x.npy ...
        # [Option1] (n_of_frames, n_of_freq_bins)
        #train_x = linear_freq.T # (431, 2049)
        train_x = mel_freq.T[:431, :] # (431, 128) only train16_x.shape = (432, 128), skip the last frame

       # [Option2] train_x.shape = (n_of_listeners, n_of_freq_bins, n_of_frames) = (40, 431, 2049)
        #train_x = [linear_freq.T] * n_of_listeners #=> shape(40,431,2049)

        # [Option3] train_x.shape = (n_of_listeners*n_of_frames, n_of_freq_bins) = (40*431, 2049)
        #train_x = np.tile(linear_freq.T, (n_of_listeners, 1)) #=> shape(17240, 2049)

        np.save(train_fprefix + '_x.npy', train_x, allow_pickle=True, fix_imports=False)
        #np.savetxt(train_fprefix + '_x.txt', train_x, fmt='%f', delimiter=',', newline='\n')

        # train_y.npy
        # frames_beats.shape = (n_of_listeners, n_of_frames) = (40, 431)
        # Only train16_y.shape = (40, 432) and the last frames don't contain any beats.
        train_y = frames_beats[:, :431]

        # [Option2] train_y.shape = (n_of_frames, n_of_listeners) = (431, 40)
        #train_y = frames_beats.T

        # [Option3] train_y.shape = (n_of_frames*n_of_listeners, 1) NOTICE: not n_of_listeners*n_of_frames. size is the same. content is different.
        #train_y = frames_beats
        #train_y.resize((train_y.size, 1), refcheck=False)

        np.save(train_fprefix + '_y.npy', train_y, allow_pickle=True, fix_imports=False)
        #np.savetxt(train_fprefix + '_y.txt', train_y, fmt='%d', delimiter=',', newline='\n')

        # then next training file
