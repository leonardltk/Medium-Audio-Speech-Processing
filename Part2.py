from __future__ import print_function
if True: ## imports / admin
    import os,sys,datetime
    sys.path.insert(0, 'utils')
    from _helper_basics_ import *
    START_TIME=datetime.datetime.now()
    datetime.datetime.now() - START_TIME
    print(f"===========\npython {' '.join(sys.argv)}\n    Start_Time:{START_TIME}\n===========")
    pp = pprint.PrettyPrinter(indent=4)

plot_dir = os.path.join('fig','Part2')
os.makedirs(plot_dir,exist_ok=True)
sr=1600
n_fft=512
win_length=512
hop_length=64
n_mels = 64
winlen=1.*win_length/sr
winstep=1.*hop_length/sr
num_freq_bins=n_fft//2 +1

#################################################################
if 0 : ## Stationary Noise
    print("\nStationary Noise")
    #################################################################
    if True : ## Waveform of the number 'seven'
        seven_path = os.path.join('wav','seven.wav')
        seven_wav, sr_out = read_audio(seven_path, sr=sr, mean_norm=False)
        seven_wav_crop = seven_wav[1000:14000]
    if True : ## Gaussian noise
        noise = np.random.normal(loc=0.0, scale=1.0, size=seven_wav_crop.shape)
    k=4;col=2;l=1; fig=plt.figure(figsize=(4*col,3*k))
    if True : ## Plot waveform
        plt.subplot(k,col,l); plt.plot(seven_wav_crop); plt.title('clean'); l+=1
        plt.subplot(k,col,l); plt.plot(noise); plt.title('noise'); l+=1
        for snr in [15,0]:
            seven_nsy = augment_noisy_wave(  seven_wav, noise, snr,  allow_clipping=True)
            plt.subplot(k,col,l); plt.plot(seven_nsy); plt.title(f'snr={snr}'); l+=1
    if True : ## Plot spectrogram
        _, _, _, seven_LPS = wav2LPS(seven_wav_crop, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        _, _, _, noise_LPS = wav2LPS(noise, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        plt.subplot(k,col,l)
        cmap = librosa.display.cmap(seven_LPS)
        librosa.display.specshow(seven_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
        plt.title(f'clean')
        plt.subplot(k,col,l)
        cmap = librosa.display.cmap(noise_LPS)
        librosa.display.specshow(noise_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
        plt.title(f'noise')
        # 
        for snr in [15,0]:
            seven_nsy = augment_noisy_wave(  seven_wav, noise, snr,  allow_clipping=True)
            _, _, _, seven_LPS = wav2LPS(seven_nsy, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            plt.subplot(k,col,l)
            librosa.display.specshow(seven_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
            plt.title(f'snr={snr}')
    plt.tight_layout(); plt.show()
    fig_path = os.path.join(plot_dir,'stationary_noise.png')
    fig.savefig(fig_path)
    plt.close()
    print('Saved to ',fig_path)

#################################################################
if 0 : ## Non-Stationary Noise
    print("\nNon-Stationary Noise")
    #################################################################
    if True : ## Waveform of the number 'seven'
        noise_path = os.path.join('wav','n5.wav')
        noise_babble, sr_out = read_audio(noise_path, sr=sr, mean_norm=False)
        noise_babble = noise_babble[1000:1000+len(seven_wav)]
    k=4;col=2;l=1; fig=plt.figure(figsize=(4*col,3*k))
    if True : ## Plot waveform
        plt.subplot(k,col,l); plt.plot(seven_wav_crop); plt.title('clean'); l+=1
        plt.subplot(k,col,l); plt.plot(noise_babble); plt.title('noise babble'); l+=1
        for snr in [15,0]:
            seven_nsy = augment_noisy_wave(  seven_wav, noise_babble, snr,  allow_clipping=True)
            plt.subplot(k,col,l); plt.plot(seven_nsy); plt.title(f'snr={snr}'); l+=1
    if True : ## Plot spectrogram
        _, _, _, seven_LPS = wav2LPS(seven_wav_crop, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        _, _, _, noise_LPS = wav2LPS(noise_babble, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        plt.subplot(k,col,l)
        cmap = librosa.display.cmap(seven_LPS)
        librosa.display.specshow(seven_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
        plt.title(f'clean')
        plt.subplot(k,col,l)
        cmap = librosa.display.cmap(noise_LPS)
        librosa.display.specshow(noise_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
        plt.title(f'noise babble')
        # 
        for snr in [15,0]:
            seven_nsy = augment_noisy_wave(  seven_wav, noise_babble, snr,  allow_clipping=True)
            _, _, _, seven_LPS = wav2LPS(seven_nsy, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            plt.subplot(k,col,l)
            librosa.display.specshow(seven_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
            plt.title(f'snr={snr}')
    plt.tight_layout(); plt.show()
    fig_path = os.path.join(plot_dir,'non_stationary_noise.png')
    fig.savefig(fig_path)
    plt.close()
    print('Saved to ',fig_path)

#################################################################
if True : ## Reveberation
    # 
    sr=8000
    n_fft=256
    win_length=256
    hop_length=128
    # 
    print("\nReveberation")
    phrase_path = os.path.join('wav','dr1_fcjf0_sa2.wav')
    phrase_wav, sr_out = read_audio(phrase_path, sr=sr, mean_norm=False)
    #################################################################
    k=2;col=2;l=1; fig=plt.figure(figsize=(4*col,3*k))
    if True : ## Plot waveform
        plt.subplot(k,col,l); plt.plot(phrase_wav); plt.title('clean'); l+=1
        for rir_bn in ['3']:
            RIR_path = os.path.join('wav',f'RIR_RT60_0.2s_{rir_bn}m_20d_ch1.mat')
            RIR_dict = sio.loadmat(RIR_path)
            RIR = RIR_dict['RIR_cell'][0][0][:,0]
            phrase_rev = augment_reverb_wave(phrase_wav, RIR,  pow_eq=True)
            # Write to file
            rev_path = os.path.join('wav',f'reverb_{rir_bn}m.wav')
            write_audio(rev_path,phrase_rev,sr)
            print(f'Written to {rev_path}')
            # 
            # plt.subplot(k,col,l); plt.plot(RIR); plt.title('RIR'); l+=1
            plt.subplot(k,col,l); plt.plot(phrase_rev); plt.title(f'reverbed dist={rir_bn}m'); l+=1
    if True : ## Plot spectrogram
        plt.subplot(k,col,l)
        _, _, _, phrase_LPS = wav2LPS(phrase_wav, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        cmap = librosa.display.cmap(phrase_LPS)
        librosa.display.specshow(phrase_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
        plt.title(f'clean')
        # 
        for rir_bn in ['3']:
            RIR_path = os.path.join('wav',f'RIR_RT60_0.2s_{rir_bn}m_20d_ch1.mat')
            RIR_dict = sio.loadmat(RIR_path)
            RIR = RIR_dict['RIR_cell'][0][0][:,0]
            phrase_rev = augment_reverb_wave(phrase_wav, RIR,  pow_eq=True)
            _, _, _, phrase_LPS = wav2LPS(phrase_rev, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            plt.subplot(k,col,l)
            librosa.display.specshow(phrase_LPS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap); l+=1
            plt.title(f'reverbed dist={rir_bn}m')
    plt.tight_layout(); plt.show()
    fig_path = os.path.join(plot_dir,'reverberation.png')
    fig.savefig(fig_path)
    plt.close()
    print('Saved to ',fig_path)



#################################################################
# pdb.set_trace()

#################################################################
END_TIME=datetime.datetime.now()
print(f"===========\
    \nDone \
    \npython {' '.join(sys.argv)}\
    \nStart_Time  :{START_TIME}\
    \nEnd_Time    :{END_TIME}\
    \nDuration    :{END_TIME-START_TIME}\
\n===========")
