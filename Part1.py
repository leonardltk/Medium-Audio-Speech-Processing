from __future__ import print_function
if True: ## imports / admin
    import os,sys,datetime
    sys.path.insert(0, 'utils')
    from _helper_basics_ import *
    START_TIME=datetime.datetime.now()
    datetime.datetime.now() - START_TIME
    print(f"===========\npython {' '.join(sys.argv)}\n    Start_Time:{START_TIME}\n===========")
    pp = pprint.PrettyPrinter(indent=4)

plot_dir = os.path.join('fig','Part1')
os.makedirs(plot_dir,exist_ok=True)
sr=16000
n_fft=512
win_length=512
hop_length=64
n_mels = 64
winlen=1.*win_length/sr
winstep=1.*hop_length/sr
num_freq_bins=n_fft//2 +1

#################################################################
if True : ## Waveform 
    print("\nWaveform")
    #################################################################
    if True : ## Waveform of the number 
        seven_path = os.path.join('wav','seven.wav')
        seven_wav, sr_out = read_audio(seven_path, sr=16000, mean_norm=False)
        seven_wav_crop = seven_wav[2000:12000]
        ## Plot
        k=1;col=1;l=1; fig=plt.figure(figsize=(8,6))
        plt.subplot(k,col,l); plt.plot(seven_wav_crop); plt.title('seven waveform'); l+=1
        plt.tight_layout(); plt.show()
        fig_path = os.path.join(plot_dir,'seven_waveform.png')
        fig.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)
    #################################################################
    if True : ## Superposition of 1Hz and 2Hz
        freq1=1
        freq2=2
        ## Sampling rate
        x=2*np.pi*np.arange(sr)/sr
        ## Wave
        y1=np.sin(freq1*x)
        y2=np.sin(freq2*x)
        y3=y1+y2
        ## Plot
        k=3;col=1;l=1; fig=plt.figure(figsize=(4*col,2*k))
        plt.subplot(k,col,l); plt.plot(y1); plt.title(f'freq1={freq1}Hz'); l+=1
        plt.subplot(k,col,l); plt.plot(y2); plt.title(f'freq2={freq2}Hz'); l+=1
        plt.subplot(k,col,l); plt.plot(y3); plt.title('freq1 (+) freq2'); l+=1
        plt.tight_layout(); plt.show()
        fig_path = os.path.join(plot_dir,'superposition.png')
        fig.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)

#################################################################
if True : ## Spectrogram 
    print("\nSpectrogram")
    #################################################################
    if True : ## Spectrogram of the number 
        seven_MAG, _, _, seven_LPS = wav2LPS(seven_wav_crop, mode="librosa", n_fft=n_fft, win_length=win_length, hop_length=hop_length,)
        ## Plot
        k=2;col=1;l=1; fig=plt.figure(figsize=(8,10));
        kwargs_plot={'colour_to_set':'black','hop_length':256,'sr':16000,'curr_fig':fig,}
        plt.subplot(k,col,l); plt.plot(seven_wav_crop); plt.title('seven waveform'); l+=1
        display_audio(seven_LPS, 'seven spectrogram',   'spec', kcoll=[k,col,l], **kwargs_plot); l+=1
        plt.tight_layout();plt.show()
        fig_path = os.path.join(plot_dir,'seven_spectrogram.png')
        fig.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)

#################################################################
if True : ## Mel-spectrogram 
    print("\nMel")
    #################################################################
    if True : ## Mel vs Linear 
        melfb = librosa.filters.mel(22050, 2048)
        plt.figure()
        librosa.display.specshow(melfb, x_axis='linear')
        plt.ylabel('Mel filter')
        plt.title('Mel filter bank')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        fig_path = os.path.join(plot_dir,'Mel_filterbank.png')
        plt.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)
    #################################################################
    if True : ## Mel-spectrogram of 'seven'
        seven_Mel=librosa.feature.melspectrogram(sr=16000, S=seven_MAG, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
        seven_LogMel = np.log(seven_Mel+1e-32)
        ## Plot
        k=1;col=2;l=1; fig=plt.figure(figsize=(8,5));
        kwargs_plot={'colour_to_set':'black','hop_length':256,'sr':16000,'curr_fig':fig,}
        display_audio(seven_LPS, 'seven spectrogram',   'spec', kcoll=[k,col,l], **kwargs_plot); l+=1
        display_audio(seven_LogMel, 'seven melspectrogram',   'spec', y_axis='mel', fmax=8000, kcoll=[k,col,l], **kwargs_plot); l+=1
        plt.tight_layout();plt.show()
        fig_path = os.path.join(plot_dir,'seven_melspectrogram.png')
        fig.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)

#################################################################
if True : ## MFCC 
    print("\nMFCC")
    #################################################################
    if True : ## Constructing MFCC
        seven_MFCC_64, seven_MFCC_64_liftered = MFCCvsMEL(seven_LogMel, n_mfcc=64, dct_type=2)
        seven_MFCC_32, seven_MFCC_32_liftered = MFCCvsMEL(seven_LogMel, n_mfcc=32, dct_type=2)
        ## Plot if True : 
        k=1;col=2;l=1; fig=plt.figure(figsize=(8,5));
        kwargs_plot={'colour_to_set':'black','hop_length':256,'sr':16000,'curr_fig':fig,}
        display_audio(seven_MFCC_64_liftered, f'seven MFCC (n_mfcc=64)',   'spec', y_axis=None, fmax=8000, kcoll=[k,col,l], **kwargs_plot); l+=1
        display_audio(seven_MFCC_32_liftered, f'seven MFCC (n_mfcc=32)',   'spec', y_axis=None, fmax=8000, kcoll=[k,col,l], **kwargs_plot); l+=1
        plt.tight_layout();plt.show()
        fig_path = os.path.join(plot_dir,'seven_MFCC.png')
        fig.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)
    if True : ## Reconstructing Log Mel from the MFCCs
        seven_LogMel_reconstruct_64, _ = MFCCvsMEL(seven_MFCC_64, n_mfcc=64, dct_type=3)
        seven_LogMel_reconstruct_32, _ = MFCCvsMEL(seven_MFCC_32, n_mfcc=64, dct_type=3)
        ## Plot if True : 
        k=1;col=2;l=1; fig=plt.figure(figsize=(8,5));
        kwargs_plot={'colour_to_set':'black','hop_length':256,'sr':16000,'curr_fig':fig,}
        display_audio(seven_LogMel_reconstruct_64, f'seven mel reconstructed (n_mfcc=64)',   'spec', y_axis=None, fmax=8000, kcoll=[k,col,l], **kwargs_plot); l+=1
        display_audio(seven_LogMel_reconstruct_32, f'seven mel reconstructed (n_mfcc=32)',   'spec', y_axis=None, fmax=8000, kcoll=[k,col,l], **kwargs_plot); l+=1
        plt.tight_layout();plt.show()
        fig_path = os.path.join(plot_dir,'seven_melspectrogram_reconstructed.png')
        fig.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)
    if True : ## Plot the Spectral envelopes
        tidx=41
        SpecEnvlp_64 = seven_LogMel_reconstruct_64[:,tidx]
        SpecEnvlp_32 = seven_LogMel_reconstruct_32[:,tidx]
        ## Plot if True : 
        k=1;col=2;l=1; fig=plt.figure(figsize=(8,5));
        kwargs_plot={'colour_to_set':'black','hop_length':256,'sr':16000,'curr_fig':fig,}
        # 
        plt.subplot(k,col,l); l+=1;
        plt.plot(SpecEnvlp_64)
        plt.title('Spectral Envlp (n_mfcc=64)')
        # 
        plt.subplot(k,col,l); l+=1;
        plt.plot(SpecEnvlp_64)
        plt.plot(SpecEnvlp_32,'r')
        plt.title('Spectral Envlp (n_mfcc=32)')
        # 
        plt.tight_layout();plt.show()
        fig_path = os.path.join(plot_dir,'seven_spectral_envelope.png')
        fig.savefig(fig_path)
        plt.close()
        print('Saved to ',fig_path)

#################################################################
END_TIME=datetime.datetime.now()
print(f"===========\
    \nDone \
    \npython {' '.join(sys.argv)}\
    \nStart_Time  :{START_TIME}\
    \nEnd_Time    :{END_TIME}\
    \nDuration    :{END_TIME-START_TIME}\
\n===========")
