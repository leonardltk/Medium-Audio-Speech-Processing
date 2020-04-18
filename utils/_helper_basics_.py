#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
if 1 : ## Imports
    import sys, os, datetime, traceback, pprint, pdb
    import subprocess, itertools, importlib , math, glob, time, random, shutil, csv, statistics, ast, heapq
    import numpy as np
    import scipy.io as sio
    import collections
    import pickle

    ## Plots
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ## Audio
    import wave, python_speech_features
    import librosa, librosa.display
    import scipy, scipy.io, scipy.io.wavfile, scipy.signal
    import soundfile as sf
    import audiofile as af
    import resampy
##########################################################################################################################################################
## Saving/Loading 
def dump_load_pickle(file_Name, mode, a=None):
    if mode == 'dump':
        # open the file for writing
        fileObject = open(file_Name,'wb') 
        
        # this writes the object a to the file named 'testfile'
        pickle.dump(a,fileObject, protocol=2)   
        # cPickle.dump(a,fileObject, protocol=2)   
        
        # here we close the fileObject
        fileObject.close()
        b = file_Name
    elif mode == 'load':
        # we open the file for reading
        fileObject = open(file_Name,'rb')  
        
        # load the object from the file into var b
        b = pickle.load(fileObject)  
        # b = cPickle.load(fileObject)  
        
        # here we close the fileObject
        fileObject.close()
    return b

##########################################################################################################################################################
## Audio - Admin
if True:
    def read_audio(filename_in, mode="audiofile", sr=None, mean_norm=False):
        """
            Input : file name
            Return : waveform in np.float16, range [-1,1]

            mode=="scipy" : will read in int16, which we will convert to float and divide by 2**15

            mean_norm=True if want to return the mean_normalised waveform

            matlab reads in the same way as librosa do.
        """

        ## Reading the audio
        if mode=="wave":
            with wave.open(filename_in) as w:
                data = w.readframes(w.getnframes())
            sig = np.frombuffer(data, dtype='<i2').reshape(-1, channels)
            normalized = utility.pcm2float(sig, np.float32)
            sound = normalized
            # 
            # sound_wav = wave.open(filename_in)
            # n = sound_wav.getnframes()
            # sound = sound_wav.readframes(n)
            # debug_cannot_extract_array_values
            # 
            sound_fs = sound_wav.getframerate()
        elif mode=="scipy":
            [sound_fs, sound] = scipy.io.wavfile.read(filename_in)
            assert sound.dtype=='int16'
            sound = 1.*sound
        elif mode=="librosa":
            # must define sr=None to get native sampling rate
            sound, sound_fs = librosa.load(filename_in,sr=sr)
            # sound *= 2**15
        elif mode=="soundfile":
            sound, sound_fs = sf.read(filename_in)
        elif mode=="audiofile":
            sound, sound_fs = af.read(filename_in)

        ## Convert to mono
        if len(sound.shape)>1:
            sound = np.mean(sound, axis=0)

        ## Resampling
        if sr and sr!=sound_fs: 
            sound = resampy.resample(sound, sound_fs, sr, axis=0)
            sound_fs=sr
        
        ## Zero-mean
        if mean_norm: sound -= sound.mean()
        
        return sound, sound_fs
    def write_audio(filename_out,x_in,sr,mode="soundfile"):
        """
            Assume input is in the form np.int16, with range [-2**15,2**15]
        """
        curr_x_in_dtype=x_in.dtype
        if mode == "librosa":
            print('\nThis is now deprecated, use mode==soundfile instead\n')
            # assert (curr_x_in_dtype==np.float16)       , '{} is wrong, save in np.float16'.format(curr_x_in_dtype)
            assert np.max(np.abs(x_in))<=1      , '{} is out of range'.format(filename_out)
            librosa.output.write_wav(filename_out, x_in, sr)
        elif mode == "scipy":
            assert curr_x_in_dtype==np.int16         , 'curr_x_in_dtype={} is wrong, save in np.int16'.format(curr_x_in_dtype)
            assert (not np.max(np.abs(x_in))>2**15)  , 'max is {} .\n {} is out of range'.format(np.max(np.abs(x_in)),filename_out)
            assert (not np.max(np.abs(x_in))==0) , 'min is {} .\n{} is either double in [-1,1] or 0Hz, please check, skipping...'.format(np.min(np.abs(x_in)),filename)
            scipy.io.wavfile.write(filename_out, sr, x_in)
        elif mode == "soundfile":
            assert np.max(np.abs(x_in))<=1      , '{} is out of range'.format(filename_out)
            sf.write(filename_out,x_in,sr)
        else:
            print('mode:{} is incorrect should be librosa/scipy/soundfile'.format(mode))
    def display_audio(sig, title, 
        mode='wave', sr=None, hop_length=None, 
            fmin=None, fmax=None,
            x_axis='time', y_axis='hz', 
            num_bins=None,
        xlims=None,ylims=None,clims=None,
        autoplay=False, colorbar=None, 
        colour_to_set='white', curr_fig=None, kcoll=None):

        if curr_fig and kcoll: 
            ax = curr_fig.add_subplot(kcoll[0],kcoll[1],kcoll[2])

        ## Modes ----------------------------------------------
        if mode=='wave':
            librosa.display.waveplot(sig, sr=sr)
            if not title==None: plt.title(title) 
        if mode=='plot':
            plt.plot(sig)
            if not title==None: plt.title(title) 
        elif mode=='spec':
            librosa.display.specshow(sig, 
                sr=sr, hop_length=hop_length,
                fmin=fmin, fmax=fmax,
                x_axis=x_axis, y_axis=y_axis)
            if not title==None: plt.title(title) 
        elif mode=='matplot':
            plt.imshow(sig)
            if not title==None: plt.title(title) 
        elif mode=='audio':
            
            import IPython.display as ipd
            # ipd.display( ipd.Audio(""+"hello.wav") )
            # ipd.display( ipd.Audio(spkr, rate=sr) )

            if not title==None: print(title)
            ipd.display( ipd.Audio(sig, rate=sr, autoplay=autoplay) )
        elif mode=='audio_terminal':
            if not title==None: print(title)
            play_audio(sig,CHUNK=1024)
        elif mode=='image':
            import IPython.display as ipd
            # ipd.display( ipd.Image("LSTM.png") )
            # ipd.display( ipd.Image(x, format='png') )

            ipd.display( ipd.Image(sig, format='png') )
            if not title==None: plt.title(title) 
        elif mode=='constellation_points':
            """ Usage : 

                ## Read Audio
                x,sr_out=read_audio(song_path, mode="librosa", sr=sr, mean_norm=False)
                
                ## Get feats
                kwargs_STFT={
                    'pad_mode':True,
                    'mode':'librosa',
                        'n_fft':conf_sr.n_fft,
                        'win_length':conf_sr.win_length,
                        'hop_length':conf_sr.hop_length,
                        'nfft':conf_sr.nfft,
                        'winstep':conf_sr.winstep,
                        'winlen':conf_sr.winlen,
                        'fs':conf_sr.sr,
                }
                x_MAG, _,_,x_LPS=wav2LPS(x, **kwargs_STFT)
                
                ## ...Dropbox/Work/BIGO/2_Projects/002_MusicHashing/jja178/Combined/v1_baseline/utils/_Shazam_.py
                import _Shazam_ as Shazam
                kwargs_hashPeaks={
                    'num_tgt_pts':3,
                    "delay_time" : seconds_to_frameidx(1, conf_sr.hop_length, conf_sr.n_fft, conf_sr.sr),
                    "delta_time" : seconds_to_frameidx(5, conf_sr.hop_length, conf_sr.n_fft, conf_sr.sr),
                    "delta_freq" : Hz_to_freqidx(1500, conf_sr.num_freq_bins, conf_sr.sr),
                    }
                raw_peaks = Shazam.get_raw_constellation_pts(x_MAG) 
                filt_peaks = Shazam.filter_peaks(raw_peaks, conf_sr.n_fft, high_peak_percentile=75,low_peak_percentile=60)
                filt_peaks_large = [(curr_peak[0],curr_peak[1],10) for curr_peak in filt_peaks]
                # hashMatrix = Shazam.hashPeaks(filt_peaks, conf_sr, **kwargs_hashPeaks)
                [(curr_peak[0],curr_peak[1],10) for curr_peak in filt_peaks]
                
                ## Plot
                k=3;col=1;l=1; curr_fig=plt.figure(figsize=(6*col,3*k)); 
                kwargs_plot={'colour_to_set':'black','hop_length':conf_sr.hop_length,'sr':conf_sr.sr,'curr_fig':curr_fig,}
                display_audio(x,     'x',       'wave', kcoll=[k,col,l], **kwargs_plot); l+=1
                display_audio(x_LPS, 'x_LPS',   'spec', kcoll=[k,col,l], **kwargs_plot); l+=1
                display_audio([x_LPS,filt_peaks_large], 'Peaks', 'constellation_points', kcoll=[k,col,l], **kwargs_plot); l+=1
                plt.tight_layout();plt.show()
                plt.savefig('plot_path.png')
            """
            curr_lps,curr_peaks = sig
            librosa.display.specshow(curr_lps, sr=sr, hop_length=hop_length)
            plt.scatter(*zip(*curr_peaks), marker='.', color='blue', alpha=0.5)
            if not title==None: plt.title(title) 
        elif mode=='histogram':
            """ Usage : 
                num_bins=100

                x,sr_out=read_audio(song_path, mode="librosa", sr=sr, mean_norm=False)
                shash_db=Shazam.get_Shazam_hash(x, kwargs["kwargs_peaks"], kwargs["kwargs_hashPeaks"], kwargs["kwargs_STFT"])
                timepairs_db = Shazam.findTimePairs(shash_db, shash_db)
                
                ## Plot
                k=3;col=1;l=1; curr_fig=plt.figure(figsize=(6*col,3*k)); 
                kwargs_plot={'colour_to_set':'black', 'mode':histogram, 'curr_fig':curr_fig,}
                display_audio(timepairs_db, f'timepairs_db', 'histogram', num_bins=num_bins, kcoll=[k,col,l], **kwargs_plot); l+=1
                plt.tight_layout();plt.show()
                plt.savefig('plot_path.png')
            """
            plt.hist(sig, bins=num_bins)
            if not title==None: plt.title(title) 
        ## Modes ----------------------------------------------
        
        if not colorbar==None: plt.colorbar() 
        if not xlims==None: plt.xlim(xlims) 
        if not ylims==None: plt.ylim(ylims) 
        if not clims==None: plt.clim(clims) 

        if colour_to_set and curr_fig and kcoll:
            ax.spines['bottom'].set_color(colour_to_set)
            ax.spines['top'].set_color(colour_to_set)
            ax.yaxis.label.set_color(colour_to_set)
            ax.xaxis.label.set_color(colour_to_set)
            ax.tick_params(axis='both', colors=colour_to_set)
            ax.title.set_color(colour_to_set)

## Audio - Feature Extraction
if True:
    ## Pre-processing Stage 2
    def get_cmvn_stats(feat_in, norm_mode, axis_cmvn=1, ):
        """
            feat_in : (num_ceps, time), use axis_cmvn=1
            norm_mode : 'cmn','cmvn'
        """
        if norm_mode=='cmn':
            curr_mean = np.mean(feat_in,axis=axis_cmvn)
            feat_mean = np.expand_dims( curr_mean, axis=axis_cmvn) 
            return feat_mean
        elif norm_mode=='cmvn':
            curr_mean = np.mean(feat_in,axis=axis_cmvn)
            feat_mean = np.expand_dims( curr_mean, axis=axis_cmvn)  
            curr_std = np.std(feat_in-feat_mean,axis=axis_cmvn)
            feat_std  = np.expand_dims( curr_std, axis=axis_cmvn) 
            return feat_mean,feat_std
        else:
            raise Exception('\n\nnorm_mode Error: \nCurrently norm_mode={}\nYou have to choose between norm_mode="cmn" or norm_mode="cmvn" \n\n'.format(norm_mode))
    def add_2_deltas(feat_in, width_in):
        ## Compute the delta features 
        delta_1 = librosa.feature.delta(feat_in, width=width_in)
        delta_2 = librosa.feature.delta(feat_in, width=width_in, order=2)
        return delta_1, delta_2
    def seconds_to_frameidx(secs, hop_length, n_fft, sr):
        num_samples = secs*sr
        frame_idx = (num_samples-n_fft)/hop_length
        return int(frame_idx)+1
    def Hz_to_freqidx(hz_in, num_freq_bins, sr):
        num_freq_res = (sr/2)/num_freq_bins
        return int(hz_in / num_freq_res)
    def frameidx_to_samples(frameidx, hop_length, n_fft, sr):    return int(((frameidx-1)*hop_length+n_fft))
    def frameidx_to_seconds(frameidx, hop_length, n_fft, sr):    return int(((frameidx-1)*hop_length+n_fft)/sr)
    def frameidx_to_minsec(frameidx, hop_length, n_fft, sr):    
        Total_secs = frameidx_to_seconds(frameidx, hop_length, n_fft, sr)
        num_sec=Total_secs%60
        num_min=(Total_secs-num_sec)/60
        return [int(num_min),int(num_sec)]

    ## STFT, LPS
    def wav2stft(x_in, spect_det=None, mode="librosa",
        n_fft=None, win_length=None, hop_length=None, nfft=None, winstep=None, winlen=None,
        pad_mode=False, fs=None, center_in=True, _window='hann'):
        """
            x_in       : shape (length)
            win_length : length of window function (111)
        """
        if spect_det is not None:
            n_fft, win_length, hop_length, n_mels, n_mfcc, nfft, winstep, winlen, nfilt = spect_det
            print('\n\n wav2stft():\
                    including spect_det in the function is deprecated, \
                    remove now\n\n')
        if mode == "librosa":
            """
                librosa.stft(y, n_fft=2048, hop_length=None, win_length=None, 
                    window='hann', center=True, dtype=<class 'numpy.complex64'>, 
                    pad_mode='reflect')
                
                ## How to use other window functions
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
                For Gaussian : window_fx = scipy.signal.get_window(('gaussian',std_dev), win_length)
                
                ## to specify filter window
                window_fx = scipy.signal.get_window(window_fx, win_length) # , window=window_fx
                
                ## To get the nearest power of 2
                if n_fft==None: 
                  n_fft = int( 2**math.ceil(math.log(win_length,2)) )
            """
            x_tmp = x_in+0.
            if pad_mode: x_tmp = get_paddedx_stft(x_tmp,n_fft)
            x_stft = librosa.stft(y=x_tmp, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=center_in, window=_window)
        elif mode=="scipy": # This one we follow qd_gatech version of window, scaling, size
            # f, t, Zxx = scipy.signal.stft(x_tmp, fs=fs, window='hann', nperseg=win_length, noverlap=None, nfft=None, 
            #                   detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
            #   window='hann'
            #   window='hamming'
            ## Refer to .../site-packages/scipy/signal/spectral.py
            ## under stft -> _spectral_helper -> search for {scaling,scale}
            x_tmp = x_in+0.
            if pad_mode: x_tmp = get_paddedx_stft(x_tmp,n_fft)

            hamming_window=scipy.signal.hamming(win_length)
            curr_scale=1/sum(hamming_window)**2
            curr_scale=np.sqrt(curr_scale)

            _, _, x_stft = scipy.signal.stft(x=x_tmp, fs=fs, window=hamming_window, nperseg=win_length, 
                noverlap=hop_length, nfft=n_fft, boundary=None, padded=False)
            x_stft/=curr_scale
        elif mode=="tf": # This one we follow qd_gatech version of window, scaling, size
            x_tmp = x_in+0.
            if pad_mode: x_tmp = get_paddedx_stft(x_tmp,n_fft)
            ## Make the wave into tf format
            # x_tmp = x_in.astype(np.float32)
            x_tmp = tf.reshape(x_tmp.astype(np.float32), [1,-1])
            ## Get stft
            if _window=='hann':
                curr_window=tf.signal.hann_window
            stft_out = tf.signal.stft(x_tmp,
                frame_length=n_fft, 
                frame_step=hop_length,
                window_fn=curr_window,
                pad_end=False)
            # ## Get magnitude/phase
            # abs_out=tf.math.abs(stft_out)
            # phase_out=tf.math.angle(stft_out)
            # ## Get lps
            # eps1e6=tf.constant(1e-6)
            # lps_out = tf.math.log(abs_out + eps1e6)
            x_stft = stft_out.numpy()[0].T
        return x_stft
    def wav2abs(x_in, mode="librosa",
        n_fft=None, win_length=None, hop_length=None, nfft=None, winstep=None, winlen=None,
        spect_det=None, pad_mode=False, fs=None, center_in=True, _window='hann'):
        if spect_det is not None:
            n_fft, win_length, hop_length, n_mels, n_mfcc, nfft, winstep, winlen, nfilt = spect_det
            print('\n\n wav2abs():\
                    including spect_det in the function is deprecated, \
                    remove now\n\n')
        _kwargs = {
            "spect_det":spect_det,
            "mode":mode,
            "n_fft":n_fft,
            "win_length":win_length,
            "hop_length":hop_length,
            "nfft":nfft,
            "winstep":winstep,
            "winlen":winlen,
            "pad_mode":pad_mode,
            "fs":fs,
            "center_in":center_in,
            "_window":_window,
        }
        x_stft = wav2stft(x_in, **_kwargs)
        return np.abs(x_stft)
    def wav2LPS(x_in, spect_det=None, mode="librosa",
        n_fft=None, win_length=None, hop_length=None, nfft=None, winstep=None, winlen=None,
        pad_mode=False, fs=None, center_in=True, _window='hann'):
        """
            x_in       : shape (length)
            win_length : length of window function (111)
        """
        # n_fft, win_length, hop_length, n_mels, n_mfcc, nfft, winstep, winlen, nfilt = spect_det
        if mode == "librosa":
            """
                librosa.stft(y, n_fft=2048, hop_length=None, win_length=None, 
                    window='hann', center=True, dtype=<class 'numpy.complex64'>, 
                    pad_mode='reflect')
                
                ## How to use other window functions
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
                For Gaussian : window_fx = scipy.signal.get_window(('gaussian',std_dev), win_length)
                
                ## to specify filter window
                window_fx = scipy.signal.get_window(window_fx, win_length) # , window=window_fx
                
                ## To get the nearest power of 2
                if n_fft==None: 
                  n_fft = int( 2**math.ceil(math.log(win_length,2)) )
            """
            x_tmp = x_in+0.
            if pad_mode: 
                spect_det = n_fft, win_length, hop_length, 0, 0, nfft, winstep, winlen, 0
                x_tmp = get_paddedx_stft(x_tmp,n_fft)
            x_stft = librosa.stft(y=x_tmp, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=center_in, window=_window)
            x_stft_mag = np.abs(x_stft);    
            x_stft_pha = np.angle(x_stft);  
            # x_PS = (x_stft_mag**2)/n_fft
            # x_PS = np.where(x_PS == 0,np.finfo(float).eps,x_PS) # replace 0 with eps
            # x_LPS = 10*np.log10(x_PS)
        elif mode=="scipy": # This one we follow qd_gatech version of window, scaling, size
            x_tmp = x_in+0.
            if pad_mode: x_tmp = get_paddedx_stft(x_tmp,n_fft)
            hamming_window=scipy.signal.hamming(win_length)
            curr_scale=1/sum(hamming_window)**2
            curr_scale=np.sqrt(curr_scale)
            _, _, x_stft = scipy.signal.stft(x=x_tmp, fs=fs, window=hamming_window, nperseg=win_length, noverlap=hop_length, nfft=n_fft, boundary=None, padded=False)
            x_stft/=curr_scale
            x_stft_mag = np.abs(x_stft); 
            # x_stft_mag/=curr_scale
            x_stft_pha = np.angle(x_stft);  
        elif mode=="tf": # This one we follow qd_gatech version of window, scaling, size
            x_tmp = x_in+0.
            if pad_mode: x_tmp = get_paddedx_stft(x_tmp,n_fft)

            ## Make the wave into tf format
            # x_tmp = x_in.astype(np.float32)
            x_tmp = tf.reshape(x_tmp.astype(np.float32), [1,-1])
            ## Get stft
            if _window=='hann':
                curr_window=tf.signal.hann_window
            stft_out = tf.signal.stft(x_tmp,
                frame_length=n_fft, 
                frame_step=hop_length,
                window_fn=curr_window,
                pad_end=False)
            ## Get magnitude/phase
            abs_out=tf.math.abs(stft_out)
            phase_out=tf.math.angle(stft_out)
            ## Get lps
            eps1e6=tf.constant(1e-6)
            lps_out = tf.math.log(abs_out + eps1e6)
            return abs_out.numpy()[0].T, phase_out.numpy()[0].T, stft_out.numpy()[0].T, lps_out.numpy()[0].T
        ## Prevent issues with log 0
        # x_stft_mag = np.where(x_stft_mag == 0,np.finfo(float).eps,x_stft_mag) # replace 0 with eps
        # x_stft_mag += np.finfo(float).eps
        x_stft_mag += 1e-6
        ## Log
        # x_LPS = 2*np.log10( np.abs(x_stft_mag) )
        x_LPS = 2*np.log( np.abs(x_stft_mag) )
        return x_stft_mag, x_stft_pha, x_stft, x_LPS
    ## i-STFT
    def spect2wav(len_x, S_Mag, S_Phase, spect_det, mode="librosa", 
        sr_scipy=8000, _window='hann'):
        """
            len_x     : expected length of the output x_istft
            S_Mag     : magnitude of the spectrogram/LPS
                        'Spect' : np.abs(y_stft)
                        'LPS'   : np.log(np.abs(y_stft))
            S_Phase   : phase of the spectrogram 
                        'Spect' : np.angle(y_stft)
            spect_det : [hop_length, win_length]
            mode      : 'librosa', 'scipy'
        """
        n_fft, win_length, hop_length, n_mels, n_mfcc, nfft, winstep, winlen, nfilt = spect_det
        if mode=='librosa':
            # librosa.istft(stft_matrix, hop_length=None, win_length=None, window='hann', 
            #               center=True, dtype=<class 'numpy.float32'>, length=None)
            S = S_Mag*np.exp(1.j*S_Phase)
            wav_out=librosa.istft(S, length=len_x, hop_length=hop_length, win_length=win_length)
        elif mode=='scipy':

            ## Scaled to window size
            hamming_win=scipy.signal.hamming(win_length)
            curr_scale=1/sum(hamming_win)**2
            curr_scale=np.sqrt(curr_scale)

            S_Mag*=curr_scale
            S=S_Mag*np.exp(1.j*S_Phase)
            # scipy.signal.istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
            _, wav_out=scipy.signal.istft(S, fs=sr_scipy, window=hamming_win, nperseg=win_length, noverlap=hop_length, nfft=n_fft, boundary=False)
            wav_out = np.append(wav_out,np.zeros((len_x-len(wav_out),)))
        elif mode=='tf':
            if _window=='hann':
                curr_window=tf.signal.hann_window

            ## computing stft.tf <- mag.tf and phase.tf
            array_0 = np.zeros_like(S_Mag.T)
            abs_out_complex = tf.dtypes.complex(S_Mag.T,array_0)
            phase_out_complex = tf.dtypes.complex(array_0,S_Phase.T)
            # stft_out = abs_out_complex * tf.math.exp(phase_out_complex)

            istft=tf.signal.inverse_stft(
                abs_out_complex * tf.math.exp(phase_out_complex), 
                frame_length=n_fft, 
                frame_step=hop_length,
                window_fn=tf.signal.inverse_stft_window_fn(frame_step=hop_length,forward_window_fn=curr_window),)

            wav_out = istft.numpy()
            crop_len = n_fft//2
            wav_out = wav_out[crop_len:]
            wav_out = wav_out[:len_x]

        return wav_out
    ## MEl
    def MFCCvsMEL(feat_in, n_mfcc, dct_type):
        """
        dct_type=2 : mel to mfcc
        dct_type=3 : mfcc to mel
        """
        n,m=feat_in.shape
        feat_out=np.zeros_like(feat_in)
        for i in range(m):
            curr_frame=feat_in[:,i]
            new_frame=scipy.fftpack.dct(curr_frame, type=dct_type, n=n, axis=-1, norm='ortho')
            feat_out[:,i]=new_frame
        feat_out_liftered = python_speech_features.lifter(feat_out.T,L=22).T
        feat_out[n_mfcc:]=0
        feat_out_liftered[n_mfcc:]=0
        return feat_out, feat_out_liftered
    
    ## Power 
    def calc_power(_x_in):
        assert len(_x_in.shape)==1, 'it should be waveform i.e. vector '
        power_out = np.linalg.norm(_x_in)**2/len(_x_in)

        return power_out
    def power_norm(_wav_in, _des_power, _prevent_clipping=True):

        wav_Power   = calc_power(_wav_in)

        ## Calc output power
        wav_pnorm = _wav_in*(_des_power/wav_Power)**.5
        renorm_Pow=calc_power(wav_pnorm)

        ## To prevent clipping
        if _prevent_clipping:
            max_val=np.max(np.abs(wav_pnorm))
            if max_val>1:
                print(max_val)
                wav_pnorm /= max_val
        
        return wav_pnorm
    
##########################################################################################################################################################
## MCT - Denoising
if True:
    def get_alpha_SNR(clean_P,noise_P, _snr, _mode):
        ## Calc alpha, assuming they are of the same power
        if _mode=="CP":
            k=noise_P/clean_P
            k*=10**(_snr/10)
            k**=.5
            alpha_out=k/(1+k)
            beta_out=1-alpha_out
        if _mode=="LL":
            k = 10 ** (_snr/10)
            alpha_out = (k/(k+1))**0.5
            beta_out = 1/(k+1)**0.5
        return alpha_out,beta_out
    ## v2 : shifted reverb, and power normalisation, clip if specified to
    def augment_noisy_wave(  _clean_WAV, _noise_WAV, _snr,  allow_clipping=True): 
        ## y = a*clean + b*noise

        num_samples_clean=float(np.max(_clean_WAV.shape))
        num_samples_noise=float(np.max(_noise_WAV.shape))
        ## Repeat the noise until it is of same length as _clean_WAV
        factor_tile=np.ceil(num_samples_clean/num_samples_noise)
        factor_tile=int(factor_tile)
        _noise_WAV=np.tile(_noise_WAV,factor_tile)
        _noise_WAV=_noise_WAV[:len(_clean_WAV)]

        ## Power 
        clean_P=calc_power(_clean_WAV)
        _noise_WAV_pnorm=power_norm(_noise_WAV, clean_P, allow_clipping)
        noise_P=calc_power(_noise_WAV_pnorm)

        ## Make Noisy
        __alpha,__beta=get_alpha_SNR(clean_P,noise_P, _snr, _mode="LL")
        _noisy_WAV_pnorm=__alpha*_clean_WAV + __beta*_noise_WAV_pnorm

        if not allow_clipping:
            num_clipped=np.sum(_noisy_WAV_pnorm>1)
            num_clipped+=np.sum(_noisy_WAV_pnorm<-1)
            _noisy_WAV_pnorm[_noisy_WAV_pnorm>1]=1.
            _noisy_WAV_pnorm[_noisy_WAV_pnorm<-1]=-1.
            # print('Clipped {} samples'.format(num_clipped))
        return _noisy_WAV_pnorm
    def augment_reverb_wave( _clean_WAV, _RIR_WAV,   pow_eq=True): 
        def shift(xs,n):
            e = np.empty_like(xs)
            if n>= 0:
                e[:n]=0.
                e[n:]=xs[:-n]
            else:
                e[n:]=0.
                e[:n]=xs[-n:]
            return e
        ## Make Noisy
        _reverb_WAV=scipy.signal.fftconvolve(_clean_WAV, _RIR_WAV, mode='full')
        # pdb.set_trace() # !import code; code.interact(local=vars())
        p_max=np.argmax(_RIR_WAV)
        _reverb_WAV=shift(_reverb_WAV, -p_max)
        _reverb_WAV = _reverb_WAV[:len(_clean_WAV)]
        ## Power 
        if pow_eq:
            clean_P=calc_power(_clean_WAV)
            reverb_P=calc_power(_reverb_WAV)
            _reverb_WAV = power_norm(_reverb_WAV, clean_P)
        return _reverb_WAV
