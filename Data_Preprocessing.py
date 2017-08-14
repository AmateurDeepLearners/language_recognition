#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os, sys, getopt
import os.path

def create_list(filename):
    first_line = open(filename, 'r').readline()
    #print len(first_line.split(",")), first_line
    if len(first_line.split(",")) == 2: languages =True
    else: languages = False

	#Default arguments here:"    
    args = [filename, str, '#', ",", None, 0, None, True]  
    
    if first_line == 'Sample Filename,Language\r\n': args[5] = 1
     
    if languages==True:
        X,Y = np.loadtxt(*args)
        return X,Y
    else:  
        return np.loadtxt(*args)


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """ short time fourier transform of audio signal, parameters: 
    1. input from wav-file, 
    2. number of bins taken for one intervall"""
    # normalization function (Gauss-like), determines how much each value is 
    #taken into account
    win = window(frameSize)

    # number of bins between two starting points of one intervall
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))                        
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)            
    # cols for windowing: defining how many intervalls 
    #for in the number of samples, importance of overlapfactor!    
    cols = int(np.ceil( (len(samples) - frameSize) / float(hopSize))) + 1    
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))                            

    #creation of an array containing the single frames. Each frame is one row. 
    #The number of columns is the number of frames
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), 
                                      strides=(samples.strides[0]*hopSize, 
                                               samples.strides[0])).copy()
   
    # assigning weights to the single data points. Points in the middle are 
    #more important. Neglect border effects in the fouriere transform
    frames *= win                                                                    
    return np.fft.rfft(frames)  # fourier transform of the signal  
    
def logscale_spec(spec, sr=44100, alpha=1.1, f0=0.9, fmax=1):
    """ Create frequency spectrum with no logarithmic scales (confusing name), 
    Parameter: 1. Fourier transform data, 
    2. Sample rate at which the audio was collected, 
    3. Parameter for data augmentation, changes frequency scale, 4. , 5. """
    # Only use first 256 values of the fourier transform
    #print spec.shape
    spec = spec[:, 0:128]
    # Get number of intervalls and number of frequency bins
    timebins, freqbins = np.shape(spec)
    # Create scale for the warp function
    scale = np.linspace(0, 1, freqbins)                                                

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    # Warping function for data augmentation: 
    #frequency is mapped on a warped frequency
    scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
    scale *= (freqbins-1)/max(scale)

    # new spectrum which will be returned
    newspec = np.complex128(np.zeros([timebins, freqbins]))                            
    # create frequencies which are checked
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./16000.)[:freqbins+1])       
    # frequencies which will be returned
    freqs = [0.0 for i in range(freqbins)]                                            
    # total frequency
    totw = [0.0 for i in range(freqbins)]

    # exact routine is not really important. It returns the frequency and the 
    #shifted spectrum
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))
           
            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down
            
            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up
    
    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]
	#print freqs
    return newspec, freqs # Ruturn of frequency and spectrum (not logarithmic)

def plotstft(audiopath, channel=0, name='tmp.png', alpha=1, 
             offset=0, im_length=768):
    """Needs following arguments: 
    1. wav-input, 
    2. number of bins during the audio , 
    3. channel from which we need the samples, 
    4. image output, 
    5. FIXME, 
    6. offset defines starting point of the intervalls. 
       Usefull if beginning of the audio is silent"""

    # returns rate and data from wav-file
    samplerate, samples = wav.read(audiopath)
    binsize=int(samplerate/45.)
    if len(samples.shape) == 2:            # take data from the proper channel
        samples = samples[:, channel]
    #print str(len(samples)) + " " +audiopath
    # short time fourier transform. 
    # Returns fourier trasform for each time intervall
    s = stft(samples, binsize)              
    # get frequencies out of the fourier transform of the signal. 
    #Scale:logarithmic
    sshow, freq = logscale_spec(s, sr=samplerate, alpha=alpha)       
    # dump first two intervalls, maybe because of log10?
    sshow = sshow[2:, :]                                                    
    # amplitude to decibel, assign log scale to not have big differences in 
    ims = 20.*np.log10(np.abs(sshow)/10e-6)                                        

    # get number of intervalls and number of frequency bins
    timebins, freqbins = np.shape(ims)    

    ims = np.transpose(ims)# transpose the array
    # 0-11kHz: this intervall is fixed and will be used for each audiofile, 
    # ~9s interval, starts depending on offset
    ims = ims[0:128, offset:offset+im_length]

    if (ims.shape !=(128, im_length)):
        print ("\n Ooops something went wrong. The Spectrogram of "+audiopath
              + " has shape "+ str(ims.shape))                                            

    image = Image.fromarray(ims) # save image from array
    image = image.convert('L')    
    image.save(name)

def convert_mp3_wav(inPath, outPath):
    """convert mp3 file at inPath to wav and save result at outPath
    please make sure you have mpg123 installed
    """   
    inPath  = os.path.expanduser(inPath)
    outPath = os.path.expanduser(outPath)
    os.system('mpg123 -w ' + outPath + ' ' + inPath + ' 2> /dev/null')


#TODO add option to augment the data
#TODO if channels conatin differing data, add option tu use other channel
def mp3toSpectrogram(inPath, outPath, 
                     keepWav = False, wavPath = "tempProcessing.wav", 
                     verbose = False):
    inPath  = os.path.expanduser(inPath)
    outPath = os.path.expanduser(outPath)
    convert_mp3_wav(inPath,wavPath)
    plotstft(wavPath, channel=0, name=outPath, alpha=1, offset=0)
    if verbose: print("used " + inPath + " to create "+ outPath)
    if not keepWav:
        os.remove(wavPath)
        if verbose: print(".wav file discarded.")
    else:
        if verbose: print(".wav file kept at "+wavPath)
def split_and_augment(csv_path, num_augments, out_path, frac_validation=0.2):
    """For the datasets from topcoder:
    splits the files provided in a csv file into training and testing sets
    of augmented spectrograms"""
  
    # get list of audiofiles and languages from csv
    mp3_files, languages = create_list(os.path.expanduser(csv_path))
  
    # the audiofiles have to be in a folder called mp3 next to the csv
    mp3_dirname = os.path.dirname(csv_path)+"/mp3"
    
    # create folder structure for keras
    os.makedirs(out_path)
    for lang in set(languages):
        os.makedirs(out_path+"/training/"+"".join(lang.split()))
        os.makedirs(out_path+"/validation/"+"".join(lang.split()))

    # create a bool array to split into training and validation data
    num_samples = len(mp3_files) 
    size_validation = int(frac_validation*num_samples)
    size_training = num_samples-size_validation                                           
    print("validation samples ",size_validation)                                
    print("training samples", size_training)
    validation_index = np.array([1] * size_validation + [0] * size_training)               
    validation_index = validation_index.astype(bool)               
    np.random.shuffle(validation_index)

    # create spectrograms
    for mp3,lang,validation in zip(sorted(mp3_files),languages,validation_index):
        convert_mp3_wav(mp3_dirname+"/"+mp3, "temp.wav")
        if validation: folder = "validation/"
        else:          folder = "training/"
    
        #print(folder)
        # for every file create multiple augmented spectrograms
        for augment_idx in range(num_augments):
            alpha = np.random.uniform(0.9, 1.1)
            offset = np.random.randint(90)
            #print(alpha,offset)
            plotstft("temp.wav", channel=0, name=out_path+"/"
                                                 +folder
                                                 +"".join(lang.split())
                                                 +"/"+mp3[:-4]
                                                 +"_"+str(augment_idx)+'.png',
                     alpha=alpha, offset=offset)

if __name__ =="__main__":
    #Reproduzierbarkeit des Datensets durch festgelegten Random Seed
    np.random.seed(seed = 121922)		
    
    # Hier muss als 1.Argument die Datei mit den Labels und Dateinamen uebergeben 
    # werden.
    # Die Ordnrstruktur muss dabei so gewaehlt sein, dass im gleichen Verzeichnis 
    # wie der Datei mit Labels und Dateinamen sich ein Ordner namens mp3 befindet 
    # mit den mp3-Dateien.
    # Als 2. Argument wird die Anzahl an erstellten Spektrogrammen pro mp3-Datei 
    # uebergeben
    # Als 3. Argument wird der Zielordner des Datensets angegeben
    
    split_and_augment("../data/topcoderSpokenLanguages1Data/training/trainingset.csv"
                    , 10, "./data/keras_topcoder_1")
    
















