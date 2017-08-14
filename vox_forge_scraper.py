import numpy as np
import urllib
import urllib2
import os
import re 
import random
import shutil
import tarfile
import scipy.io.wavfile as wav
import pickle

from Data_Preprocessing import plotstft
#os.chdir('D:\\voxforge speech files\\')#To change the current path
#refiles=open('speech_files_path.txt','w+')#Store all the download link


def download_rdn_tgz(url, path , used = []):
    """download a random tgz file that is not in used list from a voxforge url 
    and return its filename
    """
    page=urllib.urlopen(url)  
    html=page.read()  
    reg=r'href=".*\.tgz"'  
    tgzre=re.compile(reg)  
    tgzlist=re.findall(tgzre,html)  #Find all of the.Tgz files
    
    i = random.choice(tgzlist)
    while i in used:
        i = random.choice(tgzlist)
    
    filename=i.replace('href="','')
    filename=filename.replace('"','')
    print 'Downloading:'+filename # prompts the file being downloaded
    downfile=i.replace('href="',url)
    
    downfile=downfile.replace('"','') #Each file integrity
    req = urllib2.Request(downfile)  #Download the file 
    ur = urllib2.urlopen(req).read()
    open(path+"/"+filename,'wb').write(ur) #Download the tgz files
    return filename


def get_usable_wavs(path, min_samples = 16000*4 ,channel = 0, 
                    number_of_bins = 350):
    """determine which wav files are longer than the threshold and determine
    how long the maximum offset can be for the augmentations"""    
    usable = []
    allowed_offsets = []
    for f in os.listdir(path):
        #print f
        audiopath = path+"/"+f 
        samplerate, samples = wav.read(audiopath)
        bin_size = int(samplerate/45.)
        if len(samples.shape) == 2:         # take data from the proper channel
            samples = samples[:, channel]
        
        #print len(samples),min_samples, samplerate
        assert samplerate ==16000 , "Wrong samplerate in "+str(path)
        
        #take only the files that are longer than the required minimum
        if len(samples) > min_samples:
            usable.append(f)
            #determine how long the offset can be at the beginning of the file
            max_offst = np.max([len(samples)*2/ (bin_size)-(number_of_bins+10),
                               1]) 
            allowed_offsets.append(max_offst )
            
    #print(usable)
    return usable, allowed_offsets
        


def download_random_and_augment(language, url, folder, used):
    """Download a random file from a given voxforge.org url that is not yet in 
    the list of used files. Create augmented spectrograms from the sound files 
    in the archive. The 
    """
    min_samples = 16000*4
    bin_size = 16000/45
    number_of_bins = 350   
 
    png_path = folder+"/"+language+"/"
    
    
    os.mkdir("temp2")    
    fname = download_rdn_tgz(url, "temp2", used = used)
    
    tar = tarfile.open("temp2/"+fname, "r:gz")
    tar.extractall()
    tar.close()
    
    total_images_produced = 0
    
    if os.path.exists(fname[:-4]+"/wav"):
        usable, allowed_offsets = get_usable_wavs(fname[:-4]+"/wav", 
                                                  number_of_bins=number_of_bins, 
                                                  min_samples=min_samples )
        
        for f,max_offset in zip(usable, allowed_offsets):
            seconds_overlength = (max_offset/360.) *4 #This is only an approximation
            num_augments = np.min([ int(seconds_overlength * 5)+1, 20])
            #print(max_offset,seconds_overlength,num_augments)
    
            for augment_idx in range(num_augments):
                alpha = np.random.uniform(0.9,1.1)
                offset = np.random.randint(max_offset)
              
                #print(f[:-4])
                plotstft(fname[:-4]+"/wav/"+f, channel = 0, 
                         name = png_path + fname[:-4] +"_"+ f[:-4] + "_" + 
                                str(augment_idx)+".png",
                         alpha = alpha, offset = offset, 
                         im_length = number_of_bins) 
                total_images_produced +=1
    
    
    shutil.rmtree("temp2")
    shutil.rmtree(fname[:-4])
    return total_images_produced, fname


if __name__=="__main__":
    
    #fraction of files for validation and training
    fraction_validation = 0.2 
    fraction_training = 1 - fraction_validation
    
    path = "vox_forge_set_4s"
    
    languages = ["English", "French", "German", "Italian", "Spanish"]
    

    for language in languages:
        directory = path+"/training/"+language
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = path+"/validation/"+language
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    #urls for data from voxforge.org
    urls={
        "English":"http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/",
        "German":"http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/",
        "French":"http://www.repository.voxforge1.org/downloads/fr/Trunk/Audio/Main/16kHz_16bit/",
        "Italian":"http://www.repository.voxforge1.org/downloads/it/Trunk/Audio/Main/16kHz_16bit/",
        "Spanish":"http://www.repository.voxforge1.org/downloads/es/Trunk/Audio/Main/16kHz_16bit/",
    }
    
    #Lists of used files to avoid duplicates
    used = { lang:[] for lang in languages }

    # keep track of downloads LanguageV is the corresponding Validation folder
    number_of_files = {"English":0,
                       "French" :0,
                       "German" :0,
                       "Italian":0,
                       "Spanish" :0,
                       "EnglishV":0,
                       "FrenchV" :0,
                       "GermanV" :0,
                       "ItalianV":0,
                       "SpanishV" :0}

    
    #To load existing files, when there is an existing set
    #with open('used_files.pickle', 'rb') as handle:
    #    used = pickle.load(handle)
    #print(used, "\n\n\n") 
    #
    #with open('num_files.pickle', 'rb') as handle:
    #    number_of_files = pickle.load(handle)
    #print(number_of_files, "\n\n\n") 

    i=0
    while number_of_files.get(min(number_of_files, 
                                  key=number_of_files.get)) < 240000: 
        #print(number_of_files.get(min(number_of_files, 
        #                              key=number_of_files.get)))
	    
        #next class to download to
        next_to_fill = min(number_of_files, key=number_of_files.get)
        print(next_to_fill)

        #Save files to folder if the next files go into validation
        if next_to_fill[-1] == "V":
            validation_multiplier = 1/fraction_validation
            lang = next_to_fill[:-1]
            folder = path+"/validation"
        #Or into training
        else:
            validation_multiplier = 1/fraction_training
            lang = next_to_fill
            folder = path+"/training" 

        additional_ims, used_file = download_random_and_augment(lang, 
                                                                urls[lang], 
                                                                folder, 
                                                                used[lang])
        
        used[lang].append(used_file)
        number_of_files[next_to_fill] += additional_ims*validation_multiplier
        
        #print(next_to_fill)
        #print(used)
        if i%10 == 0: 
            with open('used_files.pickle', 'wb') as handle:
                pickle.dump(used, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
            with open('num_files.pickle', 'wb') as handle:
                pickle.dump(number_of_files, handle, 
                            protocol=pickle.HIGHEST_PROTOCOL)

        i+=1
    
    
    with open('used_files.pickle', 'wb') as handle:
        pickle.dump(used, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    with open('num_files.pickle', 'wb') as handle:
        pickle.dump(number_of_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(number_of_files)
