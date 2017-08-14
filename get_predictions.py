import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image



#img_width, img_height = 768,256

img_width, img_height = 350,128

def get_validation_score(directory, model,
                         verbose =  False, batch_size = 20,
                         per_class = False, percentage = False, 
                         save_predictions = True):
    """Gives an accuracy score for files in directory.
    directory should look like this:
    directory  
    |__class1  
    |  |__file1.png  
    |  |__file2.png  
    |  |__...  
    |__class2  
    |  |_...  
    ...  
    
    """ 
    classes = [name for name in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, name))]
      
    if verbose: print("found "+str(len(classes))+" classes")

    if per_class: results = []
    else: correct, total = 0,0
    for i,c in enumerate(sorted(classes)):
        print(i,c)
        files = os.listdir(directory+"/"+c)
        first = True
        num = len(files)
        predictions =[]
        for j,f in enumerate(files):     
            img = image.load_img(directory+"/"+c+"/"+f, 
                                 target_size=(img_width, img_height),
                                 grayscale=False)
          
          
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
          
            if first:
                ims = x[:,:,:,:1]
                first = False
            else:
                ims = np.vstack([ims,x[:,:,:,:1]])
            if j%(batch_size)==0 or j == (num-1) :
                predictions.append(model.predict_classes(ims, 
                                                         batch_size=batch_size))
                first = True

        predictions = np.hstack(predictions)#model.predict_classes(ims, batch_size = batch_size)
        print(predictions.shape)
        if verbose: print(predictions)
        if save_predictions: np.savetxt("predictions/"+c,predictions,fmt="%d")

        if per_class:            
            correct = np.sum(np.equal(predictions, [i]*len(files))) 
            total = len(files)
            if percentage: results.append(correct*1./total)
            else:          results.append([correct,total])
            
            if verbose: print(results[-1],c)
        
        else:
            correct += np.sum(np.equal(predictions, [i]*len(files)))
            total += len(files)


    if per_class: return results,sorted(classes)
    else: 
        if percentage: return correct*1./total
        else:          return str(correct)+"/"+str(total)
            
def get_prediction_vectors(class_dir,model, batch_size=30):
    """returns predictions for all samples in class_dir using model""" 
    files = os.listdir(class_dir)
    first = True
    num = len(files)
    print(num, class_dir)
    predictions = []
    for i,f in enumerate(files):     
        img = image.load_img(class_dir+"/"+f, 
                             target_size=(img_width, img_height),
                             grayscale=False)
      
      
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
      
        if first:
            ims = x[:,:,:,:1]
            first = False
        else:
            ims = np.vstack([ims,x[:,:,:,:1]])
        if i%100 == 0:
            print(i, num, class_dir)
        if i%(batch_size)==0 or i == (num-1) :
            predictions.append(model.predict(ims, batch_size = batch_size))
            first = True
   
    predictions = np.vstack(predictions)
    print(predictions.shape)
    return predictions, files
    

if __name__ == "__main__":
    
    model = load_model("voxforge_weights-5-Sprachen.33-0.85.hdf5")
    validation_folder =  "validation_voxforge/"
    
    
    classes = ["Italian"] #["French", "Spanish"]#["English", "French", "German"]
    
    for c in classes:
        class_dir = validation_folder+c	
        v,f = get_prediction_vectors(class_dir,model, batch_size = 50)
        np.savetxt("predictions_voxforge/vectors_"+c, v)
        np.savetxt("predictions_voxforge/files_"+c, f, fmt = "%s")


    results, classes = get_validation_score(validation_folder,model, 
                                              per_class = True, batch_size=30, 
                                              verbose= True)
    np.savetxt("results.dat", results, fmt = "%d")
    np.savetxt("classes.dat", classes, fmt = "%s")
