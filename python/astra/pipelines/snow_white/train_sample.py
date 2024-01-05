import pandas
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from astra.utils import expand_path

PIPELINE_DATA_DIR = expand_path("$MWM_ASTRA/pipelines/snow_white")

def train_classifier(n_jobs=1, input_dataset_basename="file_for_training.csv", output_basename="training_file"):

    dataset = pandas.read_csv(os.path.join(PIPELINE_DATA_DIR, input_dataset_basename))#'additional_results'
    array = dataset.values
    X = array[:,2:1753]
    Y = array[:,1]
    knn = RandomForestClassifier(n_estimators=500, criterion="log_loss",class_weight="balanced",min_samples_split=4,min_samples_leaf=2,n_jobs=20,max_features=None,bootstrap=True,verbose=0,max_samples=1000)
    weights=[]
    for xxx in range(np.size(Y)):
        if Y[xxx]=="CV":
            weights.append(10)
        elif Y[xxx]=="DA_MS":
            weights.append(0.5)
        elif Y[xxx]=="DB_MS":
            weights.append(0.5)
        else:
            weights.append(2)
        #print("here...")
        #if Y[xxx]=="DBAZ" or Y[xxx]=="DABZ" or Y[xxx]=="DBZ" or Y[xxx]=="DAZ"or Y[xxx]=="DBA" or Y[xxx]=="DAe" or Y[xxx]=="DAH":
        #   weights.append(5)
        #else:
        #   weights.append(1)
    knn.fit(X, Y)#,sample_weight=weights)
    with open(os.path.join(PIPELINE_DATA_DIR, output_basename), 'wb') as f:
        pickle.dump(knn, f)
        #joblib.dump(knn, f)
