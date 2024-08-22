import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_traj(dataset_path, train_test = True):
    
    data = np.load(dataset_path)
    scaler = MinMaxScaler()
    
    if train_test:
        X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        return X_train_scaled, X_test_scaled
    
    else:
       X_train_scaled = scaler.fit_transform(data)
       return X_train_scaled     
    
