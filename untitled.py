def auxi():
    # !pip install pandas
    # !pip install tensorflow
    # !pip install scikeras
    
    
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    
    # load and format input and out dataset
    X_dataframe = read_csv("input.csv", header=None)
    X = X_dataframe.values
    y_dataframe = read_csv("output.csv", header=None)
    y = [item[0] for item in y_dataframe.values]
    
    # define and fit the model to the training datasets
    tf.random.set_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    )
    
    model.fit(np.array(X), np.array(y), validation_split=0.33, batch_size=32, epochs=100, verbose=0)
    return model