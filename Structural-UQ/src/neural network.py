# neural_network_model_tf.py


import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def train_surrogate_tf(
    f_data: np.ndarray,
    u_data: np.ndarray,
    num_hidden_layers: int = 2,
    num_neurons: int = 10,
    lr: float = 0.01,
    epochs: int = 1000,
    batch_size: int = 32,
    dropout_rate: float = 0.2,
    verbose: int = 0,
    plot_loss: bool = True,
):
    """
    Trains a TensorFlow/Keras neural network to approximate the FEM mapping f → u.

    Parameters
    ----------
    f_data : np.ndarray
        Input forces (n_samples, n_dofs)
    u_data : np.ndarray
        Output displacements (n_samples, n_dofs)
    num_hidden_layers : int, optional
        Number of hidden layers.
    num_neurons : int, optional
        Neurons per layer.
    lr : float, optional
        Learning rate for Adam optimizer.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size for training.
    dropout_rate : float, optional
        Dropout rate between layers.
    verbose : int, optional
        Verbosity level for training (0 = silent, 1 = per epoch).
    plot_loss : bool, optional
        Whether to plot training loss.

    Returns
    -------
    model : tf.keras.Model
        Trained model.
    history : tf.keras.callbacks.History
        Keras training history object.
    inp_scaler, out_scaler : sklearn.preprocessing.StandardScaler
        Fitted input/output scalers for normalization.
    """

    # --- Scaling ---
    inp_scaler = StandardScaler()
    out_scaler = StandardScaler()

    f_scaled = inp_scaler.fit_transform(f_data)
    u_scaled = out_scaler.fit_transform(u_data)

    # --- Model Definition ---
    model = Sequential()
    model.add(tf.keras.Input(shape=(f_scaled.shape[1],)))

    for _ in range(num_hidden_layers):
        model.add(Dense(num_neurons, activation="tanh"))
        model.add(Dropout(dropout_rate))

    model.add(Dense(u_scaled.shape[1]))

    # --- Compile and Train ---
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=MeanSquaredError(),
        metrics=[MeanSquaredError()],
    )

    history = model.fit(
        f_scaled,
        u_scaled,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # --- Plot Training Loss ---
    if plot_loss:
        plt.figure(figsize=(9, 6))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss ({epochs} epochs)")
        plt.legend()
        plt.grid(True)
        plt.show()

    return model, history, inp_scaler, out_scaler


def predict_tf(model, f_data: np.ndarray, inp_scaler, out_scaler):
    """
    Predicts displacements using the trained TensorFlow model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    f_data : np.ndarray
        Force input data (n_samples, n_dofs)
    inp_scaler, out_scaler : sklearn.preprocessing.StandardScaler
        Scalers used during training.

    Returns
    -------
    np.ndarray
        Predicted displacements (n_samples, n_dofs)
    """
    f_scaled = inp_scaler.transform(f_data)
    u_pred_scaled = model.predict(f_scaled, verbose=0)
    u_pred = out_scaler.inverse_transform(u_pred_scaled)
    return u_pred
