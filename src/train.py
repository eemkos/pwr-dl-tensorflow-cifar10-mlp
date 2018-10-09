import tensorflow as tf
import logging
from src.dataset import load_dataset


def train(model : tf.estimator.Estimator,
          nb_epochs : int,
          train_data_path : str,
          val_data_path : str,
          batch_size: int = 32):

    train_epoch_history = [model.evaluate(input_fn=lambda: load_dataset(train_data_path, shuffle=False))]
    validation_epoch_history = [model.evaluate(input_fn=lambda: load_dataset(val_data_path, shuffle=False))]
    for epoch in range(nb_epochs):
        model_spec = model.train(input_fn=lambda: load_dataset('data/train.tfrecords',
                                                               epochs=1,
                                                               shuffle=True,
                                                               batch_size=batch_size))

        train_epoch_history.append(model.evaluate(input_fn=lambda: load_dataset(train_data_path, shuffle=False)))
        validation_epoch_history.append(model.evaluate(input_fn=lambda: load_dataset(val_data_path, shuffle=False)))

        logging.info(f"EPOCH: {epoch}:\n"
                     f"\tval_loss: {validation_epoch_history[-1]['loss']}\n"
                     f"\ttrain_loss: {train_epoch_history[-1]['loss']}\n")

    return train_epoch_history, validation_epoch_history

