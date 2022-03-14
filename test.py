import tensorflow as tf
import numpy as np
import time

from sklearn.metrics import precision_recall_fscore_support

from model import Model

from dataloader.staticloader import StaticLoader

loader = StaticLoader()
X_train, y_train, X_test, y_test = loader.load_kdddata()

# Initialize
starttime = time.time()

model = Model(
    comp_hiddens=[60,30,15], comp_activation=tf.nn.tanh,
    est_hiddens=[4], est_dropout_ratio=0.5, est_activation=tf.nn.tanh,
    learning_rate=0.0001, epoch_size=200, minibatch_size=1024, random_seed=123
)

# Fit the training data to model
model.fit(X_train)

# Save fitted model to the directory
model.save("./fitted_model/")

# Restore saved model from dicrectory
model.restore("./fitted_model/")

# Evaluate energies
# (the more the energy is, the more it is anomary)
y_pred = model.predict(X_test)

# Energy thleshold to detect anomaly
anomaly_energy_threshold = np.percentile(y_pred, 80)
print(f"Energy thleshold to detect anomaly : {anomaly_energy_threshold:.3f}")

# Detect anomalies from test data
y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)

prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_flag, average="binary")
print(f" Precision = {prec:.4f}")
print(f" Recall    = {recall:.4f}")
print(f" F1-Score  = {fscore:.4f}")

endtime = time.time()
dtime = endtime - starttime

print("running time: %.4s s" % dtime)
