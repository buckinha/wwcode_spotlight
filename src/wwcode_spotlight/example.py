import sklearn.datasets
import numpy as np
import boto3
import os
import tempfile
from sklearn.neural_network import MLPClassifier
import pickle
import time


BUCKET = 'wwcode-spotlight'

X_FNAME = 'X.npy'
Y_FNAME = 'y.npy'
MODEL_FNAME = 'model.pickle'

N_SAMPLES = 10000
TEST_SPLIT_N = 8000


def clear_s3_files():
    s3 = boto3.resource("s3")
    obj = s3.Object(BUCKET, X_FNAME)
    obj.delete()
    obj = s3.Object(BUCKET, Y_FNAME)
    obj.delete()
    obj = s3.Object(BUCKET, MODEL_FNAME)
    obj.delete()


def write_file_to_s3(fname):
    s3 = boto3.resource('s3')
    s3_fname = os.path.split(fname)[-1]
    s3.Object(BUCKET, s3_fname).put(Body=open(fname, 'rb'))


def download_file_from_s3(s3_fname, dst):
    s3 = boto3.client('s3')
    s3.download_file(BUCKET, s3_fname, dst)


def process_data():
    """
    In this function, we're pretending to 'pre-precess' the data. In fact, we're just creating some data and saving
    it to S3.

    :return:
    """

    X, y = sklearn.datasets.make_moons(n_samples=N_SAMPLES)

    temp_dir = tempfile.mkdtemp()

    x_path = os.path.join(temp_dir, X_FNAME)
    y_path = os.path.join(temp_dir, Y_FNAME)

    np.save(x_path, X)
    np.save(y_path, y)

    write_file_to_s3(x_path)
    write_file_to_s3(y_path)


def build_model():
    # download the data; ideally, you'd configure these paths with a config file or some such
    temp_dir = tempfile.mkdtemp()
    x_path = os.path.join(temp_dir, X_FNAME)
    y_path = os.path.join(temp_dir, Y_FNAME)

    download_file_from_s3(X_FNAME, x_path)
    download_file_from_s3(Y_FNAME, y_path)

    X = np.load(x_path)[:TEST_SPLIT_N]
    y = np.load(y_path)[:TEST_SPLIT_N]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(X, y)

    # serialize the model
    model_path = os.path.join(temp_dir, MODEL_FNAME)
    pickle.dump(clf, open(model_path, 'wb'))

    # upload the model
    write_file_to_s3(model_path)


def score_model():
    # download the data; ideally, you'd configure these paths with a config file or some such
    temp_dir = tempfile.mkdtemp()
    x_path = os.path.join(temp_dir, X_FNAME)
    y_path = os.path.join(temp_dir, Y_FNAME)
    model_path = os.path.join(temp_dir, MODEL_FNAME)

    download_file_from_s3(X_FNAME, x_path)
    download_file_from_s3(Y_FNAME, y_path)
    download_file_from_s3(MODEL_FNAME, model_path)

    X = np.load(x_path)[TEST_SPLIT_N:]
    y = np.load(y_path)[TEST_SPLIT_N:]

    clf = pickle.load(open(model_path, 'rb'))

    predictions = clf.predict(X)
    accuracy = 1 - np.sum(np.abs(y - predictions)) / y.shape[0]
    print(f"Accuracy: {accuracy}")


def build_model_longer():
    time.sleep(30)
    build_model()


def test_do_all():
    clear_s3_files()
    process_data()
    build_model()
    score_model()

