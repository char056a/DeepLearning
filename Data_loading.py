import numpy as np
import urllib.request
import tarfile
import io

# FASHION_MNIST_CLASSES = ["T-shirt/top","Trouser","Pullover","Dress","Coat", "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
#den her burde virke 
def FASHION_MNIST(flatten=True, one_hot=False):
    import numpy as np, urllib.request, gzip
    b = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    g = lambda n: gzip.decompress(urllib.request.urlopen(b+n).read())

    Xtr = np.frombuffer(g("train-images-idx3-ubyte.gz"), np.uint8, offset=16).reshape(-1, 28, 28) / 255.0
    ytr = np.frombuffer(g("train-labels-idx1-ubyte.gz"), np.uint8, offset=8)
    Xte = np.frombuffer(g("t10k-images-idx3-ubyte.gz"),  np.uint8, offset=16).reshape(-1, 28, 28) / 255.0
    yte = np.frombuffer(g("t10k-labels-idx1-ubyte.gz"),  np.uint8, offset=8)

    if flatten:
        Xtr = Xtr.reshape(len(Xtr), -1)
        Xte = Xte.reshape(len(Xte), -1)

    if one_hot:
        Ytr = np.zeros((ytr.size, 10))
        Yte = np.zeros((yte.size, 10))
        Ytr[np.arange(ytr.size), ytr] = 1
        Yte[np.arange(yte.size), yte] = 1
        return (Xtr, Ytr), (Xte, Yte)

    return (Xtr, ytr), (Xte, yte)


#CIFAR10_CLASSES = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#den her burde også virke
def cifar10(flatten=True, one_hot=False):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    tf = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz")

    def _read_bin(name):
        b = tf.extractfile(f"cifar-10-batches-bin/{name}").read()
        a = np.frombuffer(b, dtype=np.uint8).reshape(-1, 3073)
        y = a[:, 0].astype(np.int64)
        X = a[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  
        return X.astype(np.float32) / 255.0, y

    Xtr_list, ytr_list = [], []
    for i in range(1, 6):
        Xb, yb = _read_bin(f"data_batch_{i}.bin")
        Xtr_list.append(Xb); ytr_list.append(yb)
    Xtr = np.concatenate(Xtr_list, axis=0)
    ytr = np.concatenate(ytr_list, axis=0)

    Xte, yte = _read_bin("test_batch.bin")

    if flatten:
        Xtr = Xtr.reshape(len(Xtr), -1)
        Xte = Xte.reshape(len(Xte), -1)
   
    if one_hot:
        Ytr = np.zeros((ytr.size, 10), dtype=np.float32)
        Yte = np.zeros((yte.size, 10), dtype=np.float32)
        Ytr[np.arange(ytr.size), ytr] = 1.0
        Yte[np.arange(yte.size), yte] = 1.0
        return (Xtr, Ytr), (Xte, Yte)

    return (Xtr, ytr), (Xte, yte)

