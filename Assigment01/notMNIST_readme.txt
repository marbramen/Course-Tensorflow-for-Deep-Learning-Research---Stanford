##
mkdir notMNIST-to-MNIST 
cd notMNIST-to-MNIST

## Download notMNIST
curl -o notMNIST_small.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
curl -o notMNIST_large.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_sarge.tar.gz
tar xzf notMNIST_small.tar.gz
tar xzf notMNIST_large.tar.gz

## notMNIST - Input data Pre-Processing
find . -name "*.png" -size 0 -print -exec rm -f {} \;
rm notMNIST_large/A/RnJlaWdodERpc3BCb29rSXRhbGljLnR0Zg==.png

## Download David Flanagan's script to convert notMNIST at same format MNIST
### Please review David Flanagan's git: https://github.com/davidflanagan/notMNIST-to-MNIST
curl -o convert_to_mnist_format.py https://github.com/davidflanagan/notMNIST-to-MNIST/blob/master/convert_to_mnist_format.py

##Convert to format MNIST
python convert_to_mnist_format.py notMNIST_large 50000 train-labels-idx1-ubyte train-images-idx3-ubyte
python convert_to_mnist_format.py notMNIST_small 1800 t10k-labels-idx1-ubyte t10k-images-idx3-ubyte
gzip *ubyte
