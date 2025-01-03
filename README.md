# bfm_ic

To run the script, you need to install the following packages:
```
pip install beautifulsoup4 requests torch h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn
```

1. Download the braintreebank dataset (link: https://braintreebank.dev/)
First, run `braintreebank_download.py` to download the zip files into `braintreebank_zip` directory.
Then, run `braintreebank_extract.py` to extract the zip files into `braintreebank` directory.
At this point, the folder `braintreebank_zip` can be deleted to free up space.

2. Run `transformer_test.py` to test the transformer architecture.
This script will create a random input tensor and run the transformer architecture.
It will also print the memory usage of the GPU and CPU.