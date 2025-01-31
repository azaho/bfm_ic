# bfm_ic

To run the script, you need to install the following packages:
```
pip install beautifulsoup4 requests torch h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn
```

1. Download the braintreebank dataset (link: https://braintreebank.dev/)
First, run `braintreebank_download.py` to download the zip files into `braintreebank_zip` directory.
Then, run `braintreebank_extract.py` to extract the zip files into `braintreebank` directory.
At this point, the folder `braintreebank_zip` can be deleted to free up space.

2. Run `braintreebank_process_chunks.py` and `braintreebank_process_benchmark_chunks.py` to process the benchmark chunks into numpy arrays.

3. Run `ttt_clip_old_newer.py` to train a test model.