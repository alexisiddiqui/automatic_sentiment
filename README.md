# Automatic Sentiment Analysis

This repository contains code for an automatic sentiment analysis project using image and text data. The project uses a Variational Autoencoder (VAE) to combine image and text embeddings for sentiment analysis.

## Project Structure

```
├── best_vae_model.pth
├── environment.yml
├── inference_output/
├── README.md
├── scripts/
│   └── flickr_test/
│       ├── 0_flickr_dataset_setup.py
│       ├── 1_flickr_BERT_featurisation.py
│       ├── 2_create_training_validation_PCA.py
│       ├── 2_create_training_validation_tSNE.py
│       ├── 3_test_train.py
│       ├── 4_inference.py
│       └── models_and_loaders.py
├── sentiment_AE/
│   ├── models.py
│   ├── settings.py
│   └── training.py
├── setup.sh
├── train_val_split_visualization_pca.png
└── train_val_split_visualization_tsne.png
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/automatic_sentiment.git
   cd automatic_sentiment
   ```

2. Set up the Conda environment:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

   Alternatively, you can create the environment from the `environment.yml` file:
   ```
   conda env create -f environment.yml
   ```

3. Activate the Conda environment:
   ```
   conda activate automatic_sentiment
   ```

## Data Preparation and Processing

1. Set up the Flickr dataset:
   ```
   python scripts/flickr_test/0_flickr_dataset_setup.py
   ```

2. Generate BERT features for the dataset:
   ```
   python scripts/flickr_test/1_flickr_BERT_featurisation.py
   ```

3. Create training and validation splits:
   ```
   python scripts/flickr_test/2_create_training_validation_PCA.py
   ```
   or
   ```
   python scripts/flickr_test/2_create_training_validation_tSNE.py
   ```

## Training

To train the VAE model:

```
python scripts/flickr_test/3_test_train.py
```

You can modify hyperparameters in the `sentiment_AE/settings.py` file.



# Inference

## Note: Working model weights are given by default - please get in touch if this is giving strange outputs.


To run inference on new data, use the following command:

```
python scripts/flickr_test/4_inference.py --image_dir raw_data/skyscrapers --text_list 'love' 'hate'
```

This command will:
- Use images from the `raw_data/skyscrapers` directory
- Apply the sentiments 'love' and 'hate' to each image
- Output the results in the `inference_output` directory

You can modify the `--image_dir` and `--text_list` arguments to use different input images and sentiments.

## Project Components

- `scripts/flickr_test/models_and_loaders.py`: Contains model definitions and data loaders
- `sentiment_AE/models.py`: Defines the VAE model architecture
- `sentiment_AE/settings.py`: Contains project settings and hyperparameters
- `sentiment_AE/training.py`: Implements the training loop
- `best_vae_model.pth`: The best trained VAE model
- `train_val_split_visualization_pca.png` and `train_val_split_visualization_tsne.png`: Visualizations of the train-validation split using PCA and t-SNE

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Flickr dataset providers
- Contributors to PyTorch and Transformers libraries