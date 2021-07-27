

import argparse

def train_model():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('output_model_path')
    parser.add_argument('training_spectra_path')
    parser.add_argument('training_labels_path')
    parser.add_argument('validation_spectra_path')
    parser.add_argument('validation_labels_path')
    parser.add_argument('test_spectra_path')
    parser.add_argument('test_labels_path')
    #parser.add_argument('class_names')
    parser.add_argument('network_factory')

    args = parser.parse_args()

    from astra.contrib.classifier.operators import train_model

    train_model(
        args.output_model_path,
        args.training_spectra_path,
        args.training_labels_path,
        args.validation_spectra_path,
        args.validation_labels_path,
        args.test_spectra_path,
        args.test_labels_path,
        ["fgkm", "hotstar", "sb2", "yso"],
        args.network_factory
    )

    #    learning_rate=1e-5,
    #    weight_decay=1e-5,
    #    n_epochs=200,
    #    batch_size=100,
