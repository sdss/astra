import argparse

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('pks')

    args = parser.parse_args()

    from astra.contrib.apogeenet.operators import estimate_stellar_labels

    estimate_stellar_labels(args.pks, args.model_path)