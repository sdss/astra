import argparse
import json

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')

    #args, pks = parser.parse_known_args()
    #pks = [int(ea.strip("[],")) for ea in pks]
    parser.add_argument('pks_path')

    args = parser.parse_args()

    from astra.contrib.apogeenet.operators import estimate_stellar_labels

    with open(args.pks_path, "r") as fp:
        pks = json.load(fp)

    estimate_stellar_labels(pks, args.model_path)