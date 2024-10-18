import pandas as pd
import argparse
from pathlib import Path
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("-s", required=True, help="Top folder from which to read results")
parser.add_argument("-d", required=True, help="Output folder for Results")

args = parser.parse_args()

df = pd.read_csv(args.s, sep="\t")
print(df)
