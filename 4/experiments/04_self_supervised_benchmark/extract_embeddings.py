import argparse

from bootstrap import configuration
from src.self_supervised.embeddings import extract
from src.self_supervised.pretraining import Spec
from src.self_supervised.utils import PARTS

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("site")
    parser.add_argument("architecture")
    parser.add_argument("seed",type=int)
    parser.add_argument("--cohort",default="A")
    parser.add_argument("--objective",default="masked")
    parser.add_argument("--variant",default="full")
    parser.add_argument("--partition",choices=PARTS,required=True)
    args=parser.parse_args()
    spec=Spec(args.site,args.architecture,args.seed,args.cohort,args.objective,args.variant)
    extract(spec,configuration(),args.partition)

