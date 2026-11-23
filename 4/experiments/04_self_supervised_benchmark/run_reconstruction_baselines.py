from bootstrap import configuration
from src.self_supervised.embeddings import score_reconstruction

if __name__=="__main__":
    cfg=configuration()
    for site in cfg["sites"]:
        score_reconstruction(site,"validation",cfg)

