import os
from fire import Fire


def main(
    num
):
    seeds = {0:273, 1:588, 2:600, 3:681, 4:846}
    device = {0: "cuda:0", 1: "cuda:0", 2: "cuda:0", 3: "cuda:0", 4: "cuda:0"}
    augments = ["delete", "change", "reverse", "composite"]
    dim = [32]
    
    for dataset in ["review", "bonanza", "digi_music", "ml-1m_pivot3"]:
        for dim in [32]:
            for aug in augments:
                os.system(f"python main.py --dataset_name {dataset} --seed {seeds[num]} --input_dim {dim} --augment {aug} --device {device[num]}")
       

if __name__ == "__main__":
    Fire(main)
