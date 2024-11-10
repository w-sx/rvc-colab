from sys import argv
from random import shuffle
import os
import pathlib

# example:
# python train_reset.py logs/wsx 48k 0 v2 true

exp_dir = argv[1]
sr = argv[2]
spk_id = argv[3]
version = argv[4]
if_f0 = argv[5].lower() == "true"

if __name__ == '__main__':
    print('create filelist')
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id,
                )
            )
    fea_dim = 256 if version == "v1" else 768
    if if_f0:
        for _ in range(2):
            opt.append(
                "logs/mute/0_gt_wavs/mute%s.wav|logs/mute/3_feature%s/mute.npy|logs/mute/2a_f0/mute.wav.npy|logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (sr, fea_dim, spk_id)
            )
    else:
        for _ in range(2):
            opt.append(
                "logs/mute/0_gt_wavs/mute%s.wav|logs/mute/3_feature%s/mute.npy|%s"
                % (sr, fea_dim, spk_id)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print('create config')
    if version == "v1" or sr == "40k":
        config_path = "v1/%s.json" % sr
    else:
        config_path = "v2/%s.json" % sr
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")