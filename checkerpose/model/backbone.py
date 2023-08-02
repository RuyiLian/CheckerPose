''' load timm backbone
reference: https://github.com/rwightman/pytorch-image-models
reference: https://github.com/shanice-l/gdrnpp_bop2022
'''
import timm
import copy
import pathlib
import torch

def my_create_timm_model(**init_args):
    # HACK: fix the bug for feature_only=True and checkpoint_path != ""
    # https://github.com/rwightman/pytorch-image-models/issues/488
    if init_args.get("checkpoint_path", "") != "" and init_args.get("features_only", True):
        init_args = copy.deepcopy(init_args)
        full_model_name = init_args["model_name"]
        modules = timm.models.list_modules()
        # find the mod which has the longest common name in model_name
        mod_len = 0
        for m in modules:
            if m in full_model_name:
                cur_mod_len = len(m)
                if cur_mod_len > mod_len:
                    mod = m
                    mod_len = cur_mod_len
        if mod_len >= 1:
            if hasattr(timm.models.__dict__[mod], "default_cfgs"):
                ckpt_path = init_args.pop("checkpoint_path")
                ckpt_url = pathlib.Path(ckpt_path).resolve().as_uri()
                print(f"hacking model pretrained url to {ckpt_url}")
                timm.models.__dict__[mod].default_cfgs[full_model_name]["url"] = ckpt_url
                init_args["pretrained"] = True
        else:
            raise ValueError(f"model_name {full_model_name} has no module in timm")

    backbone = timm.create_model(**init_args)
    return backbone

# note we return all layer features
def get_timm_backbone(model_name="resnet34", concat_decoder=True, pretrained=True):
    if model_name in ["convnext_tiny", "convnext_small", "convnext_base"]:
        out_indices = (1, 2, 3) if concat_decoder else (3,)
    elif model_name in ["resnet34", "hrnet_w18", "hrnet_w18_small", "hrnet_w30"]:
        out_indices = (1, 2, 3, 4) if concat_decoder else (4,)
    elif model_name in ["darknet53"]:
        out_indices = (1, 2, 3, 4, 5) if concat_decoder else (5,)
    else:
        raise ValueError("timm_backbone {} not supported yet".format(model_name))
    backbone = my_create_timm_model(model_name=model_name, pretrained=pretrained, in_chans=3, features_only=True,
                                    out_indices=out_indices)
    return backbone
