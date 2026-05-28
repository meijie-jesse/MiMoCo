# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .seq_vae import build as build_vae
from .seq_vae import build_cnnmlp as build_cnnmlp

def build_chunk_seq_model(args):
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)