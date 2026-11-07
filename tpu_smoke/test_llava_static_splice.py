"""CPU bitwise check: static_splice == upstream prepare_inputs_labels_for_multimodal.

Locks one property: for the LLaVA-1.5 finetune case (one <image> per sample,
right padding) the static splice returns the SAME position_ids, attention_mask,
inputs_embeds and labels as upstream, element for element, including rows that
are padded and an image token at position 0 and at the last real position.

Uses a tiny random LlavaLlamaForCausalLM whose vision tower is replaced by a
stub returning fixed random features, so no weights are downloaded and the
comparison isolates the splice.

Run from the repository root, CPU only (needs torch and transformers, not torch_xla):
  python -m tpu_smoke.test_llava_static_splice
"""
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # repository root: holds llava/ and tpu_smoke/
sys.path.insert(0, _ROOT)

from transformers import LlamaConfig
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

from tpu_smoke.llava_static_splice import check_supported, static_splice, uniform_image_pos

N_PATCH, H = 7, 32


def tiny_model():
    cfg = LlamaConfig(vocab_size=200, hidden_size=H, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
    torch.manual_seed(0)
    m = LlavaLlamaForCausalLM(cfg).eval()
    feats = torch.randn(64, N_PATCH, H)
    # Stub the vision path: encode_images(images) -> per-sample fixed features,
    # selected by the first pixel so the batch order is checked too.
    m.get_vision_tower = lambda: object()
    m.encode_images = lambda imgs: feats[imgs[:, 0, 0, 0].long()]
    m.config.image_aspect_ratio = "pad"
    m.config.mm_patch_merge_type = "flat"
    m.config.tokenizer_padding_side = "right"
    return m


def make_batch(lens, img_pos, T=12):
    B = len(lens)
    ids = torch.randint(1, 200, (B, T))
    am = torch.zeros(B, T, dtype=torch.long)
    for b, (n, p) in enumerate(zip(lens, img_pos)):
        am[b, :n] = 1
        ids[b, p] = IMAGE_TOKEN_INDEX
        ids[b, n:] = 0
    labels = ids.clone()
    labels[:, :3] = -100
    labels[am == 0] = -100
    images = torch.zeros(B, 3, 4, 4)
    images[:, 0, 0, 0] = torch.arange(B).float()
    return ids, am, labels, images


def compare(m, ids, am, labels, images):
    with torch.no_grad():
        # Pass explicit position_ids: upstream returns None for them when the
        # caller passed None (llava_arch.py:320), and the recipe's trainer path
        # then lets the LM derive positions. The padded arange it builds is what
        # static_splice returns, so that is what has to match.
        _, up_pos, up_am, _, up_emb, up_lab = m.prepare_inputs_labels_for_multimodal(
            ids, torch.arange(ids.shape[1]).unsqueeze(0).expand_as(ids), am, None, labels, images)
        pos = check_supported(ids, am)
        outs = [static_splice(m, ids, am, labels, images, pos)]
        up = uniform_image_pos(pos)
        if up is not None:                     # the shared-position fast path, too
            outs.append(static_splice(m, ids, am, labels, images, pos, uniform_pos=up))
    for st_pos, st_am, st_emb, st_lab in outs:
        for name, a, b in (("position_ids", up_pos, st_pos), ("attention_mask", up_am, st_am),
                           ("inputs_embeds", up_emb, st_emb), ("labels", up_lab, st_lab)):
            assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
            assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} vs {b.dtype}"
            assert torch.equal(a, b), f"{name}: max diff {(a.float() - b.float()).abs().max()}"
    return len(outs)


def main():
    m = tiny_model()
    torch.manual_seed(1)
    cases = {
        "full rows, image at 1": ([12, 12, 12], [1, 1, 1]),
        "ragged right padding": ([12, 9, 5], [1, 4, 2]),
        "image at 0 and at last real token": ([12, 6, 8], [0, 5, 7]),
        "shared position 3, ragged padding": ([12, 9, 5], [3, 3, 3]),
        "shared position 0": ([12, 7, 12], [0, 0, 0]),
    }
    for name, (lens, pos) in cases.items():
        n = compare(m, *make_batch(lens, pos))
        print(f"PASS {name} ({n} path{'s' if n > 1 else ''})")
    # Mutation: the fast path with the wrong p must NOT match upstream.
    ids, am, labels, images = make_batch([12, 12], [2, 2])
    try:
        with torch.no_grad():
            _, up_pos, up_am, _, up_emb, up_lab = m.prepare_inputs_labels_for_multimodal(
                ids, torch.arange(12).unsqueeze(0).expand_as(ids), am, None, labels, images)
            bad = static_splice(m, ids, am, labels, images, check_supported(ids, am), uniform_pos=1)
        assert not torch.equal(bad[2], up_emb)
        print("PASS mutation (wrong shared position) caught")
    except AssertionError:
        raise AssertionError("fast path with wrong position matched upstream: test is blind")
    # Guard must refuse what static_splice does not cover.
    ids, am, labels, images = make_batch([12, 12], [1, 1])
    ids[0, 5] = IMAGE_TOKEN_INDEX
    try:
        check_supported(ids, am)
    except ValueError:
        print("PASS guard rejects two images in one sample")
    else:
        raise AssertionError("guard accepted two images")
    am2 = am.clone(); am2[1, 0] = 0
    try:
        check_supported(make_batch([12, 12], [1, 1])[0], am2)
    except ValueError:
        print("PASS guard rejects left padding")
    else:
        raise AssertionError("guard accepted left padding")


if __name__ == "__main__":
    main()
