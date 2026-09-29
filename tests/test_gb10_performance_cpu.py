from types import SimpleNamespace

import torch

from exllamav3.architecture.qwen4_exp_mtp import MTP_HEAD_SIZE, Qwen4ExpMTPModel
from exllamav3.generator.sampler import ArgmaxSampler, TopKSampler
from exllamav3.generator.sampler.custom import CustomSampler, SS_Argmax, SS_RepP
from exllamav3.modules.hyperconnections import GatedResidual


def test_pruned_mtp_head_slices_exl3_output_columns():
    model = object.__new__(Qwen4ExpMTPModel)
    inner = SimpleNamespace(
        trellis = torch.zeros((1, 8192, 16), dtype = torch.int16),
        svh = torch.arange(131072, dtype = torch.float16),
        bias = None,
    )

    pruned = model._get_pruned_head(SimpleNamespace(inner = inner), torch.device("cpu"))

    assert pruned is not None
    trellis, output_scale, output_size = pruned
    assert output_size == MTP_HEAD_SIZE
    assert trellis.shape == (1, MTP_HEAD_SIZE // 16, 16)
    assert output_scale.shape == (MTP_HEAD_SIZE,)


def test_pruned_mtp_head_falls_back_and_caches_failure():
    model = object.__new__(Qwen4ExpMTPModel)
    unsupported = SimpleNamespace(inner = SimpleNamespace(trellis = None))

    assert model._get_pruned_head(unsupported, torch.device("cpu")) is None
    assert model._get_pruned_head(unsupported, torch.device("cpu")) is None
    assert model._pruned_head_cache is False


def test_batch_verify_only_accepts_position_independent_greedy_sampler():
    assert ArgmaxSampler().supports_batch_verify
    assert CustomSampler([SS_Argmax()]).supports_batch_verify

    penalized = CustomSampler([SS_RepP(1.1), SS_Argmax()])
    assert penalized.reqs_past_ids
    assert not penalized.supports_batch_verify
    assert not TopKSampler(50, 0.8).supports_batch_verify


def test_decode_weight_quantization_shapes_and_releases_fp16_copies():
    module = object.__new__(GatedResidual)
    stream_count, hidden_size, rank = 4, 16, 8
    module.fn_h = torch.randn(rank + stream_count, stream_count * hidden_size, dtype = torch.float16)
    module.upx_h = torch.randn(
        stream_count, hidden_size // 4, rank, 4, dtype = torch.float16
    )

    module._quantize_decode_weights(stream_count, hidden_size)

    assert module.fn_q.shape == (rank + stream_count, stream_count * hidden_size)
    assert module.fn_q.dtype == torch.int8
    assert module.fn_s.shape == (rank + stream_count,)
    assert module.upx_q.shape == (stream_count, hidden_size // 4, rank, 4)
    assert module.upx_q.dtype == torch.int8
    assert module.upx_s.shape == (stream_count, hidden_size)
    assert module.fn_h is None
    assert module.upx_h is None
