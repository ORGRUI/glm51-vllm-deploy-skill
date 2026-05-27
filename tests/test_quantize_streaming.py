from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import FineGrainedFP8Config, PretrainedConfig

ROOT = Path(__file__).resolve().parents[1]
QUANTIZE_PATH = ROOT / "shared" / "scripts" / "quantize_glm51_fp8_block128.py"

spec = importlib.util.spec_from_file_location(
    "quantize_glm51_fp8_block128", QUANTIZE_PATH
)
assert spec is not None
assert spec.loader is not None
quantize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantize)


def test_quantizable_weight_selection_respects_glm51_contract():
    skip = [
        "model.embed_tokens",
        "lm_head",
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.indexer.weights_proj",
        "model.layers.0.self_attn.indexer.wq_b",
        "model.layers.0.self_attn.indexer.wk",
    ]
    fp8_shape = torch.zeros((128, 128), dtype=torch.bfloat16)
    bad_shape = torch.zeros((128, 64), dtype=torch.bfloat16)

    assert quantize.is_quantizable_glm51_weight(
        "model.layers.0.self_attn.q_b_proj.weight",
        tensor=fp8_shape,
        modules_to_not_convert=skip,
        weight_block_size=(128, 128),
    )
    assert quantize.is_quantizable_glm51_weight(
        "model.layers.0.mlp.experts.3.down_proj.weight",
        tensor=fp8_shape,
        modules_to_not_convert=skip,
        weight_block_size=(128, 128),
    )
    assert not quantize.is_quantizable_glm51_weight(
        "model.layers.0.self_attn.q_a_proj.weight",
        tensor=fp8_shape,
        modules_to_not_convert=skip,
        weight_block_size=(128, 128),
    )
    assert not quantize.is_quantizable_glm51_weight(
        "model.embed_tokens.weight",
        tensor=fp8_shape,
        modules_to_not_convert=skip,
        weight_block_size=(128, 128),
    )
    assert not quantize.is_quantizable_glm51_weight(
        "model.layers.0.self_attn.o_proj.weight",
        tensor=bad_shape,
        modules_to_not_convert=skip,
        weight_block_size=(128, 128),
    )
    assert not quantize.is_quantizable_glm51_weight(
        "model.layers.0.self_attn.indexer.wq_b.weight",
        tensor=fp8_shape,
        modules_to_not_convert=skip,
        weight_block_size=(128, 128),
    )
    assert not quantize.is_quantizable_glm51_weight(
        "model.layers.0.self_attn.indexer.wk.weight",
        tensor=fp8_shape,
        modules_to_not_convert=skip,
        weight_block_size=(128, 128),
    )


def test_fp8_skip_modules_cover_indexer_bf16_compatibility_layers():
    config = PretrainedConfig(num_hidden_layers=2)

    skip = quantize.build_fp32_skip_modules(config)

    assert "model.layers.0.self_attn.indexer.weights_proj" in skip
    assert "model.layers.0.self_attn.indexer.wq_b" in skip
    assert "model.layers.0.self_attn.indexer.wk" in skip
    assert "model.layers.1.self_attn.indexer.wq_b" in skip
    assert "model.layers.1.self_attn.indexer.wk" in skip


def test_streaming_export_writes_fp8_weights_scales_and_index(tmp_path):
    source_dir = tmp_path / "source"
    export_dir = tmp_path / "export"
    source_dir.mkdir()
    export_dir.mkdir()

    tensors = {
        "model.embed_tokens.weight": torch.ones((128, 128), dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_a_proj.weight": torch.ones(
            (128, 128), dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.q_b_proj.weight": torch.arange(
            128 * 128, dtype=torch.float32
        ).reshape(128, 128),
        "model.layers.0.self_attn.indexer.wq_b.weight": torch.ones(
            (128, 128), dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.indexer.wk.weight": torch.ones(
            (128, 128), dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.ones(
            (128, 128), dtype=torch.bfloat16
        ),
        "model.layers.0.input_layernorm.weight": torch.ones(
            (128,), dtype=torch.bfloat16
        ),
    }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(source_dir / shard_name), metadata={"format": "pt"})
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name in tensors},
            }
        )
    )

    config = PretrainedConfig(num_hidden_layers=1, model_type="glm_moe_dsa")
    quant_cfg = FineGrainedFP8Config(
        modules_to_not_convert=[
            "model.embed_tokens",
            "lm_head",
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            "model.layers.0.self_attn.indexer.weights_proj",
            "model.layers.0.self_attn.indexer.wq_b",
            "model.layers.0.self_attn.indexer.wk",
        ]
    )

    summary = quantize.export_streaming_fp8_checkpoint(
        base_model_path=str(source_dir),
        export_dir=export_dir,
        config_source=str(source_dir),
        config=config,
        quant_cfg=quant_cfg,
        modules_to_not_convert=quant_cfg.modules_to_not_convert,
    )

    assert summary["quantized_weight_count"] == 2
    assert summary["scale_tensor_count"] == 2

    index_payload = json.loads(
        (export_dir / "model.safetensors.index.json").read_text()
    )
    weight_map = index_payload["weight_map"]
    assert (
        weight_map["model.layers.0.self_attn.q_b_proj.weight_scale_inv"] == shard_name
    )
    assert (
        weight_map["model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"]
        == shard_name
    )
    assert "model.embed_tokens.weight_scale_inv" not in weight_map
    assert "model.layers.0.self_attn.q_a_proj.weight_scale_inv" not in weight_map
    assert "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv" not in weight_map
    assert "model.layers.0.self_attn.indexer.wk.weight_scale_inv" not in weight_map

    with safe_open(
        str(export_dir / shard_name), framework="pt", device="cpu"
    ) as handle:
        assert (
            handle.get_tensor("model.layers.0.self_attn.q_b_proj.weight").dtype
            == torch.float8_e4m3fn
        )
        assert handle.get_tensor(
            "model.layers.0.self_attn.q_b_proj.weight_scale_inv"
        ).shape == (1, 1)
        assert (
            handle.get_tensor("model.layers.0.self_attn.q_a_proj.weight").dtype
            == torch.bfloat16
        )
        assert (
            handle.get_tensor("model.layers.0.self_attn.indexer.wq_b.weight").dtype
            == torch.bfloat16
        )
        assert (
            handle.get_tensor("model.layers.0.self_attn.indexer.wk.weight").dtype
            == torch.bfloat16
        )

    exported_config = json.loads((export_dir / "config.json").read_text())
    assert exported_config["quantization_config"]["quant_method"] == "fp8"
    assert exported_config["quantization_config"]["weight_block_size"] == [128, 128]
