from __future__ import annotations

import sys


def main() -> int:
    import torch
    from audiosr.latent_diffusion.modules.diffusionmodules.openaimodel import UNetModel
    from audiosr.utils import default_audioldm_config

    config = default_audioldm_config("basic")
    model_params = config["model"]["params"]
    unet_config = model_params["unet_config"]
    unet_params = dict(unet_config["params"])

    latent_t = int(model_params["latent_t_size"])
    latent_f = int(model_params["latent_f_size"])
    latent_channels = int(model_params["channels"])
    expected_in_channels = latent_channels * 2
    configured_in_channels = int(unet_params["in_channels"])
    if configured_in_channels != expected_in_channels:
        raise RuntimeError(
            "Неожиданное число входных каналов AudioSR UNet: "
            f"конфиг={configured_in_channels}, ожидалось={expected_in_channels}."
        )

    print(
        "Meta-проверка AudioSR UNet: "
        f"input=[2, {configured_in_channels}, {latent_t}, {latent_f}]",
        flush=True,
    )

    # На meta device параметры не занимают RAM и операции не считают реальные данные.
    # Это позволяет проверить захват реального Python-графа большого UNet на CPU runner.
    with torch.device("meta"):
        model = UNetModel(**unet_params).eval()
        x = torch.randn(2, configured_in_channels, latent_t, latent_f)
        timesteps = torch.tensor([999, 500], dtype=torch.long)

    kwargs = {
        "y": None,
        "context_list": [],
        "context_attn_mask_list": [],
    }

    exported = torch.export.export(
        model,
        (x, timesteps),
        kwargs=kwargs,
        strict=False,
    )
    node_count = sum(1 for _ in exported.graph.nodes)
    if node_count < 10:
        raise RuntimeError(
            f"Экспортированный граф подозрительно мал: {node_count} узлов."
        )

    print(
        f"torch.export успешно захватил AudioSR UNet: {node_count} узлов.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(
            f"AudioSR UNet export check failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
