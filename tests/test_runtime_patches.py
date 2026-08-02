from __future__ import annotations

import unittest

from audio_restoration_colab.runtime_patches import (
    apply_audio_separator_quality_patches,
)


class RuntimePatchTests(unittest.TestCase):
    def test_quality_patch_is_safe_without_model_dependencies(self) -> None:
        apply_audio_separator_quality_patches()


if __name__ == "__main__":
    unittest.main()
