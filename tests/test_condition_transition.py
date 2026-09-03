# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
import time
import unittest

import torch

os.environ["LAYERNORM_TYPE"] = "torch"
from opendde.model.modules.transformer import ConditionedTransitionBlock


class TestConditionedTransitionBlock(unittest.TestCase):
    def setUp(self) -> None:
        self._start_time = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().setUp()

    def get_model(self, c_a: int = 768, c_s: int = 384, n: int = 2):

        model = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=n).to(self.device)

        return model

    def test_shape(self) -> None:

        c_a = 5 * 55
        c_s = 123

        N_token = 135
        bs_dims = (2, 3, 5)

        inputs = {
            "a": torch.rand(size=(*bs_dims, N_token, c_a)).to(self.device),
            "s": torch.rand(size=(*bs_dims, N_token, c_s)).to(self.device),
        }

        model = self.get_model(c_a=c_a, c_s=c_s)

        out = model(**inputs)
        target_shape = (*bs_dims, N_token, c_a)
        self.assertEqual(out.shape, out.reshape(target_shape).shape)

    def tearDown(self):
        elapsed_time = time.time() - self._start_time
        print(f"Test {self.id()} took {elapsed_time:.6f}s")


if __name__ == "__main__":
    unittest.main()
