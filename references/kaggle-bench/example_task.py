# Adapted and modified from Kaggle Benchmarks at commit:
# 0d759286b5dcb83878e27840fcb42b559062abf3
# https://github.com/Kaggle/kaggle-benchmarks/blob/0d759286b5dcb83878e27840fcb42b559062abf3/documentation/examples/use_ipython_magics.py
# Copyright 2025 Kaggle Inc.
# SPDX-License-Identifier: Apache-2.0
#
# This reduced copy is a non-normative syntax reference, not benchmark source.
import kaggle_benchmarks as kbench

@kbench.task(name="What is Kaggle?", description="Does the LLM know what Kaggle is?")
def what_is_kaggle(llm) -> None:
    response = llm.prompt("What is Kaggle?")
    kbench.assertions.assert_in("platform", response.lower())

what_is_kaggle.run(kbench.llm)
