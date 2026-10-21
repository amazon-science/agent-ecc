# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters connect task families to agent-ecc: their evidence format, how their recorded outputs are re-parsed, and
how their tests are re-executed. Each adapter module imports its third-party dependencies lazily, so it can be
imported (and its records handled) without them."""

__all__ = ["appworld", "swebench"]
