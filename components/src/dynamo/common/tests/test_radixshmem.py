# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest

from dynamo.common.radixshmem import dp_rank_endpoints, publish_radixshmem

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_publish_radixshmem_uses_the_router_key():
    runtime_config = Mock()
    data = {"kv_events_endpoint": "tcp://a:5891"}

    publish_radixshmem(runtime_config, data)

    # RADIXSHMEM_RUNTIME_KEY in lib/llm/src/kv_router/radixshmem_cache.rs
    runtime_config.set_engine_specific.assert_called_once_with(
        "radixshmem", json.dumps(data)
    )


def test_dp_rank_endpoints_follow_node_order(monkeypatch):
    monkeypatch.setenv(
        "DYN_RADIXSHMEM_KV_EVENTS_ENDPOINTS", "tcp://a:5891, tcp://b:5891"
    )

    assert dp_rank_endpoints(4, 2, "tcp://a:5891") == [
        "tcp://a:5891",
        "tcp://a:5891",
        "tcp://b:5891",
        "tcp://b:5891",
    ]
    # one endpoint per node, or none
    assert dp_rank_endpoints(6, 2, "tcp://a:5891") == []
    assert dp_rank_endpoints(2, 2, "tcp://a:5891") == []
