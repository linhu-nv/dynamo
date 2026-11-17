# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime data for a router with ``--shared-cache-type radixshmem``.

``DYN_RADIXSHMEM_KV_EVENTS_ENDPOINT`` is this node's radix-server
``--kv-events-endpoint`` as the frontend reaches it. A worker whose DP ranks
span nodes lists every node's endpoint in node order in
``DYN_RADIXSHMEM_KV_EVENTS_ENDPOINTS``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Must match `RADIXSHMEM_RUNTIME_KEY` in `lib/llm/src/kv_router/radixshmem_cache.rs`.
RADIXSHMEM_RUNTIME_KEY = "radixshmem"


def radixshmem_endpoint() -> str | None:
    """This node's placement endpoint; None when radixshmem routing is off."""
    return os.getenv("DYN_RADIXSHMEM_KV_EVENTS_ENDPOINT") or None


def dp_rank_endpoints(num_ranks: int, ranks_per_node: int, endpoint: str) -> list[str]:
    """Each DP rank's node endpoint, for ranks placed on nodes in order,
    ``ranks_per_node`` (>= 1) per node.

    Empty when ``DYN_RADIXSHMEM_KV_EVENTS_ENDPOINTS`` does not list one endpoint
    per node: the router then treats every rank as on ``endpoint``'s node.
    """
    nodes = [
        e.strip()
        for e in os.getenv("DYN_RADIXSHMEM_KV_EVENTS_ENDPOINTS", "").split(",")
        if e.strip()
    ]
    if len(nodes) != -(-num_ranks // ranks_per_node):
        logger.warning(
            "DYN_RADIXSHMEM_KV_EVENTS_ENDPOINTS should list one endpoint per node "
            "(%d DP ranks, %d per node); the router will treat every DP rank as on %s",
            num_ranks,
            ranks_per_node,
            endpoint,
        )
        return []
    return [nodes[rank // ranks_per_node] for rank in range(num_ranks)]


def publish_radixshmem(runtime_config: Any, data: dict[str, Any]) -> None:
    """Publish a worker's radixshmem runtime data for the router."""
    runtime_config.set_engine_specific(RADIXSHMEM_RUNTIME_KEY, json.dumps(data))
    logger.info(
        "Published radixshmem placement endpoint for router use: %s",
        data["kv_events_endpoint"],
    )
