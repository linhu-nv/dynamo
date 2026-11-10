// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block positions an external shared cache holds for a request, as the host looked them
//! up and the cache column reports them.

use std::ops::Range;
use std::sync::Arc;

use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use rustc_hash::FxHashMap;

/// Shared cache hit information, represented as sorted non-overlapping half-open ranges.
///
/// Ranges encode which block positions exist in the external shared KV cache pool.
/// Using ranges instead of `Vec<bool>` avoids iterating over potentially thousands
/// of blocks per worker. Typical shared cache patterns produce few contiguous regions,
/// making `hits_beyond` O(num_ranges) ~ O(1-5).
#[derive(Debug, Clone, Default)]
pub struct SharedCacheHits {
    /// Ranges of block positions that exist in the shared cache.
    /// Half-open ranges [start, end), sorted and non-overlapping.
    pub ranges: Vec<Range<u32>>,
    /// Total number of hits (sum of range lengths).
    pub total_hits: u32,
    /// Which hits are on each worker's own host, when the pool is split by host.
    pub locality: Option<SharedCacheLocality>,
}

/// Host of each worker: one entry covers every dp rank, otherwise indexed by dp rank.
pub type SharedCacheWorkerHosts = FxHashMap<WorkerId, Vec<u32>>;

/// A shared pool made of per-host parts (e.g. one radixshmem pool per node).
///
/// A worker loads hits held by its own host directly; hits only another host
/// holds must be pulled across the network first, so they earn less credit.
#[derive(Debug, Clone)]
pub struct SharedCacheLocality {
    /// Leading run of blocks held by each host, indexed by host.
    pub host_prefix_blocks: Vec<u32>,
    pub worker_hosts: Arc<SharedCacheWorkerHosts>,
    /// Credit for a block pulled from another host, relative to a local one.
    pub remote_weight: f64,
}

impl SharedCacheLocality {
    fn local_prefix_blocks(&self, worker: WorkerWithDpRank) -> Option<u32> {
        let hosts = self.worker_hosts.get(&worker.worker_id)?;
        let host = match hosts.as_slice() {
            [host] => *host,
            hosts => *hosts.get(worker.dp_rank as usize)?,
        };
        Some(
            self.host_prefix_blocks
                .get(host as usize)
                .copied()
                .unwrap_or(0),
        )
    }
}

impl SharedCacheHits {
    /// Create from sorted, non-overlapping ranges.
    pub fn from_ranges(ranges: Vec<Range<u32>>) -> Self {
        let total_hits = ranges.iter().map(|r| r.end - r.start).sum();
        Self {
            ranges,
            total_hits,
            locality: None,
        }
    }

    pub fn with_locality(mut self, locality: SharedCacheLocality) -> Self {
        self.locality = Some(locality);
        self
    }

    /// Create from a boolean hit vector (convenience for tests and simple backends).
    /// Coalesces consecutive `true` entries into ranges.
    pub fn from_hits(hits: &[bool]) -> Self {
        let mut ranges = Vec::new();
        let mut i = 0;
        while i < hits.len() {
            if hits[i] {
                let start = i as u32;
                while i < hits.len() && hits[i] {
                    i += 1;
                }
                ranges.push(start..i as u32);
            } else {
                i += 1;
            }
        }
        Self::from_ranges(ranges)
    }

    /// Count hits at positions >= `from_position`.
    /// O(num_ranges), not O(num_blocks).
    pub fn hits_beyond(&self, from_position: u32) -> u32 {
        self.ranges
            .iter()
            .map(|r| {
                if r.end <= from_position {
                    0
                } else if r.start >= from_position {
                    r.end - r.start
                } else {
                    r.end - from_position
                }
            })
            .sum()
    }

    /// `hits_beyond(from_position)` for `worker`, with hits outside its own
    /// host weighted by the locality's `remote_weight`. Without locality, or
    /// for a worker whose host is unknown, every hit counts fully.
    pub fn weighted_hits_beyond(&self, worker: WorkerWithDpRank, from_position: u32) -> f64 {
        let hits = self.hits_beyond(from_position);
        let Some(locality) = &self.locality else {
            return hits as f64;
        };
        let Some(local_prefix) = locality.local_prefix_blocks(worker) else {
            return hits as f64;
        };
        let local = local_prefix.saturating_sub(from_position).min(hits);
        local as f64 + locality.remote_weight * (hits - local) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_cache_hits_from_hits() {
        // All hits contiguous
        let hits = SharedCacheHits::from_hits(&[true, true, true, true]);
        assert_eq!(hits.ranges, vec![0..4]);
        assert_eq!(hits.total_hits, 4);

        // Sparse hits
        let hits = SharedCacheHits::from_hits(&[true, false, true, true, false, true]);
        assert_eq!(hits.ranges, vec![0..1, 2..4, 5..6]);
        assert_eq!(hits.total_hits, 4);

        // No hits
        let hits = SharedCacheHits::from_hits(&[false, false, false]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);

        // Empty
        let hits = SharedCacheHits::from_hits(&[]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);
    }

    #[test]
    fn test_shared_cache_hits_beyond() {
        // Shared has [A, B, C, D] => range 0..4
        #[allow(clippy::single_range_in_vec_init)]
        let hits = SharedCacheHits::from_ranges(vec![0..4]);

        // Device has overlap=2 (positions 0,1 on device) => shared_beyond should count positions 2,3
        assert_eq!(hits.hits_beyond(2), 2);

        // Device has overlap=0 => all 4 shared hits count
        assert_eq!(hits.hits_beyond(0), 4);

        // Device has overlap=4 => nothing beyond
        assert_eq!(hits.hits_beyond(4), 0);

        // Device overlap exceeds range
        assert_eq!(hits.hits_beyond(10), 0);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_shared_cache_hits_weighted_by_locality() {
        let hosts = SharedCacheWorkerHosts::from_iter([(1, vec![0]), (2, vec![0, 1])]);
        let hits = SharedCacheHits::from_ranges(vec![0..6]).with_locality(SharedCacheLocality {
            host_prefix_blocks: vec![2, 6],
            worker_hosts: Arc::new(hosts),
            remote_weight: 0.5,
        });
        let at = |worker_id, dp_rank| WorkerWithDpRank { worker_id, dp_rank };
        // host 0 holds 2 of 6 blocks: 2 local + 4 remote at half credit
        assert_eq!(hits.weighted_hits_beyond(at(1, 3), 0), 4.0);
        // device prefix covers the local part
        assert_eq!(hits.weighted_hits_beyond(at(1, 0), 3), 1.5);
        // per-rank hosts: rank 1 sits on host 1, which holds all 6
        assert_eq!(hits.weighted_hits_beyond(at(2, 0), 0), 4.0);
        assert_eq!(hits.weighted_hits_beyond(at(2, 1), 0), 6.0);
        // unknown worker or rank: every hit counts fully
        assert_eq!(hits.weighted_hits_beyond(at(9, 0), 0), 6.0);
        assert_eq!(hits.weighted_hits_beyond(at(2, 5), 0), 6.0);
    }

    #[test]
    fn test_shared_cache_hits_beyond_sparse() {
        // Ranges: [1..3, 5..8] => positions 1,2,5,6,7
        let hits = SharedCacheHits::from_ranges(vec![1..3, 5..8]);
        assert_eq!(hits.total_hits, 5);

        // from_position=0 => all 5 hits
        assert_eq!(hits.hits_beyond(0), 5);
        // from_position=2 => pos 2 (from first range) + 5,6,7 (from second) = 4
        assert_eq!(hits.hits_beyond(2), 4);
        // from_position=3 => only second range: 3 hits
        assert_eq!(hits.hits_beyond(3), 3);
        // from_position=6 => positions 6,7 from second range = 2
        assert_eq!(hits.hits_beyond(6), 2);
        // from_position=8 => nothing
        assert_eq!(hits.hits_beyond(8), 0);
    }
}
