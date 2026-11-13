// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared KV cache for SGLang and vLLM workers on FlexKV + radixshmem.
//!
//! Each node runs one radix-server whose host pool is shared by every worker on
//! the node. The server publishes the blocks it stores and drops; this client
//! subscribes to every node, keeps `hash -> nodes holding it`, and answers a
//! request with the leading run held by some node plus each node's own run, so
//! the scorer can prefer workers whose node already holds the prefix.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::{metrics::RoutingOverheadMetrics, shared_cache::sglang_page_digests},
    local_model::runtime_config::ModelRuntimeConfig,
    utils::zmq::{connect_sub_socket, multipart_message},
};
use dynamo_kv_indexer::host::KvRouterError;
use dynamo_router_protocol::worker::WorkerId;
use dynamo_scheduler::host::SharedKvCache;
use dynamo_worker_selection::{SharedCacheHits, SharedCacheLocality, SharedCacheWorkerHosts};

/// Runtime-data key of a worker's `RadixShmemRuntimeConfig`.
pub const RADIXSHMEM_RUNTIME_KEY: &str = "radixshmem";
/// The schemes this client can recompute from request tokens. Each keys a page
/// by its digest's first 8 bytes as a big-endian signed integer.
/// SGLang's SHA-256 page chain (SGLang's `hash_str_to_int64`).
pub const SGLANG_SHA256_I64_SCHEME: &str = "sglang-sha256-i64-v1";
/// vLLM's own block hashes with `--prefix-caching-hash-algo sha256_cbor`, for
/// blocks without extra keys (no LoRA, cache salt, multimodal or prompt embeds).
pub const VLLM_SHA256_CBOR_I64_SCHEME: &str = "vllm-sha256-cbor-i64-v1";
const RECONNECT_DELAY: Duration = Duration::from_secs(1);
/// Nodes heartbeat every second. A SUB socket never reports a vanished publisher,
/// so a silent node's view is dropped after this long.
const STALE_AFTER: Duration = Duration::from_secs(5);
/// Nodes are bits in a u64 mask.
const MAX_NODES: usize = 64;
const DEFAULT_MAX_INDEX_ENTRIES: usize = 32 * 1024 * 1024;
/// Credit for a page only another node holds, relative to one on the worker's node.
const DEFAULT_REMOTE_HIT_WEIGHT: f64 = 0.5;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RadixShmemRuntimeConfig {
    pub page_size: u32,
    pub hash_scheme: String,
    /// This worker's node placement stream, reachable from the frontend.
    pub kv_events_endpoint: String,
    /// Placement stream of each DP rank's node, for an instance spanning
    /// several nodes. Empty: every rank is on `kv_events_endpoint`'s node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dp_rank_endpoints: Vec<String>,
    /// vLLM's `NONE_HASH` (hex), the parent of a request's first block, for
    /// `VLLM_SHA256_CBOR_I64_SCHEME`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub none_hash: Option<String>,
}

impl RadixShmemRuntimeConfig {
    fn endpoints(&self) -> &[String] {
        if self.dp_rank_endpoints.is_empty() {
            std::slice::from_ref(&self.kv_events_endpoint)
        } else {
            &self.dp_rank_endpoints
        }
    }
}

/// How every worker keys its pages; one lookup serves them all.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Layout {
    page_size: u32,
    scheme: KeyScheme,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KeyScheme {
    Sglang,
    Vllm { none_hash: [u8; 32] },
}

impl Layout {
    fn from_config(config: &RadixShmemRuntimeConfig) -> Option<Self> {
        let scheme = match config.hash_scheme.as_str() {
            SGLANG_SHA256_I64_SCHEME => KeyScheme::Sglang,
            VLLM_SHA256_CBOR_I64_SCHEME => KeyScheme::Vllm {
                none_hash: parse_digest(config.none_hash.as_deref()?)?,
            },
            _ => return None,
        };
        Some(Self {
            page_size: config.page_size,
            scheme,
        })
    }

    /// The keys a request's full pages are stored under.
    fn page_keys(&self, tokens: &[u32]) -> Vec<i64> {
        let digests = match &self.scheme {
            KeyScheme::Sglang => sglang_page_digests(tokens, self.page_size, false),
            KeyScheme::Vllm { none_hash } => vllm_block_digests(tokens, self.page_size, none_hash),
        };
        digests.iter().map(page_key).collect()
    }
}

/// vLLM's block hashes under `--prefix-caching-hash-algo sha256_cbor` for blocks
/// without extra keys: block `i` is the SHA-256 of the canonical CBOR of
/// `[block i-1's hash (NONE_HASH for the first), [its token ids], null]`.
fn vllm_block_digests(tokens: &[u32], block_size: u32, none_hash: &[u8; 32]) -> Vec<[u8; 32]> {
    let block_size = block_size as usize;
    if block_size == 0 {
        return Vec::new();
    }
    let mut digests: Vec<[u8; 32]> = Vec::with_capacity(tokens.len() / block_size);
    let mut cbor = Vec::with_capacity(40 + 5 * block_size);
    for block in tokens.chunks_exact(block_size) {
        let parent = *digests.last().unwrap_or(none_hash);
        cbor.clear();
        cbor.push(0x83); // array(3)
        cbor_head(&mut cbor, 2, 32); // byte string: the parent hash
        cbor.extend_from_slice(&parent);
        cbor_head(&mut cbor, 4, block.len() as u64); // array: the tokens
        for &token in block {
            cbor_head(&mut cbor, 0, token as u64);
        }
        cbor.push(0xf6); // null: no extra keys
        digests.push(Sha256::digest(&cbor).into());
    }
    digests
}

/// A canonical CBOR head: the major type and the argument in its shortest form.
fn cbor_head(out: &mut Vec<u8>, major: u8, value: u64) {
    let major = major << 5;
    match value {
        0..=23 => out.push(major | value as u8),
        24..=0xff => out.extend_from_slice(&[major | 24, value as u8]),
        0x100..=0xffff => {
            out.push(major | 25);
            out.extend_from_slice(&(value as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(major | 26);
            out.extend_from_slice(&(value as u32).to_be_bytes());
        }
        _ => {
            out.push(major | 27);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }
}

fn parse_digest(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 || !hex.is_ascii() {
        return None;
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(digest)
}

/// One placement batch from a node's radix-server (radixshmem.kv_events).
#[derive(Debug, Deserialize)]
struct PlacementBatch {
    v: u32,
    incarnation: String,
    #[serde(default)]
    node: i64,
    #[serde(default)]
    events: Vec<(String, Vec<i64>)>,
}

#[derive(Default)]
struct NodeStream {
    last_seq: Option<u64>,
    incarnation: Option<String>,
}

struct Subscription {
    bit: u32,
    cancel: CancellationToken,
}

#[derive(Default)]
struct Subscriptions {
    by_endpoint: HashMap<String, Subscription>,
    free_bits: Vec<u32>,
    next_bit: u32,
}

/// Event-driven index of what each radixshmem node holds.
#[derive(Clone)]
pub struct RadixShmemSharedKvCache {
    runtime_configs: RuntimeConfigWatch,
    holders: Arc<DashMap<i64, u64>>,
    subscriptions: Arc<Mutex<Subscriptions>>,
    /// Node bit of each worker's DP ranks; rebuilt when workers or nodes change.
    worker_hosts: Arc<Mutex<Arc<SharedCacheWorkerHosts>>>,
    /// Page layout shared by all workers; `None` when they disagree or it is
    /// not recomputable here.
    layout: Arc<Mutex<Option<Layout>>>,
    page_size_warned: Arc<AtomicBool>,
    cancellation_token: CancellationToken,
    max_entries: usize,
    remote_weight: f64,
}

impl RadixShmemSharedKvCache {
    pub fn new(runtime_configs: RuntimeConfigWatch, cancellation_token: CancellationToken) -> Self {
        let max_entries = std::env::var("DYN_RADIXSHMEM_MAX_INDEX_ENTRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_INDEX_ENTRIES);
        let remote_weight = std::env::var("DYN_RADIXSHMEM_REMOTE_HIT_WEIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|w: &f64| (0.0..=1.0).contains(w))
            .unwrap_or(DEFAULT_REMOTE_HIT_WEIGHT);
        Self {
            runtime_configs,
            holders: Arc::new(DashMap::new()),
            subscriptions: Arc::new(Mutex::new(Subscriptions::default())),
            worker_hosts: Arc::new(Mutex::new(Arc::new(SharedCacheWorkerHosts::default()))),
            layout: Arc::new(Mutex::new(None)),
            page_size_warned: Arc::new(AtomicBool::new(false)),
            cancellation_token,
            max_entries,
            remote_weight,
        }
    }

    /// Subscribes to a node when its first worker appears and drops it when the last leaves.
    pub fn start_subscriber(&self) {
        let cache = self.clone();
        tokio::spawn(async move { cache.run_supervisor().await });
    }

    fn configs(&self) -> Vec<(WorkerId, RadixShmemRuntimeConfig)> {
        let workers = self.runtime_configs.borrow();
        workers
            .iter()
            .filter_map(|(worker_id, runtime_config)| {
                config_from_runtime(*worker_id, runtime_config).map(|c| (*worker_id, c))
            })
            .collect()
    }

    async fn run_supervisor(self) {
        let mut runtime_configs = self.runtime_configs.clone();
        loop {
            self.reconcile_subscriptions();
            tokio::select! {
                _ = self.cancellation_token.cancelled() => {
                    self.cancel_all();
                    return;
                }
                result = runtime_configs.changed() => {
                    if result.is_err() {
                        self.cancel_all();
                        self.holders.clear();
                        return;
                    }
                }
            }
        }
    }

    fn reconcile_subscriptions(&self) {
        let configs = self.configs();
        let wanted: HashSet<String> = configs
            .iter()
            .flat_map(|(_, c)| c.endpoints().iter().cloned())
            .filter(|e| !e.is_empty())
            .collect();
        let mut subs = self.subscriptions.lock().unwrap();
        let gone: Vec<String> = subs
            .by_endpoint
            .keys()
            .filter(|e| !wanted.contains(*e))
            .cloned()
            .collect();
        for endpoint in gone {
            if let Some(sub) = subs.by_endpoint.remove(&endpoint) {
                sub.cancel.cancel();
                self.clear_node(sub.bit);
                subs.free_bits.push(sub.bit);
                tracing::info!(%endpoint, "radixshmem node left; dropped its placement");
            }
        }
        for endpoint in wanted {
            if subs.by_endpoint.contains_key(&endpoint) {
                continue;
            }
            let bit = match subs.free_bits.pop() {
                Some(bit) => bit,
                None if (subs.next_bit as usize) < MAX_NODES => {
                    subs.next_bit += 1;
                    subs.next_bit - 1
                }
                None => {
                    tracing::warn!(%endpoint, max_nodes = MAX_NODES, "too many radixshmem nodes; ignoring this one");
                    continue;
                }
            };
            let cancel = self.cancellation_token.child_token();
            let node = self.clone();
            let (ep, token) = (endpoint.clone(), cancel.clone());
            tokio::spawn(async move { node.run_node(ep, bit, token).await });
            subs.by_endpoint
                .insert(endpoint, Subscription { bit, cancel });
        }
        let layout = shared_layout(&configs);
        *self.layout.lock().unwrap() = layout;
        let hosts = worker_hosts(&configs, &subs.by_endpoint);
        let mut current = self.worker_hosts.lock().unwrap();
        if **current != hosts {
            tracing::info!(
                ?hosts,
                "radixshmem worker nodes (worker id -> node bit per DP rank)"
            );
            *current = Arc::new(hosts);
        }
    }

    fn cancel_all(&self) {
        let mut subs = self.subscriptions.lock().unwrap();
        for (_, sub) in subs.by_endpoint.drain() {
            sub.cancel.cancel();
        }
    }

    async fn run_node(self, endpoint: String, bit: u32, cancel: CancellationToken) {
        loop {
            let mut stream = NodeStream::default();
            match connect_sub_socket(&endpoint, None).await {
                Ok(mut socket) => {
                    tracing::info!(%endpoint, bit, "subscribed to radixshmem node placement");
                    let mut deadline = tokio::time::Instant::now() + STALE_AFTER;
                    loop {
                        tokio::select! {
                            _ = cancel.cancelled() => return,
                            _ = tokio::time::sleep_until(deadline), if stream.incarnation.is_some() => {
                                record_error();
                                tracing::warn!(%endpoint, bit, "radixshmem node silent; dropping its placement until it publishes again");
                                self.clear_node(bit);
                                stream = NodeStream::default();
                            }
                            message = socket.next() => {
                                deadline = tokio::time::Instant::now() + STALE_AFTER;
                                let frames = match message {
                                    Some(Ok(frames)) => multipart_message(frames),
                                    Some(Err(error)) => {
                                        record_error();
                                        tracing::warn!(%endpoint, %error, "radixshmem placement stream failed; reconnecting");
                                        break;
                                    }
                                    None => {
                                        record_error();
                                        tracing::warn!(%endpoint, "radixshmem placement stream ended; reconnecting");
                                        break;
                                    }
                                };
                                match parse_frames(&frames) {
                                    Ok((seq, batch)) => self.apply(bit, &mut stream, seq, batch),
                                    Err(error) => {
                                        record_error();
                                        tracing::warn!(%endpoint, %error, "dropping invalid radixshmem placement frame");
                                    }
                                }
                            }
                        }
                    }
                }
                Err(error) => {
                    record_error();
                    tracing::warn!(%endpoint, %error, "failed to subscribe to radixshmem node placement; retrying");
                }
            }
            self.clear_node(bit);
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(RECONNECT_DELAY) => {}
            }
        }
    }

    fn apply(&self, bit: u32, stream: &mut NodeStream, seq: u64, batch: PlacementBatch) {
        if batch.v != 1 {
            record_error();
            tracing::warn!(
                version = batch.v,
                "unsupported radixshmem placement version"
            );
            return;
        }
        let restarted = stream.incarnation.as_deref() != Some(batch.incarnation.as_str());
        let gap = stream
            .last_seq
            .is_some_and(|last| seq != last.wrapping_add(1));
        if restarted || gap {
            // New, restarted or lost batches: relearn the node from here.
            if stream.incarnation.is_some() {
                tracing::warn!(
                    bit,
                    node = batch.node,
                    restarted,
                    gap,
                    "radixshmem placement reset for node"
                );
            }
            self.clear_node(bit);
            stream.incarnation = Some(batch.incarnation.clone());
        }
        stream.last_seq = Some(seq);
        let mask = 1u64 << bit;
        for (kind, hashes) in batch.events {
            match kind.as_str() {
                "stored" => {
                    for hash in hashes {
                        *self.holders.entry(hash).or_insert(0) |= mask;
                    }
                }
                "removed" => {
                    for hash in hashes {
                        self.holders.remove_if_mut(&hash, |_, holders| {
                            *holders &= !mask;
                            *holders == 0
                        });
                    }
                }
                "cleared" => self.clear_node(bit),
                _ => {}
            }
        }
        if self.holders.len() > self.max_entries {
            tracing::warn!(
                entries = self.holders.len(),
                max_entries = self.max_entries,
                "radixshmem placement index exceeded its size limit; cleared it"
            );
            self.holders.clear();
        }
    }

    fn clear_node(&self, bit: u32) {
        let mask = 1u64 << bit;
        self.holders.retain(|_, holders| {
            *holders &= !mask;
            *holders != 0
        });
    }

    /// Leading pages held by some node, and by each node itself (indexed by
    /// node bit), for a request's page keys.
    fn held_prefix_pages(&self, keys: &[i64]) -> (u32, Vec<u32>) {
        let mut run = 0u32;
        let mut node_runs = vec![0u32; MAX_NODES];
        // Nodes whose own run has not ended yet.
        let mut extending = u64::MAX;
        for key in keys {
            let Some(held) = self.holders.get(key).map(|h| *h) else {
                break;
            };
            run += 1;
            extending &= held;
            let mut nodes = extending;
            while nodes != 0 {
                node_runs[nodes.trailing_zeros() as usize] = run;
                nodes &= nodes - 1;
            }
        }
        (run, node_runs)
    }
}

/// The layout every worker must share for one lookup to be valid for all.
fn shared_layout(configs: &[(WorkerId, RadixShmemRuntimeConfig)]) -> Option<Layout> {
    let (_, first) = configs.first()?;
    let Some(layout) = Layout::from_config(first) else {
        tracing::warn!(
            scheme = %first.hash_scheme,
            "radixshmem hash scheme not recomputable here; skipping shared-cache lookups"
        );
        return None;
    };
    if configs
        .iter()
        .any(|(_, c)| Layout::from_config(c).as_ref() != Some(&layout))
    {
        tracing::warn!(
            "radixshmem runtime configs differ across workers; skipping shared-cache lookups"
        );
        return None;
    }
    Some(layout)
}

/// Node bit of each worker's DP ranks. A worker with a node we do not follow
/// (over `MAX_NODES`) is left out, so its hits count as local.
fn worker_hosts(
    configs: &[(WorkerId, RadixShmemRuntimeConfig)],
    subscriptions: &HashMap<String, Subscription>,
) -> SharedCacheWorkerHosts {
    configs
        .iter()
        .filter_map(|(worker_id, config)| {
            let bits = config
                .endpoints()
                .iter()
                .map(|endpoint| subscriptions.get(endpoint).map(|sub| sub.bit))
                .collect::<Option<Vec<u32>>>()?;
            Some((*worker_id, bits))
        })
        .collect()
}

/// SGLang's `hash_str_to_int64`: the digest's first 8 bytes, big-endian, signed.
fn page_key(digest: &[u8; 32]) -> i64 {
    i64::from_be_bytes(digest[..8].try_into().unwrap())
}

fn record_error() {
    if let Some(metrics) = RoutingOverheadMetrics::get() {
        metrics.inc_shared_cache_errors();
    }
}

fn parse_frames(frames: &[Vec<u8>]) -> anyhow::Result<(u64, PlacementBatch)> {
    let [_, seq, payload] = frames else {
        anyhow::bail!("expected three frames, got {}", frames.len());
    };
    let seq = u64::from_be_bytes(
        seq.as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected an 8-byte sequence frame"))?,
    );
    Ok((seq, rmp_serde::from_slice(payload)?))
}

fn config_from_runtime(
    worker_id: WorkerId,
    runtime_config: &ModelRuntimeConfig,
) -> Option<RadixShmemRuntimeConfig> {
    match runtime_config.get_engine_specific::<RadixShmemRuntimeConfig>(RADIXSHMEM_RUNTIME_KEY) {
        Ok(config) => config,
        Err(error) => {
            tracing::warn!(worker_id, %error, "failed to parse radixshmem runtime config");
            None
        }
    }
}

#[async_trait]
impl SharedKvCache for RadixShmemSharedKvCache {
    async fn check_blocks(
        &self,
        tokens: &[u32],
        block_size: u32,
        cache_namespace: Option<&str>,
    ) -> Result<SharedCacheHits, KvRouterError> {
        // A namespace enters the engine's hashes, which are recomputed here without it.
        if cache_namespace.filter(|ns| !ns.is_empty()).is_some() {
            return Ok(SharedCacheHits::default());
        }
        let Some(layout) = self.layout.lock().unwrap().clone() else {
            return Ok(SharedCacheHits::default());
        };
        let page_size = layout.page_size;
        if page_size == 0 || page_size != block_size {
            if !self.page_size_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    worker_page_size = page_size,
                    router_page_size = block_size,
                    "radixshmem page size mismatch; skipping shared-cache lookups"
                );
            }
            return Ok(SharedCacheHits::default());
        }
        let (run, node_runs) = self.held_prefix_pages(&layout.page_keys(tokens));
        tracing::debug!(
            pages = tokens.len() as u32 / page_size,
            held = run,
            node_held = ?&node_runs[..node_runs.iter().rposition(|&n| n > 0).map_or(0, |i| i + 1)],
            "radixshmem shared-cache lookup"
        );
        if run == 0 {
            return Ok(SharedCacheHits::default());
        }
        let worker_hosts = self.worker_hosts.lock().unwrap().clone();
        Ok(
            SharedCacheHits::from_ranges(vec![0..run]).with_locality(SharedCacheLocality {
                host_prefix_blocks: node_runs,
                worker_hosts,
                remote_weight: self.remote_weight,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::shared_cache::sglang_page_digests;

    fn cache() -> RadixShmemSharedKvCache {
        let (_tx, rx) = tokio::sync::watch::channel(HashMap::new());
        RadixShmemSharedKvCache::new(rx, CancellationToken::new())
    }

    fn batch(incarnation: &str, events: Vec<(&str, Vec<i64>)>) -> PlacementBatch {
        PlacementBatch {
            v: 1,
            incarnation: incarnation.into(),
            node: 0,
            events: events.into_iter().map(|(k, h)| (k.into(), h)).collect(),
        }
    }

    #[test]
    fn page_key_matches_sglang_hash_str_to_int64() {
        // hash_str_to_int64("ffffffffffffffff...") == -1; "00..01..." == 1
        let mut d = [0u8; 32];
        d[..8].copy_from_slice(&[0xff; 8]);
        assert_eq!(page_key(&d), -1);
        let mut d = [0u8; 32];
        d[7] = 1;
        assert_eq!(page_key(&d), 1);
    }

    #[test]
    fn a_copy_evicted_on_one_node_stays_held_by_the_other() {
        let c = cache();
        let (mut a, mut b) = (NodeStream::default(), NodeStream::default());
        c.apply(0, &mut a, 0, batch("a", vec![("stored", vec![1, 2])]));
        c.apply(1, &mut b, 0, batch("b", vec![("stored", vec![2, 3])]));
        c.apply(0, &mut a, 1, batch("a", vec![("removed", vec![2])]));
        assert_eq!(*c.holders.get(&2).unwrap(), 0b10);
        c.apply(1, &mut b, 1, batch("b", vec![("removed", vec![2])]));
        assert!(!c.holders.contains_key(&2));
        assert_eq!(*c.holders.get(&1).unwrap(), 0b01);
    }

    #[test]
    fn duplicate_stores_and_unknown_removes_are_harmless() {
        let c = cache();
        let mut a = NodeStream::default();
        c.apply(
            0,
            &mut a,
            0,
            batch("a", vec![("stored", vec![5]), ("stored", vec![5])]),
        );
        c.apply(0, &mut a, 1, batch("a", vec![("removed", vec![99])]));
        assert_eq!(c.holders.len(), 1);
    }

    #[test]
    fn gap_restart_or_cleared_forgets_only_that_node() {
        let c = cache();
        let (mut a, mut b) = (NodeStream::default(), NodeStream::default());
        c.apply(0, &mut a, 0, batch("a", vec![("stored", vec![1])]));
        c.apply(1, &mut b, 0, batch("b", vec![("stored", vec![1, 7])]));
        // seq 1 lost: node a is reset before applying seq 2
        c.apply(0, &mut a, 2, batch("a", vec![("stored", vec![4])]));
        assert_eq!(*c.holders.get(&1).unwrap(), 0b10);
        assert_eq!(*c.holders.get(&4).unwrap(), 0b01);
        // node a restarted
        c.apply(0, &mut a, 0, batch("a2", vec![]));
        assert!(!c.holders.contains_key(&4));
        // node b cleared
        c.apply(1, &mut b, 1, batch("b", vec![("cleared", vec![])]));
        assert!(c.holders.is_empty());
    }

    #[test]
    fn lookup_counts_only_the_leading_held_pages() {
        let c = cache();
        let tokens: Vec<u32> = (0..12).collect();
        let keys: Vec<i64> = sglang_page_digests(&tokens, 4, false)
            .iter()
            .map(page_key)
            .collect();
        let mut a = NodeStream::default();
        // pages 0 and 2 held, page 1 missing: only page 0 is usable
        c.apply(
            0,
            &mut a,
            0,
            batch("a", vec![("stored", vec![keys[0], keys[2]])]),
        );
        assert_eq!(c.held_prefix_pages(&keys).0, 1);
        c.apply(0, &mut a, 1, batch("a", vec![("stored", vec![keys[1]])]));
        assert_eq!(c.held_prefix_pages(&keys).0, 3);
        // the trailing partial page never counts
        let sglang = Layout {
            page_size: 4,
            scheme: KeyScheme::Sglang,
        };
        assert_eq!(c.held_prefix_pages(&sglang.page_keys(&tokens[..11])).0, 2);
    }

    #[test]
    fn lookup_reports_each_nodes_own_run() {
        let c = cache();
        let tokens: Vec<u32> = (0..16).collect();
        let keys: Vec<i64> = sglang_page_digests(&tokens, 4, false)
            .iter()
            .map(page_key)
            .collect();
        let (mut a, mut b) = (NodeStream::default(), NodeStream::default());
        // node 0 holds pages 0..1, node 1 holds pages 0..3
        c.apply(
            0,
            &mut a,
            0,
            batch("a", vec![("stored", keys[..1].to_vec())]),
        );
        c.apply(
            1,
            &mut b,
            0,
            batch(
                "b",
                keys[..3].iter().map(|k| ("stored", vec![*k])).collect(),
            ),
        );
        let (run, node_runs) = c.held_prefix_pages(&keys);
        assert_eq!(run, 3);
        assert_eq!(&node_runs[..3], &[1, 3, 0]);
    }

    #[test]
    fn workers_map_to_the_node_of_each_dp_rank() {
        let config = |endpoint: &str, ranks: &[&str]| RadixShmemRuntimeConfig {
            page_size: 4,
            hash_scheme: SGLANG_SHA256_I64_SCHEME.into(),
            kv_events_endpoint: endpoint.into(),
            dp_rank_endpoints: ranks.iter().map(|e| e.to_string()).collect(),
            none_hash: None,
        };
        let sub = |bit| Subscription {
            bit,
            cancel: CancellationToken::new(),
        };
        let subs = HashMap::from([
            ("tcp://a".to_string(), sub(0)),
            ("tcp://b".to_string(), sub(1)),
        ]);
        let hosts = worker_hosts(
            &[
                (1, config("tcp://b", &[])),
                (
                    2,
                    config("tcp://a", &["tcp://a", "tcp://a", "tcp://b", "tcp://b"]),
                ),
                (3, config("tcp://c", &[])),
            ],
            &subs,
        );
        assert_eq!(hosts[&1], vec![1]);
        assert_eq!(hosts[&2], vec![0, 0, 1, 1]);
        assert!(!hosts.contains_key(&3));
    }

    #[test]
    fn reads_the_runtime_data_workers_publish() {
        // As dynamo.common.radixshmem.publish_radixshmem writes it.
        let mut runtime = ModelRuntimeConfig::default();
        runtime
            .set_engine_specific(
                "radixshmem",
                serde_json::json!({
                    "page_size": 64,
                    "hash_scheme": SGLANG_SHA256_I64_SCHEME,
                    "kv_events_endpoint": "tcp://a",
                }),
            )
            .unwrap();
        let config = config_from_runtime(1, &runtime).unwrap();
        assert_eq!(config.page_size, 64);
        assert_eq!(config.kv_events_endpoint, "tcp://a");
    }

    #[test]
    fn vllm_keys_match_vllm() {
        // vllm.v1.core.kv_cache_utils with sha256_cbor and the default seed:
        // NONE_HASH, then hash_block_tokens over blocks of 4 with no extra keys.
        let none_hash = "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0";
        let tokens = [
            0,
            23,
            24,
            255,
            256,
            65535,
            65536,
            151643,
            7,
            1000,
            u32::MAX,
            42,
        ];
        let config = RadixShmemRuntimeConfig {
            page_size: 4,
            hash_scheme: VLLM_SHA256_CBOR_I64_SCHEME.into(),
            kv_events_endpoint: "tcp://a".into(),
            dp_rank_endpoints: vec![],
            none_hash: Some(none_hash.into()),
        };
        let layout = Layout::from_config(&config).unwrap();
        assert_eq!(
            layout.page_keys(&tokens),
            [
                -2558115152377755722,
                -5707059813888821707,
                9074829146518922321
            ]
        );
        // the trailing partial block never counts
        assert_eq!(layout.page_keys(&tokens[..11]).len(), 2);
        // the scheme cannot be recomputed without the worker's NONE_HASH
        let config = RadixShmemRuntimeConfig {
            none_hash: None,
            ..config
        };
        assert!(Layout::from_config(&config).is_none());
        assert!(shared_layout(&[(1, config)]).is_none());
    }

    #[test]
    fn parses_the_radixshmem_wire_batch() {
        #[derive(Serialize)]
        struct Wire<'a> {
            v: u32,
            cluster_id: &'a str,
            node: i64,
            node_name: &'a str,
            incarnation: &'a str,
            hash_scheme: &'a str,
            block_size: u32,
            events: Vec<(&'a str, Vec<i64>)>,
        }
        let payload = rmp_serde::to_vec_named(&Wire {
            v: 1,
            cluster_id: "c",
            node: 3,
            node_name: "n3",
            incarnation: "x",
            hash_scheme: SGLANG_SHA256_I64_SCHEME,
            block_size: 256,
            events: vec![("stored", vec![-1, 2]), ("cleared", vec![])],
        })
        .unwrap();
        let frames = vec![b"".to_vec(), 7u64.to_be_bytes().to_vec(), payload];
        let (seq, batch) = parse_frames(&frames).unwrap();
        assert_eq!(seq, 7);
        assert_eq!(batch.node, 3);
        assert_eq!(batch.events[0], ("stored".to_string(), vec![-1, 2]));
        assert_eq!(batch.events[1].0, "cleared");
    }
}
