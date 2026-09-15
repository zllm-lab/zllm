//! event 感知 TLSF arena：offset-only、bin 只索引 FreeReady node、
//! free 挂完成证明进 pending、promote 时级联 coalesce。
//!
//! 算法骨架参照 OffsetAllocator（TLSF 家族，256 浮点间隔 bin，内部碎片
//! 最坏 +12.5%）；对 GPU 时序语义的核心改造：node 三态
//! `Used / FreePending / FreeReady`，仅 Ready 参与 bin 与合并。

use super::Completion;

pub type NodeIndex = u32;
pub const NODE_UNUSED: NodeIndex = u32::MAX;

/// 分配粒度：arena 内一切尺寸以 ELEMENT 为单位（u32 element 覆盖 1 TiB 段）。
pub const ELEMENT_BYTES: u64 = 256;

const MANTISSA_BITS: u32 = 3;
const LEAF_MASK: u32 = 0x7;
const NUM_TOP: usize = 32;
const NUM_LEAF: usize = NUM_TOP * 8;
/// bin 0..15 精确映射 element 0..15；bin 16..31 是浮点段的空洞槽位，永不使用。
const EXACT_BINS: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeState {
    Used,
    FreePending,
    FreeReady,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    data_offset: u32,
    data_size: u32,
    bin_next: NodeIndex,
    bin_prev: NodeIndex,
    nbr_prev: NodeIndex,
    nbr_next: NodeIndex,
    state: NodeState,
}

impl Node {
    const fn vacant() -> Self {
        Self { data_offset: 0, data_size: 0, bin_next: NODE_UNUSED, bin_prev: NODE_UNUSED, nbr_prev: NODE_UNUSED, nbr_next: NODE_UNUSED, state: NodeState::Used }
    }
}

/// 一次 arena 分配。`bytes` 是实际容量；node 耗尽时会放弃 split，
/// 因而可能大于请求字节数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Allocation {
    pub node: NodeIndex,
    pub offset_bytes: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    pub allocate_calls: u64,
    pub allocate_full: u64,
    pub splits: u64,
    pub promotes: u64,
    pub coalesces: u64,
    pub coalesced_elements: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageReport {
    pub total_bytes: u64,
    pub ready_bytes: u64,
    pub pending_bytes: u64,
    pub used_bytes: u64,
    /// 最大空闲区间下界（最高非空 bin 的代表尺寸；同 bin 可能挂更大块）。
    pub largest_free_region_bytes: u64,
    pub ready_regions: usize,
}

/// bin 代表尺寸（该 bin 的下界尺寸），element 单位。
fn float_to_uint(bin: u32) -> u32 {
    if bin < EXACT_BINS {
        return bin;
    }
    debug_assert!(bin >= 32, "bin {bin} 落在空洞槽位");
    let e = bin >> MANTISSA_BITS;
    let m = bin & LEAF_MASK;
    (8 | m) << (e - MANTISSA_BITS)
}

/// 最大 bin，其代表尺寸 ≤ size。
fn round_down(size: u32) -> u32 {
    if size < EXACT_BINS {
        return size;
    }
    let e = 31 - size.leading_zeros();
    (e << MANTISSA_BITS) | ((size >> (e - MANTISSA_BITS)) & LEAF_MASK)
}

/// 最小 bin，其代表尺寸 ≥ size。超出 bin 表顶部（>960 GiB 等效）时钳到
/// 顶 bin——段尺寸上限在 [`Arena::new`] 断言，正常分配不可达。
fn round_up(size: u32) -> u32 {
    if size < EXACT_BINS {
        return size;
    }
    let bin = round_down(size);
    if bin >= (NUM_LEAF as u32) - 1 {
        return (NUM_LEAF as u32) - 1;
    }
    if float_to_uint(bin) < size { bin + 1 } else { bin }
}

/// event 感知 TLSF arena。`C` 是 backend 注入的完成证明。
///
/// 线程安全由调用方保证（预期外层 per-device Mutex）。
pub struct Arena<C> {
    size_elements: u32,
    nodes: Vec<Node>,
    free_nodes: Vec<NodeIndex>,
    used_bins_top: u32,
    used_bins: [u8; NUM_TOP],
    bin_heads: [NodeIndex; NUM_LEAF],
    pending: Vec<(NodeIndex, C)>,
    ready_storage: u64,
    pending_storage: u64,
    stats: ArenaStats,
}

impl<C: Completion> Arena<C> {
    /// `size_bytes` 段总字节（向下取整到 ELEMENT 粒度）；`max_nodes` 上限
    /// 决定元数据内存（每 node 32B）。段内最多 `max_nodes` 个并发区间。
    pub fn new(size_bytes: u64, max_nodes: usize) -> Self {
        let size_elements = u32::try_from(size_bytes / ELEMENT_BYTES).expect("arena 段超过 1 TiB 上限");
        assert!(size_elements > 0, "arena 段尺寸不能为 0");
        assert!(size_elements <= float_to_uint((NUM_LEAF as u32) - 1), "arena 段超过 bin 表可表示上限（960 GiB）");
        assert!(max_nodes > 0, "arena max_nodes 不能为 0");
        let mut arena = Self {
            size_elements,
            nodes: vec![Node::vacant(); max_nodes],
            free_nodes: (0..max_nodes as NodeIndex).rev().collect(),
            used_bins_top: 0,
            used_bins: [0; NUM_TOP],
            bin_heads: [NODE_UNUSED; NUM_LEAF],
            pending: Vec::new(),
            ready_storage: 0,
            pending_storage: 0,
            stats: ArenaStats::default(),
        };
        let root = arena.take_node().expect("max_nodes>0 必有空闲 node");
        arena.nodes[root as usize].data_size = size_elements;
        arena.insert_free_node(root, size_elements, 0);
        arena
    }

    pub fn stats(&self) -> ArenaStats {
        self.stats
    }

    fn take_node(&mut self) -> Option<NodeIndex> {
        self.free_nodes.pop()
    }

    fn set_bin_bit(&mut self, bin: u32) {
        self.used_bins_top |= 1 << (bin >> MANTISSA_BITS);
        self.used_bins[(bin >> MANTISSA_BITS) as usize] |= 1 << (bin & LEAF_MASK);
    }

    fn clear_bin_bit(&mut self, bin: u32) {
        let top = (bin >> MANTISSA_BITS) as usize;
        self.used_bins[top] &= !(1 << (bin & LEAF_MASK));
        if self.used_bins[top] == 0 {
            self.used_bins_top &= !(1 << (bin >> MANTISSA_BITS));
        }
    }

    /// 定位代表尺寸 ≥ 请求的最优非空 bin（2×trailing_zeros，O(1)）。
    fn find_bin(&self, min_bin: u32) -> Option<u32> {
        let top = min_bin >> MANTISSA_BITS;
        let leaf = min_bin & LEAF_MASK;
        if (self.used_bins_top & (1 << top)) != 0 {
            let mask = self.used_bins[top as usize] & (u8::MAX << leaf);
            if mask != 0 {
                return Some((top << MANTISSA_BITS) | mask.trailing_zeros());
            }
        }
        let higher = if top + 1 >= 32 { 0 } else { self.used_bins_top & (u32::MAX << (top + 1)) };
        if higher == 0 {
            return None;
        }
        let top = higher.trailing_zeros();
        Some((top << MANTISSA_BITS) | self.used_bins[top as usize].trailing_zeros())
    }

    /// 把状态已是 FreeReady 的 node 按 (size, offset) 挂进 bin。
    fn insert_free_node(&mut self, index: NodeIndex, size: u32, offset: u32) {
        let bin = round_down(size);
        let old_head = self.bin_heads[bin as usize];
        {
            let node = &mut self.nodes[index as usize];
            node.data_offset = offset;
            node.data_size = size;
            node.state = NodeState::FreeReady;
            node.bin_prev = NODE_UNUSED;
            node.bin_next = old_head;
        }
        if old_head != NODE_UNUSED {
            self.nodes[old_head as usize].bin_prev = index;
        }
        self.bin_heads[bin as usize] = index;
        self.set_bin_bit(bin);
        self.ready_storage += u64::from(size);
    }

    /// 从 bin 摘除 node（不改动其物理相邻链与 state）。
    fn remove_from_bin(&mut self, index: NodeIndex) {
        let node = self.nodes[index as usize];
        debug_assert_eq!(node.state, NodeState::FreeReady);
        let bin = round_down(node.data_size);
        if node.bin_prev != NODE_UNUSED {
            self.nodes[node.bin_prev as usize].bin_next = node.bin_next;
        } else {
            self.bin_heads[bin as usize] = node.bin_next;
            if node.bin_next == NODE_UNUSED {
                self.clear_bin_bit(bin);
            }
        }
        if node.bin_next != NODE_UNUSED {
            self.nodes[node.bin_next as usize].bin_prev = node.bin_prev;
        }
        self.ready_storage -= u64::from(node.data_size);
    }

    /// 分配 `bytes`（向上取整到 ELEMENT 粒度）。段满或 node 耗尽返回 None。
    pub fn allocate(&mut self, bytes: u64) -> Option<Allocation> {
        self.stats.allocate_calls += 1;
        let elements = u32::try_from(bytes.div_ceil(ELEMENT_BYTES)).ok()?;
        let Some(bin) = self.find_bin(round_up(elements)) else {
            self.stats.allocate_full += 1;
            return None;
        };
        let index = self.bin_heads[bin as usize];
        debug_assert!(index != NODE_UNUSED);
        self.remove_from_bin(index);
        let total = self.nodes[index as usize].data_size;
        let offset = self.nodes[index as usize].data_offset;
        self.nodes[index as usize].state = NodeState::Used;
        let remainder = total - elements;
        // node 耗尽时放弃 split，整块给出（内部碎片由 capacity 账本承接）。
        if remainder > 0
            && let Some(rest) = self.take_node()
        {
            self.stats.splits += 1;
            self.nodes[index as usize].data_size = elements;
            let old_next = self.nodes[index as usize].nbr_next;
            self.nodes[rest as usize].nbr_prev = index;
            self.nodes[rest as usize].nbr_next = old_next;
            if old_next != NODE_UNUSED {
                self.nodes[old_next as usize].nbr_prev = rest;
            }
            self.nodes[index as usize].nbr_next = rest;
            self.insert_free_node(rest, remainder, offset + elements);
        }
        let capacity = u64::from(self.nodes[index as usize].data_size) * ELEMENT_BYTES;
        Some(Allocation { node: index, offset_bytes: u64::from(offset) * ELEMENT_BYTES, bytes: capacity })
    }

    /// 释放并挂完成证明；设备完成前不参与复用与合并。
    pub fn free(&mut self, index: NodeIndex, completion: C) {
        let node = &mut self.nodes[index as usize];
        debug_assert_eq!(node.state, NodeState::Used, "double free 或状态错误");
        node.state = NodeState::FreePending;
        self.pending_storage += u64::from(node.data_size);
        self.pending.push((index, completion));
    }

    /// 特快路径：调用方已证明设备完成（如 stage completion 蕴含），直接
    /// 置 Ready 并尝试级联合并。
    pub fn free_ready(&mut self, index: NodeIndex) {
        debug_assert_eq!(self.nodes[index as usize].state, NodeState::Used);
        self.make_ready(index);
    }

    /// 晋升已完成 completion 的 pending node；返回本次晋升数。
    pub fn promote(&mut self) -> usize {
        let mut promoted = 0;
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].1.is_complete() {
                let (index, _) = self.pending.swap_remove(i);
                self.pending_storage -= u64::from(self.nodes[index as usize].data_size);
                self.make_ready(index);
                promoted += 1;
            } else {
                i += 1;
            }
        }
        self.stats.promotes += promoted as u64;
        promoted
    }

    /// 置 Ready 并与两侧 Ready 物理邻居级联合并，最终进 bin。
    fn make_ready(&mut self, index: NodeIndex) {
        let mut offset = self.nodes[index as usize].data_offset;
        let mut size = self.nodes[index as usize].data_size;
        let prev = self.nodes[index as usize].nbr_prev;
        if prev != NODE_UNUSED && self.nodes[prev as usize].state == NodeState::FreeReady {
            self.remove_from_bin(prev);
            offset = self.nodes[prev as usize].data_offset;
            size += self.nodes[prev as usize].data_size;
            self.stats.coalesces += 1;
            self.stats.coalesced_elements += u64::from(self.nodes[prev as usize].data_size);
            self.nodes[index as usize].nbr_prev = self.nodes[prev as usize].nbr_prev;
            let grand = self.nodes[prev as usize].nbr_prev;
            if grand != NODE_UNUSED {
                self.nodes[grand as usize].nbr_next = index;
            }
            self.nodes[prev as usize] = Node::vacant();
            self.free_nodes.push(prev);
        }
        let next = self.nodes[index as usize].nbr_next;
        if next != NODE_UNUSED && self.nodes[next as usize].state == NodeState::FreeReady {
            self.remove_from_bin(next);
            size += self.nodes[next as usize].data_size;
            self.stats.coalesces += 1;
            self.stats.coalesced_elements += u64::from(self.nodes[next as usize].data_size);
            self.nodes[index as usize].nbr_next = self.nodes[next as usize].nbr_next;
            let grand = self.nodes[next as usize].nbr_next;
            if grand != NODE_UNUSED {
                self.nodes[grand as usize].nbr_prev = index;
            }
            self.nodes[next as usize] = Node::vacant();
            self.free_nodes.push(next);
        }
        self.insert_free_node(index, size, offset);
    }

    pub fn storage_report(&self) -> StorageReport {
        let largest = if self.used_bins_top == 0 {
            0
        } else {
            let top = 31 - self.used_bins_top.leading_zeros();
            let leaf = 7 - self.used_bins[top as usize].leading_zeros();
            u64::from(float_to_uint((top << MANTISSA_BITS) | leaf)) * ELEMENT_BYTES
        };
        let ready = self.ready_storage * ELEMENT_BYTES;
        let pending = self.pending_storage * ELEMENT_BYTES;
        StorageReport {
            total_bytes: u64::from(self.size_elements) * ELEMENT_BYTES,
            ready_bytes: ready,
            pending_bytes: pending,
            used_bytes: u64::from(self.size_elements) * ELEMENT_BYTES - ready - pending,
            largest_free_region_bytes: largest,
            ready_regions: self.nodes.iter().filter(|node| node.state == NodeState::FreeReady).count(),
        }
    }

    /// 调试用：live（非 freelist）node 数。
    pub fn live_node_count(&self) -> usize {
        self.nodes.len() - self.free_nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone)]
    struct MockCompletion(Arc<AtomicBool>);
    impl MockCompletion {
        fn ready() -> Self {
            Self(Arc::new(AtomicBool::new(true)))
        }
        fn pending() -> Self {
            Self(Arc::new(AtomicBool::new(false)))
        }
        fn fire(&self) {
            self.0.store(true, Ordering::Release);
        }
    }
    impl Completion for MockCompletion {
        fn is_complete(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    const MIB: u64 = 1 << 20;

    #[test]
    fn smallfloat_monotonic_and_bounded() {
        let mut prev_bin = 0u32;
        for v in 1u32..=1_000_000 {
            let up = round_up(v);
            let down = round_down(v);
            assert!(up >= prev_bin || v < EXACT_BINS, "round_up 单调性 v={v}");
            prev_bin = up;
            assert!(up < EXACT_BINS || up >= 32, "bin {up} 不得落入空洞");
            assert!(float_to_uint(down) <= v, "round_down 下界 v={v}");
            assert!(float_to_uint(up) >= v, "round_up 上界 v={v}");
            let waste = float_to_uint(up) - v;
            assert!(waste * 8 <= v, "内碎超 12.5%: v={v} bin_size={}", float_to_uint(up));
        }
        for &v in &[1 << 20, (1 << 20) + 1, 15, 16, 17, 30, 31, 32, 33] {
            let up = round_up(v);
            assert!(float_to_uint(round_down(v)) <= v && float_to_uint(up) >= v);
        }
        // 超出 bin 表顶部的尺寸钳到顶 bin（不可达，段上限已断言）。
        assert_eq!(round_up(u32::MAX), (NUM_LEAF as u32) - 1);
        assert_eq!(round_up(u32::MAX - 1), (NUM_LEAF as u32) - 1);
    }

    #[test]
    fn alloc_free_reuse() {
        let mut arena: Arena<MockCompletion> = Arena::new(4 * MIB, 64);
        let a = arena.allocate(MIB).expect("分配");
        assert_eq!(a.offset_bytes, 0);
        assert!(arena.allocate(4 * MIB).is_none(), "段满必须失败");
        arena.free(a.node, MockCompletion::ready());
        assert_eq!(arena.promote(), 1);
        let b = arena.allocate(MIB).expect("复用");
        assert_eq!(b.offset_bytes, 0, "同尺寸应复用同区间");
    }

    #[test]
    fn split_remainder_reusable() {
        let mut arena: Arena<MockCompletion> = Arena::new(4 * MIB, 64);
        let _a = arena.allocate(MIB).expect("a");
        let b = arena.allocate(MIB).expect("b");
        let c = arena.allocate(MIB).expect("c");
        assert_eq!(b.offset_bytes, MIB);
        assert_eq!(c.offset_bytes, 2 * MIB);
        assert_eq!(arena.stats().splits, 3);
    }

    #[test]
    fn coalesce_after_promote() {
        let mut arena: Arena<MockCompletion> = Arena::new(4 * MIB, 64);
        let a = arena.allocate(MIB).expect("a");
        let b = arena.allocate(MIB).expect("b");
        let c = arena.allocate(MIB).expect("c");
        arena.free(a.node, MockCompletion::ready());
        arena.free(c.node, MockCompletion::ready());
        arena.promote();
        assert!(arena.allocate(3 * MIB).is_none(), "B 未释放，3MiB 连续区间不存在");
        arena.free(b.node, MockCompletion::ready());
        arena.promote();
        let big = arena.allocate(3 * MIB).expect("三区间合并后应可分配");
        assert_eq!(big.offset_bytes, 0);
        assert!(arena.stats().coalesces >= 2);
    }

    #[test]
    fn pending_neighbor_blocks_coalesce() {
        let mut arena: Arena<MockCompletion> = Arena::new(4 * MIB, 64);
        let a = arena.allocate(2 * MIB).expect("a");
        let b = arena.allocate(2 * MIB).expect("b");
        let pending = MockCompletion::pending();
        arena.free(a.node, pending.clone());
        arena.free(b.node, MockCompletion::ready());
        arena.promote();
        let report = arena.storage_report();
        assert_eq!(report.largest_free_region_bytes, 2 * MIB, "pending 邻居不得合并");
        pending.fire();
        arena.promote();
        assert_eq!(arena.storage_report().largest_free_region_bytes, 4 * MIB, "完成后应合并回整段");
    }

    #[test]
    fn free_ready_fast_path() {
        let mut arena: Arena<MockCompletion> = Arena::new(2 * MIB, 64);
        let a = arena.allocate(MIB).expect("a");
        let b = arena.allocate(MIB).expect("b");
        arena.free_ready(a.node);
        arena.free_ready(b.node);
        let report = arena.storage_report();
        assert_eq!(report.ready_bytes, 2 * MIB);
        assert_eq!(report.largest_free_region_bytes, 2 * MIB, "两块应合并为完整段");
    }

    #[test]
    fn exhausted_nodes_give_whole_block() {
        let mut arena: Arena<MockCompletion> = Arena::new(4 * MIB, 2);
        let a = arena.allocate(MIB).expect("a");
        let b = arena.allocate(MIB).expect("node 耗尽，整块给出");
        assert!(b.offset_bytes >= MIB);
        assert_eq!(b.bytes, 3 * MIB, "node 耗尽时必须报告整块实际容量");
        arena.free_ready(a.node);
        arena.free_ready(b.node);
        let report = arena.storage_report();
        assert_eq!(report.ready_bytes, 4 * MIB, "无泄漏");
    }

    /// xorshift64，避免测试引入 rand 依赖。
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn random_alloc_free_model_check() {
        let segment = 64 * MIB;
        let mut arena: Arena<MockCompletion> = Arena::new(segment, 4096);
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut live: Vec<(Allocation, MockCompletion)> = Vec::new();
        let mut retired: Vec<MockCompletion> = Vec::new();
        for round in 0..4000 {
            if !live.is_empty() && rng.below(100) < 45 {
                let idx = rng.below(live.len() as u64) as usize;
                let (alloc, completion) = live.swap_remove(idx);
                retired.push(completion.clone());
                arena.free(alloc.node, completion);
            } else {
                let size = (1 + rng.below(4096)) * ELEMENT_BYTES;
                if let Some(alloc) = arena.allocate(size) {
                    for (other, _) in &live {
                        let end = alloc.offset_bytes + alloc.bytes.max(ELEMENT_BYTES);
                        let other_end = other.offset_bytes + other.bytes.max(ELEMENT_BYTES);
                        assert!(end <= other.offset_bytes || other_end <= alloc.offset_bytes, "区间重叠: {alloc:?} vs {other:?} round={round}");
                    }
                    let completion = if rng.below(100) < 70 { MockCompletion::ready() } else { MockCompletion::pending() };
                    live.push((alloc, completion));
                }
            }
            if round % 7 == 0 {
                for (_, c) in &live {
                    c.fire();
                }
                arena.promote();
            }
        }
        for (_, c) in &live {
            c.fire();
        }
        for c in &retired {
            c.fire();
        }
        let live_nodes_before = arena.live_node_count();
        for (alloc, _) in live.drain(..) {
            arena.free(alloc.node, MockCompletion::ready());
        }
        arena.promote();
        let report = arena.storage_report();
        assert_eq!(report.used_bytes, 0, "全部释放后 used 必须为 0");
        assert_eq!(report.largest_free_region_bytes, segment, "全部释放后必须合并回整段");
        assert!(arena.live_node_count() <= live_nodes_before);
    }

    #[test]
    fn report_accounting() {
        let mut arena: Arena<MockCompletion> = Arena::new(16 * MIB, 256);
        let a = arena.allocate(4 * MIB).expect("a");
        let pending = MockCompletion::pending();
        let b = arena.allocate(4 * MIB).expect("b");
        arena.free(a.node, MockCompletion::ready());
        arena.free(b.node, pending);
        arena.promote();
        let report = arena.storage_report();
        assert_eq!(report.total_bytes, 16 * MIB);
        assert_eq!(report.pending_bytes, 4 * MIB);
        assert_eq!(report.ready_bytes + report.pending_bytes + report.used_bytes, report.total_bytes);
    }
}
