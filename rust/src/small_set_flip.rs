use petgraph::graph::{NodeIndex, DiGraph};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use enum_as_inner::EnumAsInner;

use crate::error_correcting_code::{TannerGraph, tanner_graph_edge_orientation, TannerGraphNode, Decoder};

#[derive(Debug, Clone)]
struct CheckNode {
    idx : usize
}

#[derive(Debug, Clone)]
struct BitNode {
    idx : usize,
    flip_set_size : i32,
}

#[derive(Debug, Clone, EnumAsInner)]
enum SsfTannerGraphNode {
    CheckNode(CheckNode),
    BitNode(BitNode),
}

#[derive(Debug, Clone)]
pub struct SmallSetFlip {
    tanner_graph : DiGraph<SsfTannerGraphNode, ()>,
    flip_set_size_heap : BinaryHeap<FlipSizeHeapElement>,
    check_node_count : usize,
    bit_node_count : usize,
    // Heap to keep track of largest flip variable
    // Use same stale element heap tracking trick as UFD
}

#[derive(Debug, Clone, PartialEq)]
struct FlipSizeHeapElement {
    node_idx : NodeIndex,
    flipped_set_size : i32,
}

impl Eq for FlipSizeHeapElement { }

impl PartialOrd for FlipSizeHeapElement {
    fn partial_cmp(&self, other : &FlipSizeHeapElement) -> Option<Ordering> {
        self.flipped_set_size.partial_cmp(&other.flipped_set_size)
    }
}

impl Ord for FlipSizeHeapElement {
    fn cmp(&self, other : &FlipSizeHeapElement) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl SmallSetFlip {
    pub fn new(tanner_graph : &TannerGraph) -> SmallSetFlip {
        assert!(tanner_graph_edge_orientation(tanner_graph));

        let check_node_count = tanner_graph.node_indices().filter_map(|node_idx| tanner_graph[node_idx].as_check_node()).count();
        let bit_node_count = tanner_graph.node_indices().filter_map(|node_idx| tanner_graph[node_idx].as_bit_node()).count();


        let ssf_tanner_graph = tanner_graph.map(|node_index, weight| {
            match weight {
                TannerGraphNode::CheckNode(idx) => SsfTannerGraphNode::CheckNode(CheckNode{idx:*idx}),
                TannerGraphNode::BitNode(idx) => SsfTannerGraphNode::BitNode(BitNode{idx:*idx, flip_set_size:0}),
            }
        }, |_, _| ());

        SmallSetFlip{bit_node_count, check_node_count, flip_set_size_heap:BinaryHeap::new(), tanner_graph:ssf_tanner_graph}
    }

    /// Returns the signed size of the flip set (Net number of data nodes flipped from non-trivial to trivial)
    fn check_flip_set_size(self : &Self, bit_node_idx : NodeIndex, syndrome : &Vec<bool>) -> i32 {
        self.tanner_graph.neighbors_undirected(bit_node_idx).map(|neighbor_idx| if syndrome[self.tanner_graph[neighbor_idx].as_check_node().unwrap().idx] { -1 } else { 1 }).sum()
    }

    /// Subroutine to update the flip set sizes when decoding
    fn update_flip_set_sizes(self : &mut Self, node_idx : NodeIndex, syndrome : &Vec<bool>) {
        // TODO: we can remove one layer of pointer chasing with an updatable priority queue
        // Walk neighbors
        let mut neighbor_walker = self.tanner_graph.neighbors_undirected(node_idx).detach();
        while let Some((_, neighbor_check_node_idx)) = neighbor_walker.next(&self.tanner_graph)  {
            // Walk neighbors of neighbors
            let mut neighbor_of_neighbor_walker = self.tanner_graph.neighbors_undirected(neighbor_check_node_idx).detach();
            while let Some((_, neighor_data_node_idx)) = neighbor_of_neighbor_walker.next(&self.tanner_graph)  {
            
                // Update flip set sizes
                let flip_set_size = self.check_flip_set_size(neighor_data_node_idx, &syndrome);
                self.tanner_graph[neighor_data_node_idx].as_bit_node_mut().unwrap().flip_set_size = flip_set_size;

                // Push onto the queue if they're a candidate for a future update
                if flip_set_size > 0 {
                    self.flip_set_size_heap.push(FlipSizeHeapElement {
                        node_idx:neighor_data_node_idx, flipped_set_size:flip_set_size
                    });
                }
            }
        }
    }

    /// Initialize flip set size heap
    fn init_flip_set_size_heap(self : &mut Self, syndrome : &Vec<bool>) {
        self.flip_set_size_heap.clear();
        for node_idx in self.tanner_graph.node_indices() {
            if let SsfTannerGraphNode::BitNode(BitNode {idx, flip_set_size:_}) = self.tanner_graph[node_idx] {
                let new_flip_set_size = self.check_flip_set_size(node_idx, syndrome);

                self.tanner_graph[node_idx] = SsfTannerGraphNode::BitNode(BitNode {idx, flip_set_size:new_flip_set_size});

                self.flip_set_size_heap.push(FlipSizeHeapElement {
                    node_idx, flipped_set_size:new_flip_set_size
                });
            }
        }
    }
}

impl Decoder for SmallSetFlip { 
    fn correct_syndrome(self : &mut Self, syndrome : &mut Vec<bool>, correction : &mut Vec<bool>) {
        assert!(syndrome.len() == self.check_node_count);
        
        // Initialize correction vector and flip set size heap
        correction.resize(self.bit_node_count, false);
        correction.fill(false);

        self.init_flip_set_size_heap(syndrome);

        // ====== Inner loop ======
        // While there are candidate bits to flip
        while let Some(FlipSizeHeapElement{node_idx, flipped_set_size}) = self.flip_set_size_heap.pop() {
            // Check if current
            if flipped_set_size == self.check_flip_set_size(node_idx, syndrome) {
                // Flip bit
                correction[self.tanner_graph[node_idx].as_bit_node().unwrap().idx] ^= true;
                // Flip (update) checks
                for neighbor_check_node_idx in self.tanner_graph.neighbors_undirected(node_idx) {
                    syndrome[self.tanner_graph[neighbor_check_node_idx].as_check_node().unwrap().idx] ^= true;
                }

                // Recompute flip set sizes
                self.update_flip_set_sizes(node_idx, syndrome);
            }
        }
    } 
}