use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use enum_as_inner::EnumAsInner;

use crate::error_correcting_code::TannerGraphNode;

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
    tanner_graph : UnGraph<SsfTannerGraphNode, ()>,
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
    pub fn new(tanner_graph : &UnGraph<TannerGraphNode, ()>,) -> SmallSetFlip {
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

    pub fn correct_syndrome(self : &mut Self, mut syndrome : Vec<bool>, mut correction : Vec<bool>) {
        assert!(syndrome.len() == self.check_node_count);
        
        // Initialize correction vector
        correction.resize(self.bit_node_count, false);
        correction.fill(false);

        // Initialize all flip set sizes
        let check_flip_set_size = |bit_node_idx : NodeIndex, syndrome : &Vec<bool>, tanner_graph : &UnGraph<SsfTannerGraphNode, ()>| -> i32 {
            tanner_graph.neighbors(bit_node_idx).map(|neighbor_idx| if syndrome[tanner_graph[neighbor_idx].as_check_node().unwrap().idx] { -1 } else { 1 }).sum()
        };

        self.flip_set_size_heap.clear();
        for node_idx in self.tanner_graph.node_indices() {
            if let SsfTannerGraphNode::BitNode(BitNode {idx, flip_set_size:_}) = self.tanner_graph[node_idx] {
                let new_flip_set_size = check_flip_set_size(node_idx, &syndrome, &self.tanner_graph);

                self.tanner_graph[node_idx] = SsfTannerGraphNode::BitNode(BitNode {idx, flip_set_size:new_flip_set_size});

                self.flip_set_size_heap.push(FlipSizeHeapElement {
                    node_idx, flipped_set_size:new_flip_set_size
                });
            }
        }

        // ====== Inner loop ======
        while let Some(FlipSizeHeapElement{node_idx, flipped_set_size}) = self.flip_set_size_heap.pop() {
            // Check if current
            if flipped_set_size == check_flip_set_size(node_idx, &syndrome, &self.tanner_graph) {
                // Flip data
                correction[self.tanner_graph[node_idx].as_bit_node().unwrap().idx] ^= true;
                // Flip Checks
                for neighbor_check_node_idx in self.tanner_graph.neighbors(node_idx) {
                    syndrome[self.tanner_graph[neighbor_check_node_idx].as_check_node().unwrap().idx] ^= true;
                }

                // Recompute flip set sizes
                // TODO: we can remove one layer of pointer chasing with an updatable priority queue
                let mut neighbor_walker = self.tanner_graph.neighbors(node_idx).detach();
                while let Some((_, neighbor_check_node_idx)) = neighbor_walker.next(&self.tanner_graph)  {

                    let mut neighbor_of_neighbor_walker = self.tanner_graph.neighbors(neighbor_check_node_idx).detach();
                    while let Some((_, neighor_data_node_idx)) = neighbor_of_neighbor_walker.next(&self.tanner_graph)  {
                    
                        let flip_set_size = check_flip_set_size(neighor_data_node_idx, &syndrome, &self.tanner_graph);
                        self.tanner_graph[neighor_data_node_idx].as_bit_node_mut().unwrap().flip_set_size = flip_set_size;

                        if flip_set_size > 0 {
                            self.flip_set_size_heap.push(FlipSizeHeapElement {
                                node_idx:neighor_data_node_idx, flipped_set_size:flip_set_size
                            });
                        }
                    }
                }
            }
        }
    } 
}