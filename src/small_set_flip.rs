use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::{BinaryHeap,HashSet};
use enum_as_inner::EnumAsInner;

use crate::TannerGraphNode;

#[derive(Debug)]
struct CheckNode {
    idx : usize
}

#[derive(Debug)]
struct BitNode {
    idx : usize,
    flip_set_size : i32,
}

#[derive(Debug, EnumAsInner)]
enum SsfTannerGraphNode {
    CheckNode(CheckNode),
    BitNode(BitNode),
}

#[derive(Debug)]
pub struct SmallSetFlip {
    tanner_graph : UnGraph<SsfTannerGraphNode, ()>,
    flip_set_size_heap : BinaryHeap<FlipSizeHeapElement>,
    check_node_count : usize,
    bit_node_count : usize,
    // Heap to keep track of largest flip variable
    // Use same stale element heap tracking trick as UFD
}

#[derive(Clone, Debug)]
struct FlipSizeHeapElement {
    node_idx : NodeIndex,
    flipped_set_size : i32,
}

impl Ord for FlipSizeHeapElement {
    fn cmp(&self, other : &FlipSizeHeapElement) -> Ordering {
        self.flipped_set_size.cmp(other)
    }
}

impl SmallSetFlip {
    pub fn new(tanner_graph : &UnGraph<TannerGraphNode, _>,) -> SmallSetFlip {
        let check_node_count = tanner_graph.nodes().filter_map(|node_idx| tanner_graph.as_checkNode()).count();
        let bit_node_count = tanner_graph.nodes().filter_map(|node_idx| tanner_graph.as_bitNode()).count();


        let ssf_tanner_graph = tanner_graph.map(|node_index| {
            match tanner_graph[node_index] {
                TannerGraphNode::CheckNode(idx) => SsfTannerGraphNode{CheckNode{idx}},
                TannerGraphNode::DataNode(idx) => SsfTannerGraphNode{DataNode{idx, flipped_set_size:0}},
            }
        }, |edge_index| ());

        SmallSetFlip{bit_node_count, check_node_count, flip_set_size_heap:BinaryHeap::new(), tanner_graph:ssf_tanner_graph}
    }

    pub fn correct_syndrome(self : &mut Self, mut syndrome : Vec<bool>, mut correction : Vec<bool>) {
        assert!(syndrome.len() == check_node_count);
        
        // Initialize correction vector
        correction.resize(bit_node_count);
        correction.fill(false);

        // Initialize all flip set sizes
        check_flip_set_size = |bit_node_idx : NodeIndex, syndrome : &Vec<bool>, tanner_graph : &UnGraph<TannerGraphBit, _>| -> i32 {
            tanner_graph.neighbors(bit_node_idx).filter(|neighbor_idx| syndrome[tanner_graph[neighbor_idx].as_checkNode().unwrap().idx]).count()
        };

        flip_set_size_heap.clear();
        for node_idx in tanner_graph.nodes() {
            if let SsfTannerGraphNode::BitNode(BitNode {idx, ref mut flip_set_size}) = tanner_graph[node_idx] {
                *flip_set_size = check_flip_set_size(node_idx, &syndrome, tanner_graph);
                flip_set_size_heap.push(FlipSizeHeapElement {
                    node_idx, flip_set_size
                });
            }
        }

        // ====== Inner loop ======
        while let Some(FlipSizeHeapElement{node_idx, flipped_set_size}) = flip_set_size_heap.pop() {
            // Check if current
            if flipped_set_size == check_flip_set_size(node_idx, &syndrome, tanner_graph) {
                // Flip data
                correction[tanner_graph[node_idx].as_dataNode().unwrap().idx] ^= true;
                // Flip Checks
                for neighbor_check_node_idx in tanner_graph.neighbors(node_idx) {
                    syndrome[tanner_graph[neighbor_check_node_idx].as_checkNode().unwrap().idx] ^= true;
                }

                // Recompute flip set sizes
                // TODO: we can remove one layer of pointer chasing with an updatable priority queue
                for neighbor_check_node_idx in tanner_graph.neighbors(node_idx) {
                    for neighor_data_node_idx in tanner_graph.neighbors(neighbor_check_node_idx) {
                        let flip_set_size = check_flip_set_size(neighor_data_node_idx, &syndrome, tanner_graph);
                        tanner_graph[neighor_data_node_idx].as_dataNode().unwrap().flip_set_size = flip_set_size;

                        if flip_set_size > 0 {
                            flip_set_size_heap.push(FlipSizeHeapElement {
                                neighor_data_node_idx, flip_set_size
                            });
                        }
                    }
                }
            }
        }
    } 
}