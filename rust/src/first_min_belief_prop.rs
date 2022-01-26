use petgraph::{graph::{NodeIndex, DiGraph}, EdgeDirection::*, visit::{EdgeRef, IntoEdgesDirected, IntoNeighbors}};
use enum_as_inner::EnumAsInner;

use crate::error_correcting_code::{TannerGraph, tanner_graph_edge_orientation, TannerGraphNode, Decoder, ErrorCorrectingCode};

/// First-min Belief Propagation from
/// Grospellier et al., Quantum 5, 432 (2021).
#[derive(Debug, Clone)]
pub struct FirstMinBeliefProp {
    tanner_graph : DiGraph<TannerGraphNode, usize>,
    check_node_count : usize,
    bit_node_count : usize,
    bit_to_check_message : Vec<f64>,
    check_to_bit_message : Vec<f64>,
    prior_log_likelihood : f64,
    correction_diff : Vec<bool>,
    syndrome_diff : Vec<bool>,
}

impl FirstMinBeliefProp {
    pub fn new(ErrorCorrectingCode {logicals:_, tanner_graph} : &ErrorCorrectingCode, error_prior : f64) -> FirstMinBeliefProp {
        assert!(tanner_graph_edge_orientation(&tanner_graph));

        let check_node_count = tanner_graph.node_indices().filter_map(|node_idx| tanner_graph[node_idx].as_check_node()).count();
        let bit_node_count = tanner_graph.node_indices().filter_map(|node_idx| tanner_graph[node_idx].as_bit_node()).count();

        let bit_to_check_message = vec![0.0; tanner_graph.edge_count()];
        let check_to_bit_message = vec![0.0; tanner_graph.edge_count()];
        let correction_diff = vec![false; bit_node_count];
        let syndrome_diff = vec![false; check_node_count];

        // Attach an index into the message arrays to each edge
        let mut edge_idx = 0usize;
        let edge_labeled_tanner_graph = tanner_graph.map(|_, x| x.clone(), |_, _| {
            let v = edge_idx;
            edge_idx += 1;
            v
        });

        FirstMinBeliefProp {
            tanner_graph:edge_labeled_tanner_graph, 
            check_node_count, 
            bit_node_count, 
            bit_to_check_message, 
            check_to_bit_message, 
            prior_log_likelihood: ((1.0-error_prior)/error_prior).ln(),
            correction_diff,
            syndrome_diff,
        }
    }

    /// Bits to checks belief propagation step
    fn update_bit_to_checks_messages(self : &mut Self) {
        for update_edge_idx in self.tanner_graph.edge_indices() {
            let (check_node_idx, bit_node_idx) = self.tanner_graph.edge_endpoints(update_edge_idx).unwrap();
            
            // Sum the check to bit messages on the boundary of target bit node that does not contain the source check node
            let neighbor_messages_sum : f64 = self.tanner_graph.edges_directed(bit_node_idx, Incoming).map(
                |edge_ref| if edge_ref.source() != check_node_idx { self.check_to_bit_message[*edge_ref.weight()] } else { 0.0 }).sum();

            // Update the bit to check message
            let message_idx = self.tanner_graph[update_edge_idx];
            self.bit_to_check_message[message_idx] = self.prior_log_likelihood + neighbor_messages_sum;
        }
    }

    /// Checks to bits belief propagation step
    fn update_check_to_bits_messages(self : &mut Self, syndrome : &Vec<bool>) {
        for update_edge_idx in self.tanner_graph.edge_indices() {
            let (check_node_idx, bit_node_idx) = self.tanner_graph.edge_endpoints(update_edge_idx).unwrap();
            
            // Sum the check to bit messages on the boundary of target bit node that does not contain the source check node
            let neighbor_messages_prod : f64 = self.tanner_graph.edges_directed(check_node_idx, Outgoing).map(
                |edge_ref| if edge_ref.target() != bit_node_idx { (self.bit_to_check_message[*edge_ref.weight()]/2.0).tanh() } else { 1.0 }).product();

            // Update the bit to check message
            let message_idx = self.tanner_graph[update_edge_idx];
            //  Syndrome is F2
            //  1 (non-trivial) |-> -1
            //  The syndrome entry is true if it is non-trivial
            let sign = if syndrome[*self.tanner_graph[check_node_idx].as_check_node().unwrap()] { -1.0 } else { 1.0 };
            self.check_to_bit_message[message_idx] = sign*2.0*neighbor_messages_prod.atanh();
        }
    }

    /// Propagate the beliefs
    fn sum_product_step(self : &mut Self, syndrome : &Vec<bool>) {
        self.update_check_to_bits_messages(syndrome);
        self.update_bit_to_checks_messages();
    }

    /// Compute the change in syndrome and correction vectors from the last round
    fn update_diffs(self : &mut Self, correction : &Vec<bool>) {
        // Init diffs
        self.syndrome_diff.fill(false);
        self.correction_diff.fill(false);

        // Compute correction diff
        for node_idx in self.tanner_graph.node_indices() {
            if let TannerGraphNode::BitNode(bit_idx) = self.tanner_graph[node_idx] {
                // Prior + sum of incoming messages
                let bit_log_likelihood = self.prior_log_likelihood + self.tanner_graph.edges_directed(node_idx, Incoming).map(
                        |edge_ref| self.check_to_bit_message[*edge_ref.weight()]).sum::<f64>();
                // Positive is trivial
                let new_bit_correction = bit_log_likelihood < 0.0;
                self.correction_diff[bit_idx] = new_bit_correction ^ correction[bit_idx];
            }
        }

        // Compute syndrome diff
        for node_idx in self.tanner_graph.node_indices() {
            if let TannerGraphNode::CheckNode(check_idx) = self.tanner_graph[node_idx] {
                let non_triv_diff = self.tanner_graph.neighbors(node_idx).map(|node_idx| {
                        let bit_idx = self.tanner_graph[node_idx].as_bit_node().unwrap();
                        self.correction_diff[*bit_idx]
                    }).reduce(|a, b| a ^ b).unwrap();

                self.syndrome_diff[check_idx] = non_triv_diff;
            }
        }
    }

    /// Apply the syndrome and correction differences to a given syndrome and correction vector
    fn apply_diffs(self : &Self, syndrome : &mut Vec<bool> , correction : &mut Vec<bool>) {
        for i in 0..correction.len() {
            if self.correction_diff[i] {
                correction[i] ^= true;
            }
        }
        for i in 0..syndrome.len() {
            if self.syndrome_diff[i] {
                syndrome[i] ^= true;
            }
        }
    }
}

impl Decoder for FirstMinBeliefProp {
    fn correct_syndrome(self : &mut Self, syndrome : &mut Vec<bool>, correction : &mut Vec<bool>) {
        assert!(syndrome.len() == self.check_node_count);
        
        // Initialize correction vector
        correction.resize(self.bit_node_count, false);
        correction.fill(false);

        // Initialize variables
        self.check_to_bit_message.fill(0.0);
        self.bit_to_check_message.fill(self.prior_log_likelihood);
        
        loop {
            // Update messages
            self.sum_product_step(syndrome);

            // Compute new diffs
            self.update_diffs(correction);
            
            // Difference of syndrome weights
            // New syndrome weight - old syndrome weight
            let weight_difference : i32 = syndrome.iter().zip(self.syndrome_diff.iter()).map(
                |(s, s_diff)| if *s_diff { if *s { -1 } else { 1 }} else { 0 }).sum();

            // If syndrome weight decreases then apply the diff and continue the loop
            if weight_difference < 0 {
                self.apply_diffs(syndrome, correction);
                continue;
            }

            break;
        }

    }
}