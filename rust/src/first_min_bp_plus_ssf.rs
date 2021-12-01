use crate::error_correcting_code::{TannerGraph, Decoder};
use crate::first_min_belief_prop::FirstMinBeliefProp;
use crate::small_set_flip::SmallSetFlip;

/// First-min Belief Propagation + Small Set Flipfrom
/// Grospellier et al., Quantum 5, 432 (2021).
#[derive(Debug, Clone)]
struct FirstMinBPplusSSF {
    belief_prop : FirstMinBeliefProp,
    small_set_flip : SmallSetFlip,
}

impl FirstMinBPplusSSF {
    pub fn new(tanner_graph : &TannerGraph, error_prior : f64) -> FirstMinBPplusSSF {
        FirstMinBPplusSSF {
            belief_prop:FirstMinBeliefProp::new(tanner_graph, error_prior),
            small_set_flip:SmallSetFlip::new(tanner_graph),
        }
    }
}

impl Decoder for FirstMinBPplusSSF {
    fn correct_syndrome(self : &mut Self, syndrome : &mut Vec<bool>, correction : &mut Vec<bool>) {
        // Apply BP then SSF
        // The syndrome value is required to be consistent with the input syndrome and output correction so we can compose
        self.belief_prop.correct_syndrome(syndrome, correction);
        self.small_set_flip.correct_syndrome(syndrome, correction);
    }
}