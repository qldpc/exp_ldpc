use petgraph::graph::{NodeIndex, UnGraph};
use enum_as_inner::EnumAsInner;

#[derive(Debug, EnumAsInner)]
enum TannerGraphNode {
    CheckNode(usize),
    BitNode(usize),
}
