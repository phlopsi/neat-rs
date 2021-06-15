use ::core::cmp::Ordering;
use ::core::hint::unreachable_unchecked;
use ::rustc_hash::FxHashMap;

pub use rustc_hash::FxHashSet;
pub use std::collections::VecDeque;
pub use std::hash::BuildHasherDefault;

pub type NodeGeneId = usize;
pub type ConnectionGeneId = usize;
pub type OrganismIndex = usize;
pub type Fitness = CheckedF64;
pub type Value = f32;

#[derive(Debug)]
pub struct Population {
    pub organisms: Vec<Organism>,
    pub inputs: Vec<NodeGeneId>,
    pub outputs: Vec<NodeGeneId>,
    pub last_node_gene_id: NodeGeneId,
    pub last_connection_gene_id: ConnectionGeneId,
    pub initial_organism: Organism,
    pub ideal_population_size: usize,
    pub mutate_add_node_history: FxHashMap<ConnectionGeneId, NodeGeneId>,
    pub mutate_add_connection_history: FxHashMap<(NodeGeneId, NodeGeneId), ConnectionGeneId>,
    pub solution: Option<Organism>,
}

#[derive(Debug, Clone)]
pub struct Organism {
    pub node_genes: Vec<NodeGene>,
    pub connection_genes: Vec<ConnectionGene>,
    pub fitness: CheckedF64,
    pub age: u8,
}

impl Organism {
    pub fn number_of_hidden_node_genes(&self) -> usize {
        (&self.node_genes)
            .into_iter()
            .filter(|node_gene| node_gene.category.is_hidden())
            .count()
    }

    pub fn number_of_enabled_connection_genes(&self) -> usize {
        (&self.connection_genes)
            .into_iter()
            .filter(|connection_gene| connection_gene.state.is_enabled())
            .count()
    }
}

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: NodeGeneId,
    pub category: NodeGeneCategory,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub id: ConnectionGeneId,
    pub source_id: NodeGeneId,
    pub target_id: NodeGeneId,
    pub state: ConnectionGeneState,
    pub weight: Value,
    pub value: Value,
}

#[derive(Debug, Clone, Copy)]
pub enum NodeGeneCategory {
    Bias,
    Input,
    Output,
    Hidden,
}

impl NodeGeneCategory {
    /// Returns `true` if the node gene category is [`Bias`].
    ///
    /// [`Bias`]: #variant.Bias
    #[must_use]
    #[inline]
    pub fn is_bias(&self) -> bool {
        match *self {
            NodeGeneCategory::Bias => true,
            _ => false,
        }
    }

    /// Returns `true` if the node gene category is [`Input`].
    ///
    /// [`Input`]: #variant.Input
    #[must_use]
    #[inline]
    pub fn is_input(&self) -> bool {
        match *self {
            NodeGeneCategory::Input => true,
            _ => false,
        }
    }

    /// Returns `true` if the node gene category is [`Output`].
    ///
    /// [`Output`]: #variant.Output
    #[must_use]
    #[inline]
    pub fn is_output(&self) -> bool {
        match *self {
            NodeGeneCategory::Output => true,
            _ => false,
        }
    }

    /// Returns `true` if the node gene category is [`Hidden`].
    ///
    /// [`Hidden`]: #variant.Hidden
    #[must_use]
    #[inline]
    pub fn is_hidden(&self) -> bool {
        match *self {
            NodeGeneCategory::Hidden => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ConnectionGeneState {
    Disabled,
    Enabled,
}

impl ConnectionGeneState {
    /// Returns `true` if the node gene category is [`Enabled`].
    ///
    /// [`Enabled`]: #variant.Enabled
    #[must_use]
    #[inline]
    pub fn is_enabled(&self) -> bool {
        match *self {
            ConnectionGeneState::Enabled => true,
            ConnectionGeneState::Disabled => false,
        }
    }

    /// Returns `true` if the node gene category is [`Disabled`].
    ///
    /// [`Disabled`]: #variant.Disabled
    #[must_use]
    #[inline]
    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Distance {
    Infinite,
    Finite(usize),
}

impl Distance {
    /// Returns `true` if the distance is a [`Finite`] value.
    ///
    /// [`Finite`]: #variant.Finite
    #[must_use]
    #[inline]
    pub fn is_finite(&self) -> bool {
        match *self {
            Distance::Finite(_) => true,
            Distance::Infinite => false,
        }
    }

    /// Returns `true` if the distance is an [`Infinite`] value.
    ///
    /// [`Infinite`]: #variant.Infinite
    #[must_use]
    #[inline]
    pub fn is_infinite(&self) -> bool {
        !self.is_finite()
    }
}

pub struct EvaluationContext<'a, 'b, 'c> {
    pub(crate) organism: &'a mut Organism,
    pub(crate) inputs: &'b [NodeGeneId],
    pub(crate) outputs: &'c [NodeGeneId],
}

#[derive(Debug, Clone)]
pub struct Species {
    pub representative: OrganismIndex,
    pub remaining: Vec<SpeciatedOrganism>,
}

impl Species {
    pub fn fitness(&self, organisms: &[Organism]) -> CheckedF64 {
        CheckedF64::new(
            self.remaining
                .iter()
                .fold(
                    organisms[self.representative].fitness,
                    |accumulator, speciated_organism| {
                        accumulator + organisms[speciated_organism.organism_index].fitness
                    },
                )
                .0
                / (1.0 + self.remaining.len() as Value),
        )
    }

    pub fn sort_by_genetic_distance(&mut self) {
        self.remaining
            .sort_by(|a, b| a.genetic_distance.cmp(&b.genetic_distance));
    }
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct CheckedF64(Value);

impl CheckedF64 {
    pub(crate) fn new(f: Value) -> Self {
        assert!(!f.is_nan());

        Self(f)
    }

    pub(crate) fn zero() -> Self {
        Self(0.0)
    }

    pub fn as_float(self) -> Value {
        self.0
    }
}

impl Eq for CheckedF64 {}

impl Ord for CheckedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or_else(|| unsafe {
            unreachable_unchecked();
        })
    }
}

impl ::core::ops::Add for CheckedF64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        CheckedF64(self.0 + rhs.0)
    }
}

impl ::core::iter::Sum for CheckedF64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(CheckedF64::zero(), ::core::ops::Add::add)
    }
}

#[derive(Debug, Clone)]
pub struct SpeciatedOrganism {
    pub organism_index: OrganismIndex,
    pub genetic_distance: CheckedF64,
}

#[allow(dead_code)]
struct Network {
    vertices: Vec<VertexVector>,
    edges: Vec<EdgeVector>,
}

impl Network {}

pub unsafe fn feed_forward(vertices: Box<[Box<[VertexVector]>]>, edges: Box<[Box<[EdgeVector]>]>) {
    use ::core::arch::x86_64::_mm256_add_ps;
    use ::core::arch::x86_64::_mm256_mul_ps;

    for (layers, weight_matrix) in vertices.windows(2).zip((&edges).into_iter()) {
        // let layer = layers.get_unchecked(0);
        // let next_layer = layers.get_unchecked(1);
        let layer = &layers[0];
        let next_layer = &layers[1];
        let next_layer_len = next_layer.len();
        let mut start = 0;

        for source_vertex_vec in layer.into_iter() {
            for (edge_vec, target_vertex_vec) in (&weight_matrix[start..start + next_layer_len])
                .into_iter()
                .zip(next_layer.into_iter())
            {
                target_vertex_vec.value.set(_mm256_add_ps(
                    target_vertex_vec.value.get(),
                    _mm256_mul_ps(source_vertex_vec.value.get(), edge_vec.weight),
                ));
            }

            start += next_layer_len;
        }
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct EdgeVector {
    pub weight: ::core::arch::x86_64::__m256,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct VertexVector {
    pub value: ::core::cell::Cell<::core::arch::x86_64::__m256>,
}
