use ::core::cmp::Ordering;
use ::core::hint::unreachable_unchecked;
use ::rustc_hash::FxHashMap;

pub type NodeGeneId = usize;
pub type ConnectionGeneId = usize;

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
}

#[derive(Debug, Clone)]
pub struct Organism {
    pub node_genes: Vec<NodeGene>,
    pub connection_genes: Vec<ConnectionGene>,
    pub fitness: CheckedF64,
    pub age: u8,
}

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: NodeGeneId,
    pub category: NodeGeneCategory,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub id: ConnectionGeneId,
    pub source_id: NodeGeneId,
    pub target_id: NodeGeneId,
    pub state: ConnectionGeneState,
    pub weight: f64,
    pub value: f64,
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

pub(crate) type OrganismIndex = usize;

#[derive(Debug)]
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
                / (1.0 + self.remaining.len() as f64),
        )
    }

    pub fn sort_by_genetic_distance(&mut self) {
        self.remaining
            .sort_by(|a, b| a.genetic_distance.cmp(&b.genetic_distance));
    }
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct CheckedF64(f64);

impl CheckedF64 {
    pub(crate) fn new(f: f64) -> Self {
        assert!(!f.is_nan());

        Self(f)
    }

    pub(crate) fn zero() -> Self {
        Self(0.0)
    }

    pub fn as_f64(self) -> f64 {
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

#[derive(Debug)]
pub struct SpeciatedOrganism {
    pub organism_index: OrganismIndex,
    pub genetic_distance: CheckedF64,
}
