pub type NodeGeneId = usize;
pub type ConnectionGeneId = usize;

#[derive(Debug)]
pub struct Population {
    pub organisms: Vec<Organism>,
    pub inputs: Vec<NodeGeneId>,
    pub outputs: Vec<NodeGeneId>,
    pub last_node_gene_id: NodeGeneId,
    pub last_connection_gene_id: ConnectionGeneId,
}

#[derive(Debug, Clone)]
pub struct Organism {
    pub node_genes: Vec<NodeGene>,
    pub connection_genes: Vec<ConnectionGene>,
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
