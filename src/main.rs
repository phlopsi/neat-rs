use ::neat::*;

const MAX_DISTANCE_QUEUE_SIZE: usize = 1;
const MAX_EVALUATION_QUEUE_SIZE: usize = 2;

// const PROBABILITY_MUTATE_ONLY: f32 = 0.25;
// const PROBABILITY_MATE_ONLY: f32 = 0.2;
// const PROBABILITY_MATE_AND_MUTATE: f32 = 1.0 - PROBABILITY_MUTATE_ONLY - PROBABILITY_MATE_ONLY;

const PROBABILITY_ADD_NODE: f32 = 0.03;
const PROBABILITY_ADD_CONNECTION: f32 = 0.05;
const PROBABILITY_MUTATE_WEIGHTS: f32 = 1.0 - PROBABILITY_ADD_NODE - PROBABILITY_ADD_CONNECTION;

const WEIGHT_MUTATION_POWER: f64 = 2.0;

const COEFFICIENT_EXCESS_GENES: f64 = 1.0;
const COEFFICIENT_DISJOINT_GENES: f64 = 1.0;
const COEFFICIENT_WEIGHT_DIFFERENCE: f64 = 1.0;

fn main() {
    let mut thread_rng = rand::thread_rng();

    // Initialize population
    let instant = std::time::Instant::now();
    let mut population;

    {
        let population_size = 10;
        let number_of_inputs = 2;
        let number_of_outputs = 1;
        population = initialize(population_size, number_of_inputs, number_of_outputs);
    }

    eprintln!(
        "Initialize population: {} µs",
        instant.elapsed().as_micros()
    );

    // // Mutate population (add connection)
    // let instant = std::time::Instant::now();
    //
    // for organism in &mut population.organisms {
    //     mutate_add_connection(organism, &mut population.last_connection_gene_id);
    // }
    //
    // eprintln!(
    //     "Mutate population (add connection): {} µs",
    //     instant.elapsed().as_micros()
    // );
    //
    // // Mutate population (add node)
    // let instant = std::time::Instant::now();
    //
    // for organism in &mut population.organisms {
    //     mutate_add_node(
    //         organism,
    //         &mut population.last_node_gene_id,
    //         &mut population.last_connection_gene_id,
    //     );
    // }
    //
    // eprintln!(
    //     "Mutate population (add node): {} µs",
    //     instant.elapsed().as_micros()
    // );

    // Reproduce population
    let instant = std::time::Instant::now();
    reproduce(&mut population, &mut thread_rng);
    eprintln!("Reproduce population: {} µs", instant.elapsed().as_micros());

    // DEBUG
    // reproduce(&mut population, &mut thread_rng);
    // reproduce(&mut population, &mut thread_rng);
    // reproduce(&mut population, &mut thread_rng);
    // reproduce(&mut population, &mut thread_rng);
    // reproduce(&mut population, &mut thread_rng);
    // reproduce(&mut population, &mut thread_rng);

    // Evaluate population
    let mut evaluation = Vec::with_capacity(population.organisms.len());
    let instant = std::time::Instant::now();

    for organism in &mut population.organisms {
        evaluation.push(evaluate(
            organism,
            &population.inputs,
            &population.outputs,
            &[1.0, 0.0],
        ));
    }

    eprintln!("Evaluate population: {} µs", instant.elapsed().as_micros());
    eprintln!("Evaluation: {:?}", evaluation);
    // eprintln!("Population: {:#?}", population);
}

fn initialize(pop_size: usize, num_input: usize, num_output: usize) -> Population {
    const NUM_BIAS: usize = 1;

    let mut organism = Organism {
        node_genes: Vec::with_capacity(NUM_BIAS + num_input + num_output),
        connection_genes: Vec::new(),
    };

    let mut id = 1;

    organism.node_genes.push(NodeGene {
        id,
        category: NodeGeneCategory::Bias,
        value: 0.0,
    });

    let mut inputs = Vec::with_capacity(num_input);

    for _ in 0..num_input {
        id += 1;

        organism.node_genes.push(NodeGene {
            id,
            category: NodeGeneCategory::Input,
            value: 0.0,
        });

        inputs.push(id);
    }

    let mut outputs = Vec::with_capacity(num_output);

    for _ in 0..num_output {
        id += 1;

        organism.node_genes.push(NodeGene {
            id,
            category: NodeGeneCategory::Output,
            value: 0.0,
        });

        outputs.push(id);
    }

    let mut organisms = Vec::<Organism>::with_capacity(pop_size);

    for _ in 1..pop_size {
        organisms.push(organism.clone());
    }

    organisms.push(organism);

    Population {
        organisms,
        inputs,
        outputs,
        last_node_gene_id: id,
        last_connection_gene_id: 0,
    }
}

fn evaluate(
    organism: &mut Organism,
    inputs: &[NodeGeneId],
    outputs: &[NodeGeneId],
    input_values: &[f64],
) -> Vec<f64> {
    use arraydeque::ArrayDeque;
    use std::collections::HashSet;

    assert_eq!(inputs.len(), input_values.len());

    let mut evaluated_node_genes = HashSet::<NodeGeneId>::new();
    let mut evaluated_connection_genes = HashSet::<ConnectionGeneId>::new();
    let mut queue = ArrayDeque::<[NodeGeneId; MAX_EVALUATION_QUEUE_SIZE]>::new();

    // Evaluate network inputs
    {
        let zipped = inputs
            .iter()
            .zip(input_values)
            .map(|(node_gene_id, value)| (*node_gene_id, *value))
            .collect::<Vec<(NodeGeneId, f64)>>();

        for (node_gene_id, value) in zipped {
            node_gene_mut(&mut organism.node_genes, node_gene_id).value = value;

            // Extract into function? (Copy below)
            {
                evaluated_node_genes.insert(node_gene_id);

                for connection_gene in
                    outgoing_connection_genes_mut(&mut organism.connection_genes, node_gene_id)
                        .filter(|connection_gene| connection_gene.state.is_enabled())
                {
                    connection_gene.value = value * connection_gene.weight;
                    evaluated_connection_genes.insert(connection_gene.id);
                    queue
                        .push_back(connection_gene.target_id)
                        .expect("maximum evaluation queue size reached");
                }
            }
        }
    }

    // Evaluate the rest of the network
    {
        'queue: while let Option::Some(node_gene_id) = queue.pop_front() {
            if evaluated_node_genes.contains(&node_gene_id) {
                continue;
            }

            let mut value = 0.0;

            for connection_gene in
                incoming_connection_genes(&organism.connection_genes, node_gene_id)
                    .filter(|connection_gene| connection_gene.state.is_enabled())
            {
                if !evaluated_connection_genes.contains(&connection_gene.id) {
                    continue 'queue;
                }

                value += connection_gene.value;
            }

            value = sigmoid(value);
            node_gene_mut(&mut organism.node_genes, node_gene_id).value = value;

            // Extract into function? (Copied from above)
            {
                evaluated_node_genes.insert(node_gene_id);

                for connection_gene in
                    outgoing_connection_genes_mut(&mut organism.connection_genes, node_gene_id)
                        .filter(|connection_gene| connection_gene.state.is_enabled())
                {
                    connection_gene.value = value * connection_gene.weight;
                    evaluated_connection_genes.insert(connection_gene.id);
                    queue.push_back(connection_gene.target_id).unwrap();
                }
            }
        }
    }

    // Retrieve outputs
    {
        let mut output_values = Vec::<f64>::with_capacity(outputs.len());

        for connection_gene_id in outputs {
            output_values.push(node_gene(&organism.node_genes, *connection_gene_id).value);
        }

        output_values
    }
}

#[inline(always)]
fn node_gene<'a>(node_genes: &'a Vec<NodeGene>, node_gene_id: NodeGeneId) -> &'a NodeGene {
    node_genes
        .iter()
        .find(|node_gene| node_gene_id == node_gene.id)
        .unwrap()
}

#[inline(always)]
fn node_gene_mut<'a>(
    node_genes: &'a mut Vec<NodeGene>,
    node_gene_id: NodeGeneId,
) -> &'a mut NodeGene {
    node_genes
        .iter_mut()
        .find(|node_gene| node_gene_id == node_gene.id)
        .unwrap()
}

#[inline(always)]
fn outgoing_connection_genes_mut<'a>(
    connection_genes: &'a mut Vec<ConnectionGene>,
    node_gene_id: NodeGeneId,
) -> impl Iterator<Item = &'a mut ConnectionGene> {
    connection_genes
        .iter_mut()
        .filter(move |connection_gene| node_gene_id == connection_gene.source_id)
}

#[inline(always)]
fn incoming_connection_genes<'a>(
    connection_genes: &'a Vec<ConnectionGene>,
    node_gene_id: NodeGeneId,
) -> impl Iterator<Item = &'a ConnectionGene> {
    connection_genes
        .iter()
        .filter(move |connection_gene| node_gene_id == connection_gene.target_id)
}

fn mutate_add_connection(
    organism: &mut Organism,
    last_connection_gene_id: &mut ConnectionGeneId,
    thread_rng: &mut rand::rngs::ThreadRng,
) {
    use rand::seq::SliceRandom;

    loop {
        let choices = organism
            .node_genes
            .choose_multiple(thread_rng, 2)
            .collect::<Vec<&NodeGene>>();

        let source_id;
        let target_id;

        match choices[0].category {
            NodeGeneCategory::Bias | NodeGeneCategory::Input => match choices[1].category {
                NodeGeneCategory::Bias | NodeGeneCategory::Input => {
                    continue;
                }
                NodeGeneCategory::Hidden | NodeGeneCategory::Output => {
                    source_id = choices[0].id;
                    target_id = choices[1].id;
                }
            },
            NodeGeneCategory::Hidden => match choices[1].category {
                NodeGeneCategory::Bias | NodeGeneCategory::Input => {
                    source_id = choices[1].id;
                    target_id = choices[0].id;
                }
                NodeGeneCategory::Hidden | NodeGeneCategory::Output => {
                    source_id = choices[0].id;
                    target_id = choices[1].id;
                }
            },
            NodeGeneCategory::Output => match choices[1].category {
                NodeGeneCategory::Bias | NodeGeneCategory::Input | NodeGeneCategory::Hidden => {
                    source_id = choices[1].id;
                    target_id = choices[0].id;
                }
                NodeGeneCategory::Output => {
                    continue;
                }
            },
        }

        if organism
            .connection_genes
            .iter()
            .find(|connection_gene| {
                source_id == connection_gene.source_id && target_id == connection_gene.target_id
            })
            .is_some()
        {
            continue;
        }

        if distance(&organism.connection_genes, target_id, source_id).is_finite() {
            continue;
        }

        *last_connection_gene_id += 1;

        organism.connection_genes.push(ConnectionGene {
            id: *last_connection_gene_id,
            source_id,
            target_id,
            state: ConnectionGeneState::Enabled,
            weight: 1.0,
            value: 0.0,
        });

        break;
    }
}

fn mutate_add_node(
    organism: &mut Organism,
    last_node_gene_id: &mut NodeGeneId,
    last_connection_gene_id: &mut ConnectionGeneId,
    thread_rng: &mut rand::rngs::ThreadRng,
) {
    use rand::seq::IteratorRandom;

    let connection_genes = &mut organism.connection_genes;
    let node_genes = &mut organism.node_genes;

    let connection_gene = connection_genes
        .iter_mut()
        .filter(|connection_gene| {
            connection_gene.state.is_enabled()
                && node_gene(node_genes, connection_gene.target_id)
                    .category
                    .is_output()
        })
        .choose(thread_rng);

    if let Some(connection_gene) = connection_gene {
        connection_gene.state = ConnectionGeneState::Disabled;
        *last_node_gene_id += 1;

        node_genes.push(NodeGene {
            id: *last_node_gene_id,
            category: NodeGeneCategory::Hidden,
            value: 0.0,
        });

        let source_id = connection_gene.source_id;
        let target_id = connection_gene.target_id;
        let weight = connection_gene.weight;

        *last_connection_gene_id += 1;

        connection_genes.push(ConnectionGene {
            id: *last_connection_gene_id,
            source_id,
            target_id: *last_node_gene_id,
            state: ConnectionGeneState::Enabled,
            weight: 1.0,
            value: 0.0,
        });

        *last_connection_gene_id += 1;

        connection_genes.push(ConnectionGene {
            id: *last_connection_gene_id,
            source_id: *last_node_gene_id,
            target_id,
            state: ConnectionGeneState::Enabled,
            weight,
            value: 0.0,
        });
    }
}

fn distance(connection_genes: &Vec<ConnectionGene>, from: NodeGeneId, to: NodeGeneId) -> Distance {
    use arraydeque::ArrayDeque;
    use rustc_hash::FxHashSet;

    if from == to {
        return Distance::Finite(0);
    }

    let mut visited = FxHashSet::<NodeGeneId>::default();

    let mut queue = connection_genes
        .iter()
        .filter(|connection_gene| from == connection_gene.source_id)
        .map(|connection_gene| (connection_gene.target_id, 1))
        .collect::<ArrayDeque<[(NodeGeneId, usize); MAX_DISTANCE_QUEUE_SIZE]>>();

    while let Option::Some((target_id, distance)) = queue.pop_front() {
        if visited.insert(target_id) {
            if target_id == to {
                return Distance::Finite(distance);
            }

            queue.extend(
                connection_genes
                    .iter()
                    .filter(|connection_gene| from == connection_gene.source_id)
                    .map(|connection_gene| (connection_gene.target_id, distance + 1)),
            );
        }
    }

    Distance::Infinite
}

fn mutate_change_connection_weight(
    organism: &mut Organism,
    thread_rng: &mut rand::rngs::ThreadRng,
) {
    use rand::distributions::Distribution;
    use rand::distributions::Uniform;

    let weight_change_distribution = Uniform::from(-WEIGHT_MUTATION_POWER..=WEIGHT_MUTATION_POWER);

    for connection_gene in &mut organism.connection_genes {
        connection_gene.weight += weight_change_distribution.sample(thread_rng);
    }
}

fn reproduce(population: &mut neat::Population, thread_rng: &mut rand::rngs::ThreadRng) {
    use rand::distributions::weighted::alias_method::WeightedIndex;
    use rand::distributions::Distribution;

    // let mutate_mate_distribution = WeightedIndex::new(vec![
    //     PROBABILITY_MUTATE_ONLY,
    //     PROBABILITY_MATE_ONLY,
    //     PROBABILITY_MATE_AND_MUTATE,
    // ])
    // .unwrap();

    let mutate_distribution = WeightedIndex::new(vec![
        PROBABILITY_ADD_NODE,
        PROBABILITY_ADD_CONNECTION,
        PROBABILITY_MUTATE_WEIGHTS,
    ])
    .unwrap();

    let organisms = &mut population.organisms;
    let length = organisms.len();
    organisms.reserve(length);

    for index in 0..length {
        let mut cloned = unsafe { organisms.get_unchecked(index) }.clone();
        let sample = mutate_distribution.sample(thread_rng);

        match sample {
            0 => mutate_add_node(
                &mut cloned,
                &mut population.last_node_gene_id,
                &mut population.last_connection_gene_id,
                thread_rng,
            ),
            1 => mutate_add_connection(
                &mut cloned,
                &mut population.last_connection_gene_id,
                thread_rng,
            ),
            2 => mutate_change_connection_weight(&mut cloned, thread_rng),
            _ => unreachable!(),
        }

        organisms.push(cloned);
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1f64 / (1f64 + (-x).exp())
}

fn genetic_distance<'a, 'b>(genes1: &[ConnectionGene], genes2: &[ConnectionGene]) -> f64 {
    use ::core::cmp::Ordering;
    use ::core::hint::unreachable_unchecked;

    let mut number_of_matching_genes = 0;
    let mut number_of_disjoint_genes = 0;
    let mut total_weight_difference = 0.0;
    let number_of_excess_genes;

    let genes1_len = genes1.len();
    let genes2_len = genes2.len();

    if genes1_len == 0 {
        number_of_excess_genes = genes2_len;
    } else if genes2_len == 0 {
        number_of_excess_genes = genes1_len;
    } else {
        let mut genes1_iter = genes1.iter();
        let mut genes2_iter = genes2.iter();

        let mut gene1 = genes1_iter
            .next()
            .unwrap_or_else(|| unsafe { unreachable_unchecked() });

        let mut gene2 = genes2_iter
            .next()
            .unwrap_or_else(|| unsafe { unreachable_unchecked() });

        loop {
            match gene1.id.cmp(&gene2.id) {
                Ordering::Less => {
                    number_of_disjoint_genes += 1;

                    gene1 = match genes1_iter.next() {
                        Some(g) => g,
                        None => {
                            number_of_excess_genes = genes2_iter.size_hint().0;
                            break;
                        }
                    };
                }
                Ordering::Equal => {
                    number_of_matching_genes += 1;
                    total_weight_difference += (gene1.weight - gene2.weight).abs();

                    gene1 = match genes1_iter.next() {
                        Some(g) => g,
                        None => {
                            number_of_excess_genes = genes2_iter.size_hint().0;
                            break;
                        }
                    };

                    gene2 = match genes2_iter.next() {
                        Some(g) => g,
                        None => {
                            number_of_excess_genes = 1 + genes1_iter.size_hint().0;
                            break;
                        }
                    };
                }
                Ordering::Greater => {
                    number_of_disjoint_genes += 1;

                    gene2 = match genes2_iter.next() {
                        Some(g) => g,
                        None => {
                            number_of_excess_genes = genes1_iter.size_hint().0;
                            break;
                        }
                    };
                }
            }
        }
    }

    let average_weight_difference;

    if number_of_matching_genes > 0 {
        average_weight_difference = total_weight_difference / f64::from(number_of_matching_genes);
    } else {
        average_weight_difference = 0.0;
    }

    return (number_of_excess_genes as f64 * COEFFICIENT_EXCESS_GENES
        + number_of_disjoint_genes as f64 * COEFFICIENT_DISJOINT_GENES)
        / ::core::cmp::max(genes1.len(), genes2.len()) as f64
        + average_weight_difference * COEFFICIENT_WEIGHT_DIFFERENCE;
}
