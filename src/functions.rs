use crate::types::*;
use ::core::cmp::Ordering;
use ::core::hint::unreachable_unchecked;
use ::core::ops::DerefMut;
use ::rustc_hash::FxHashMap;

const MAX_DISTANCE_QUEUE_SIZE: usize = 1;
const MAX_EVALUATION_QUEUE_SIZE: usize = 100;

// const PROBABILITY_MUTATE_ONLY: f32 = 0.25;
// const PROBABILITY_MATE_ONLY: f32 = 0.2;
// const PROBABILITY_MATE_AND_MUTATE: f32 = 1.0 - PROBABILITY_MUTATE_ONLY - PROBABILITY_MATE_ONLY;

const PROBABILITY_ADD_NODE: f32 = 0.001;
const PROBABILITY_ADD_CONNECTION: f32 = 0.03;
const PROBABILITY_MUTATE_WEIGHTS: f32 = 1.0 - PROBABILITY_ADD_NODE - PROBABILITY_ADD_CONNECTION;

const WEIGHT_MUTATION_POWER: f64 = 2.0;

const COEFFICIENT_EXCESS_GENES: f64 = 1.0;
const COEFFICIENT_DISJOINT_GENES: f64 = 1.0;
const COEFFICIENT_WEIGHT_DIFFERENCE: f64 = 1.0;

#[inline]
pub fn initialize(
    pop_size: usize,
    num_input: usize,
    num_output: usize,
    mut fitness: impl FnMut(&mut EvaluationContext) -> f64,
) -> Population {
    const NUM_BIAS: usize = 1;

    let mut organism = Organism {
        node_genes: Vec::with_capacity(NUM_BIAS + num_input + num_output),
        connection_genes: Vec::new(),
        fitness: CheckedF64::zero(),
        age: 0,
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

    organism.fitness = CheckedF64::new(fitness(&mut EvaluationContext {
        organism: &mut organism,
        inputs: &inputs,
        outputs: &outputs,
    }));

    let mut organisms = Vec::<Organism>::with_capacity(pop_size);

    for _ in 1..pop_size {
        organisms.push(organism.clone());
    }

    organisms.push(organism.clone());

    Population {
        organisms,
        inputs,
        outputs,
        last_node_gene_id: id,
        last_connection_gene_id: 0,
        initial_organism: organism,
        ideal_population_size: pop_size,
        mutate_add_node_history: Default::default(),
        mutate_add_connection_history: Default::default(),
    }
}

#[inline]
pub fn evaluate<'a, 'b, 'c>(
    mut context: impl DerefMut<Target = EvaluationContext<'a, 'b, 'c>>,
    input_values: &[f64],
) -> Vec<f64> {
    use arraydeque::ArrayDeque;
    use rustc_hash::FxHashSet;
    use std::hash::BuildHasherDefault;

    const BIAS_ID: usize = 1;

    let EvaluationContext {
        organism,
        inputs,
        outputs,
    } = &mut *context;

    assert_eq!(inputs.len(), input_values.len());

    let mut evaluated_node_genes = FxHashSet::<NodeGeneId>::with_capacity_and_hasher(
        organism.node_genes.len() - outputs.len(),
        BuildHasherDefault::default(),
    );

    let mut evaluated_connection_genes = FxHashSet::<ConnectionGeneId>::with_capacity_and_hasher(
        organism.connection_genes.len(),
        BuildHasherDefault::default(),
    );

    let mut queue = ArrayDeque::<[NodeGeneId; MAX_EVALUATION_QUEUE_SIZE]>::new();

    // Initialize bias node gene value
    node_gene_mut(&mut organism.node_genes, BIAS_ID).value = 1.0;
    evaluated_node_genes.insert(BIAS_ID);

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
                    queue
                        .push_back(connection_gene.target_id)
                        .expect("maximum evaluation queue size reached");
                }
            }
        }
    }

    // Retrieve outputs
    let output_values = {
        let mut output_values = Vec::<f64>::with_capacity(outputs.len());

        for node_gene_id in *outputs {
            output_values.push(node_gene(&organism.node_genes, *node_gene_id).value);
        }

        output_values
    };

    // Zero values
    organism
        .node_genes
        .iter_mut()
        .for_each(|connection_gene| connection_gene.value = 0.0);

    organism
        .connection_genes
        .iter_mut()
        .for_each(|connection_gene| connection_gene.value = 0.0);

    output_values
}

#[inline]
fn node_gene<'a>(node_genes: &'a Vec<NodeGene>, node_gene_id: NodeGeneId) -> &'a NodeGene {
    node_genes
        .iter()
        .find(|node_gene| node_gene_id == node_gene.id)
        .expect(format!("{}:{}:{}", file!(), line!(), column!()).as_str())
}

#[inline]
fn node_gene_mut<'a>(
    node_genes: &'a mut Vec<NodeGene>,
    node_gene_id: NodeGeneId,
) -> &'a mut NodeGene {
    node_genes
        .iter_mut()
        .find(|node_gene| node_gene_id == node_gene.id)
        .expect(format!("{}:{}:{}", file!(), line!(), column!()).as_str())
}

#[inline]
fn outgoing_connection_genes_mut<'a>(
    connection_genes: &'a mut Vec<ConnectionGene>,
    node_gene_id: NodeGeneId,
) -> impl Iterator<Item = &'a mut ConnectionGene> {
    connection_genes
        .iter_mut()
        .filter(move |connection_gene| node_gene_id == connection_gene.source_id)
}

#[inline]
fn incoming_connection_genes<'a>(
    connection_genes: &'a Vec<ConnectionGene>,
    node_gene_id: NodeGeneId,
) -> impl Iterator<Item = &'a ConnectionGene> {
    connection_genes
        .iter()
        .filter(move |connection_gene| node_gene_id == connection_gene.target_id)
}

#[inline]
fn mutate_add_connection(
    organism: &mut Organism,
    mutate_add_connection_history: &mut FxHashMap<(NodeGeneId, NodeGeneId), ConnectionGeneId>,
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

        let connection_gene = organism
            .connection_genes
            .iter_mut()
            .find(|connection_gene| {
                source_id == connection_gene.source_id && target_id == connection_gene.target_id
            });

        if let Some(connection_gene) = connection_gene {
            if connection_gene.state.is_disabled() {
                connection_gene.state = ConnectionGeneState::Enabled;
                break;
            } else {
                continue;
            }
        }

        if distance(&organism.connection_genes, target_id, source_id).is_finite() {
            continue;
        }

        let id = mutate_add_connection_history
            .get(&(source_id, target_id))
            .copied();

        let id = match id {
            Some(id) => id,
            None => {
                *last_connection_gene_id += 1;

                mutate_add_connection_history
                    .insert((source_id, target_id), *last_connection_gene_id);

                *last_connection_gene_id
            }
        };

        organism.connection_genes.push(ConnectionGene {
            id,
            source_id,
            target_id,
            state: ConnectionGeneState::Enabled,
            weight: 1.0,
            value: 0.0,
        });

        break;
    }
}

#[inline]
fn mutate_add_node(
    organism: &mut Organism,
    mutate_add_node_history: &mut FxHashMap<ConnectionGeneId, NodeGeneId>,
    mutate_add_connection_history: &mut FxHashMap<(NodeGeneId, NodeGeneId), ConnectionGeneId>,
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
        let source_id = connection_gene.source_id;
        let target_id = connection_gene.target_id;
        let id = mutate_add_node_history.get(&connection_gene.id).copied();
        let node_gene_id;
        let incoming_connection_gene_id;
        let outgoing_connection_gene_id;

        if let Some(id) = id {
            node_gene_id = id;

            incoming_connection_gene_id = mutate_add_connection_history
                .get(&(source_id, node_gene_id))
                .copied()
                .unwrap();

            outgoing_connection_gene_id = mutate_add_connection_history
                .get(&(node_gene_id, target_id))
                .copied()
                .unwrap();
        } else {
            *last_node_gene_id += 1;
            node_gene_id = *last_node_gene_id;
            *last_connection_gene_id += 1;
            incoming_connection_gene_id = *last_connection_gene_id;
            *last_connection_gene_id += 1;
            outgoing_connection_gene_id = *last_connection_gene_id;
        }

        node_genes.push(NodeGene {
            id: node_gene_id,
            category: NodeGeneCategory::Hidden,
            value: 0.0,
        });

        let weight = connection_gene.weight;

        connection_genes.push(ConnectionGene {
            id: incoming_connection_gene_id,
            source_id,
            target_id: node_gene_id,
            state: ConnectionGeneState::Enabled,
            weight: 1.0,
            value: 0.0,
        });

        connection_genes.push(ConnectionGene {
            id: outgoing_connection_gene_id,
            source_id: node_gene_id,
            target_id,
            state: ConnectionGeneState::Enabled,
            weight,
            value: 0.0,
        });
    }
}

#[inline]
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
                // dbg!(connection_genes.len());
                // dbg!(visited.capacity());

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

    // dbg!(connection_genes.len());
    // dbg!(visited.capacity());
    Distance::Infinite
}

#[inline]
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

#[inline]
pub fn reproduce(
    population: &mut Population,
    genus: Vec<Species>,
    mut fitness: impl FnMut(&mut EvaluationContext) -> f64,
    thread_rng: &mut rand::rngs::ThreadRng,
) {
    use rand::distributions::weighted::alias_method::WeightedIndex;
    use rand::distributions::Distribution;
    use rand::seq::SliceRandom;

    let total_fitness = genus
        .iter()
        .map(|species| species.fitness(&population.organisms))
        .sum::<CheckedF64>();

    let total_offspring = population.organisms.len();

    // let mutate_mate_distribution = WeightedIndex::new(vec![
    //     PROBABILITY_MUTATE_ONLY,
    //     PROBABILITY_MATE_ONLY,
    //     PROBABILITY_MATE_AND_MUTATE,
    // ])
    // .expect(dbg!(""));

    let mutate_distribution = WeightedIndex::new(vec![
        PROBABILITY_ADD_NODE,
        PROBABILITY_ADD_CONNECTION,
        PROBABILITY_MUTATE_WEIGHTS,
    ])
    .expect(format!("{}:{}:{}", file!(), line!(), column!()).as_str());

    let organisms = &mut population.organisms;
    organisms.reserve(organisms.len() + genus.len());

    for species in genus {
        let offspring = (total_offspring as f64 * species.fitness(organisms).as_f64()
            / total_fitness.as_f64())
        .ceil() as usize;

        let organism_indices = ::core::iter::once(species.representative)
            .chain(
                species
                    .remaining
                    .iter()
                    .map(|speciated_organism| speciated_organism.organism_index),
            )
            .collect::<Vec<_>>();

        for _ in 0..offspring {
            let organism_index = *organism_indices
                .choose(thread_rng)
                .expect(format!("{}:{}:{}", file!(), line!(), column!()).as_str());

            let mut cloned = unsafe { organisms.get_unchecked(organism_index) }.clone();
            let sample = mutate_distribution.sample(thread_rng);

            match sample {
                0 => mutate_add_node(
                    &mut cloned,
                    &mut population.mutate_add_node_history,
                    &mut population.mutate_add_connection_history,
                    &mut population.last_node_gene_id,
                    &mut population.last_connection_gene_id,
                    thread_rng,
                ),
                1 => mutate_add_connection(
                    &mut cloned,
                    &mut population.mutate_add_connection_history,
                    &mut population.last_connection_gene_id,
                    thread_rng,
                ),
                2 => mutate_change_connection_weight(&mut cloned, thread_rng),
                _ => unreachable!(),
            }

            cloned.fitness = CheckedF64::new(fitness(&mut EvaluationContext {
                organism: &mut cloned,
                inputs: &population.inputs,
                outputs: &population.outputs,
            }));

            if cloned.fitness.as_f64() >= 99.9 {
                println!("{:#?}", cloned);
                ::std::process::exit(0);
            }

            organisms.push(cloned);
        }
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1f64 / (1f64 + (-x).exp())
}

// let mut genes1 = genes1.iter().peekable();
// let mut genes2 = genes2.iter().peekable();
//
// while let (Some(gene1), Some(gene2)) = (genes1.peek(), genes2.peek()) {
//     match gene1.id.cmp(&gene2.id) {
//         Ordering::Less => {
//             number_of_disjoint_genes += 1;
//
//             let _ = genes1.next();
//         }
//         Ordering::Equal => {
//             number_of_matching_genes += 1;
//             total_weight_difference += (gene1.weight - gene2.weight).abs();
//
//             let _ = genes1.next();
//             let _ = genes2.next();
//         }
//         Ordering::Greater => {
//             number_of_disjoint_genes += 1;
//
//             let _ = genes2.next();
//         }
//     }
// }
//
// number_of_excess_genes += genes1.count();
// number_of_excess_genes += genes2.count();

#[inline]
pub(crate) fn genetic_distance<'a, 'b>(
    genes1: &[ConnectionGene],
    genes2: &[ConnectionGene],
) -> CheckedF64 {
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

    let max_len = ::core::cmp::max(1, ::core::cmp::max(genes1_len, genes2_len)) as f64;

    return CheckedF64::new(
        (number_of_excess_genes as f64 * COEFFICIENT_EXCESS_GENES
            + number_of_disjoint_genes as f64 * COEFFICIENT_DISJOINT_GENES)
            / max_len
            + average_weight_difference * COEFFICIENT_WEIGHT_DIFFERENCE,
    );
}

#[inline]
fn cost(expected_output: &[f64], actual_output: &[f64]) -> f64 {
    assert_eq!(expected_output.len(), actual_output.len());

    expected_output
        .iter()
        .zip(actual_output.iter())
        .map(|(expected_value, actual_value)| (expected_value - actual_value).powi(2))
        .fold(0f64, |accumulator, x| accumulator + x)
}

#[inline]
pub fn average_cost(expected_outputs: &[&[f64]], actual_outputs: &[&[f64]]) -> f64 {
    assert_eq!(expected_outputs.len(), actual_outputs.len());

    for (expected_output, actual_output) in expected_outputs.iter().zip(actual_outputs.iter()) {
        assert_eq!(expected_output.len(), actual_output.len());
    }

    expected_outputs
        .iter()
        .zip(actual_outputs.iter())
        .map(|(expected_output, actual_output)| cost(expected_output, actual_output))
        .fold(0f64, |accumulator, x| accumulator + x)
        / (actual_outputs.len() as f64)
}

type Fitness = CheckedF64;

pub fn speciate(organisms: &[Organism], target_number_of_species: usize) -> Vec<Species> {
    let mut species_list = Vec::<Species>::with_capacity(target_number_of_species);

    let mut organisms_sorted_by_fitness: Vec<(OrganismIndex, Fitness)> = organisms
        .iter()
        .enumerate()
        .map(|(index, organism)| (index, organism.fitness))
        .collect();

    organisms_sorted_by_fitness.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    {
        let (representative, remaining) = organisms_sorted_by_fitness
            .split_first()
            .expect(format!("{}:{}:{}", file!(), line!(), column!()).as_str());

        // dbg!(representative.1);
        let representative = representative.0;

        let remaining = remaining
            .iter()
            .copied()
            .map(|(organism_index, _fitness)| SpeciatedOrganism {
                organism_index,
                genetic_distance: genetic_distance(
                    &organisms[organism_index].connection_genes,
                    &organisms[representative].connection_genes,
                ),
            })
            .collect();

        species_list.push(Species {
            representative,
            remaining,
        });
    }

    for _ in 1..target_number_of_species {
        species_list
            .iter_mut()
            .for_each(Species::sort_by_genetic_distance);

        let species_representative_candidates = species_list
            .iter()
            .enumerate()
            .filter_map(|(index, species)| {
                if species.remaining.last()?.genetic_distance == CheckedF64::zero() {
                    None
                } else {
                    Some((index, species.remaining.len() - 1))
                }
            })
            .collect::<Vec<_>>();

        if species_representative_candidates.is_empty() {
            break;
        }

        let (index, _) = species_representative_candidates
            .iter()
            .map(|(species_index, speciated_organism_index)| {
                (
                    species_index,
                    species_list
                        .iter()
                        .map(|species| {
                            genetic_distance(
                                &organisms[species.representative].connection_genes,
                                &organisms[species_list[*species_index].remaining
                                    [*speciated_organism_index]
                                    .organism_index]
                                    .connection_genes,
                            )
                        })
                        .min(),
                )
            })
            .max_by(|a, b| a.1.cmp(&b.1))
            .expect(format!("{}:{}:{}", file!(), line!(), column!()).as_str());

        let new_representative = species_list[*index]
            .remaining
            .pop()
            .expect(format!("{}:{}:{}", file!(), line!(), column!()).as_str())
            .organism_index;

        let mut new_remaining = Vec::new();

        for species in species_list.iter_mut() {
            new_remaining.extend(species.remaining.drain_filter(|speciated_organism| {
                let new_genetic_distance = genetic_distance(
                    &organisms[new_representative].connection_genes,
                    &organisms[speciated_organism.organism_index].connection_genes,
                );

                if speciated_organism.genetic_distance > new_genetic_distance {
                    speciated_organism.genetic_distance = new_genetic_distance;

                    true
                } else {
                    false
                }
            }));
        }

        species_list.push(Species {
            representative: new_representative,
            remaining: new_remaining,
        });
    }

    return species_list;
}

pub fn eliminate(population: &mut Population, genus: Vec<Species>) {
    let organisms = &mut population.organisms;
    let population_size = organisms.len();
    let ideal_population_size = population.ideal_population_size;
    let total_deaths = population_size - ideal_population_size;
    let mortal_population_size = population_size - genus.len();
    let mortality_rate = total_deaths as f64 / mortal_population_size as f64;

    let mut to_be_eliminated_organisms = genus
        .into_iter()
        .flat_map(|species| {
            let deaths = (species.remaining.len() as f64 * mortality_rate).ceil() as usize;

            let mut organism_indices = ::core::iter::once(species.representative)
                .chain(
                    species
                        .remaining
                        .into_iter()
                        .map(|speciated_organism| speciated_organism.organism_index),
                )
                .collect::<Vec<_>>();

            organism_indices
                .sort_unstable_by(|a, b| organisms[*b].fitness.cmp(&organisms[*a].fitness));

            organism_indices
                .drain(organism_indices.len() - deaths..)
                .collect::<Vec<OrganismIndex>>()
        })
        .collect::<Vec<OrganismIndex>>();

    to_be_eliminated_organisms.sort_unstable_by(|a, b| b.cmp(a));

    for organism_index in to_be_eliminated_organisms {
        organisms.remove(organism_index);
    }

    for _ in organisms.len()..ideal_population_size {
        organisms.push(population.initial_organism.clone());
    }
}

pub fn exponential_linear_unit(x: f64, a: f64) -> f64 {
    if x > 0f64 {
        x
    } else {
        (x.exp() - 1f64) * a
    }
}

pub fn softmax(slice: &mut [f64]) {
    let mut sum = 0.0;

    for value in &mut *slice {
        *value = value.exp();
        sum += *value;
    }

    for value in &mut *slice {
        *value /= sum;
    }
}