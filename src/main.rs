use ::neat::*;

const POPULATION_SIZE: usize = 400;
const NUMBER_OF_INPUTS: usize = 2;
const NUMBER_OF_OUTPUTS: usize = 1;
const NUMBER_OF_SPECIES: usize = 20;

fn main() {
    test_performance();
    unsafe {
        use ::core::arch::x86_64::_mm256_set1_ps;
        use ::core::arch::x86_64::_mm256_setzero_ps;
        use ::core::cell::Cell;

        let edges: Box<[Box<[EdgeVector]>]> = vec![
            vec![
                EdgeVector {
                    weight: _mm256_set1_ps(1.0)
                };
                4
            ]
            .into_boxed_slice(),
            vec![
                EdgeVector {
                    weight: _mm256_set1_ps(1.0)
                };
                2
            ]
            .into_boxed_slice(),
        ]
        .into_boxed_slice();

        let vertices: Box<[Box<[VertexVector]>]> = vec![
            vec![
                VertexVector {
                    value: Cell::new(_mm256_set1_ps(1.0))
                };
                2
            ]
            .into_boxed_slice(),
            vec![
                VertexVector {
                    value: Cell::new(_mm256_setzero_ps())
                };
                2
            ]
            .into_boxed_slice(),
            vec![
                VertexVector {
                    value: Cell::new(_mm256_setzero_ps())
                };
                1
            ]
            .into_boxed_slice(),
        ]
        .into_boxed_slice();

        let instant = std::time::Instant::now();
        feed_forward(vertices, edges);
        eprintln!("feed_forward: {} ns", instant.elapsed().as_nanos());
    }

    let mut thread_rng = rand::thread_rng();
    // let mut generation = 0;

    // eprintln!("GENERATION #0");

    // Initialize population
    // let instant = std::time::Instant::now();

    let mut population = initialize(
        POPULATION_SIZE,
        NUMBER_OF_INPUTS,
        NUMBER_OF_OUTPUTS,
        fitness,
    );

    let mut loop_count = 0;
    let mut duration = std::time::Duration::default();
    let genus = &mut Vec::<Species>::with_capacity(NUMBER_OF_SPECIES);
    let organisms_sorted_by_fitness = &mut Vec::<(OrganismIndex, Fitness)>::new();
    let organism_indices = &mut Vec::<usize>::new();

    loop {
        loop_count += 1;
        let instant = std::time::Instant::now();
        speciate(
            &population.organisms,
            NUMBER_OF_SPECIES,
            genus,
            organisms_sorted_by_fitness,
        );
        (&mut population.organisms);
        reproduce(
            &mut population,
            genus.to_vec(),
            fitness,
            &mut thread_rng,
            organism_indices,
        );
        (&mut population.organisms);
        speciate(
            &population.organisms,
            NUMBER_OF_SPECIES,
            genus,
            organisms_sorted_by_fitness,
        );
        (&mut population.organisms);
        eliminate_a(&mut population.organisms, genus.to_vec());
        (&mut population.organisms);
        speciate(
            &population.organisms,
            NUMBER_OF_SPECIES,
            genus,
            organisms_sorted_by_fitness,
        );
        eliminate_b(&mut population, genus.to_vec());
        age(&mut population.organisms);

        refill(
            &mut population.organisms,
            population.ideal_population_size,
            &population.initial_organism,
        );

        duration += instant.elapsed();

        if loop_count == 1000 {
            eprintln!("loop: {} ms", duration.as_millis());
            loop_count = 0;
            duration = std::time::Duration::default();
        }
    }
}

#[inline]
fn fitness(context: &mut EvaluationContext) -> f32 {
    let mut output_values_0 = Vec::<Value>::new();
    evaluate(&mut *context, &[0.0, 0.0], &mut output_values_0);

    let mut output_values_1 = Vec::<Value>::new();
    evaluate(&mut *context, &[0.0, 1.0], &mut output_values_1);

    let mut output_values_2 = Vec::<Value>::new();
    evaluate(&mut *context, &[1.0, 0.0], &mut output_values_2);

    let mut output_values_3 = Vec::<Value>::new();
    evaluate(&mut *context, &[1.0, 1.0], &mut output_values_3);

    1.0 - average_cost(
        &[&[0.0], &[1.0], &[1.0], &[0.0]],
        &[
            &output_values_0,
            &output_values_1,
            &output_values_2,
            &output_values_3,
        ],
    ) / 4.0
}
