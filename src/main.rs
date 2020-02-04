use ::neat::*;

const POPULATION_SIZE: usize = 400;
const NUMBER_OF_INPUTS: usize = 2;
const NUMBER_OF_OUTPUTS: usize = 1;
const NUMBER_OF_SPECIES: usize = 20;

fn main() {
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

    let mut genus;
    let mut loop_count = 0;
    let mut duration = std::time::Duration::default();

    loop {
        loop_count += 1;
        let instant = std::time::Instant::now();
        genus = speciate(&population.organisms, NUMBER_OF_SPECIES);
        reproduce(&mut population, genus, fitness, &mut thread_rng);
        genus = speciate(&population.organisms, NUMBER_OF_SPECIES);
        eliminate_a(&mut population.organisms, genus);
        genus = speciate(&population.organisms, NUMBER_OF_SPECIES);
        eliminate_b(&mut population, genus);
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
    1.0 - average_cost(
        &[&[0.0], &[1.0], &[0.0], &[1.0]],
        &[
            &evaluate(&mut *context, &[0.0, 0.0]),
            &evaluate(&mut *context, &[1.0, 0.0]),
            &evaluate(&mut *context, &[1.0, 1.0]),
            &evaluate(&mut *context, &[0.0, 1.0]),
        ],
    ) / 4.0
}
