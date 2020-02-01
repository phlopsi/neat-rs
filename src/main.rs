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

    // eprintln!(
    //     "Initialize population: {} µs",
    //     instant.elapsed().as_micros()
    // );

    // Speciate population
    // let instant = std::time::Instant::now();
    let mut genus = speciate(&population.organisms, NUMBER_OF_SPECIES);
    // eprintln!("Speciate population: {} µs", instant.elapsed().as_micros());

    // genus.iter().enumerate().for_each(|(index, species)| {
    //     eprintln!(
    //         "Species #{}: {} organisms / fitness: {}",
    //         index,
    //         1 + species.remaining.len(),
    //         species.fitness(&population.organisms).as_f64()
    //     );
    // });

    loop {
        // generation += 1;
        // eprintln!("GENERATION #{}", generation);

        // Reproduce population
        // let instant = std::time::Instant::now();
        reproduce(&mut population, genus, fitness, &mut thread_rng);
        // eprintln!("Reproduce population: {} µs", instant.elapsed().as_micros());

        // Speciate population
        // let instant = std::time::Instant::now();
        genus = speciate(&population.organisms, NUMBER_OF_SPECIES);
        // eprintln!("Speciate population: {} µs", instant.elapsed().as_micros());

        // Eliminate population
        // let instant = std::time::Instant::now();
        eliminate(&mut population, genus);
        // eprintln!("Eliminate population: {} µs", instant.elapsed().as_micros());

        // Speciate population
        // let instant = std::time::Instant::now();
        genus = speciate(&population.organisms, NUMBER_OF_SPECIES);
        // eprintln!("Speciate population: {} µs", instant.elapsed().as_micros());

        // genus.iter().enumerate().for_each(|(index, species)| {
        //     eprintln!(
        //         "Species #{}: {} organisms / fitness: {}",
        //         index,
        //         1 + species.remaining.len(),
        //         species.fitness(&population.organisms).as_f64()
        //     );
        // });

        // println!("\nPress ENTER to continue...");

        // ::std::io::stdin()
        //     .read_line(&mut ::std::string::String::default())
        //     .unwrap();
    }
}

#[inline]
fn fitness(context: &mut EvaluationContext) -> f64 {
    100.0
        - average_cost(
            &[&[0f64], &[1f64], &[0f64], &[1f64]],
            &[
                &evaluate(&mut *context, &[0f64, 0f64]),
                &evaluate(&mut *context, &[1f64, 0f64]),
                &evaluate(&mut *context, &[1f64, 1f64]),
                &evaluate(&mut *context, &[0f64, 1f64]),
            ],
        ) * 100.0
            / 4.0
}
