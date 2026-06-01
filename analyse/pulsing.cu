// ============================================================
//  Simulation of a pulsating blood vessel cylinder
//
//  Basic idea:
//    - N cells lie on the surface of a cylinder
//    - Cells repel each other when they are too close and attract each
//      other when they are too far apart
//    - A time-dependent pulse wave changes the target radius along z
//    - The GPU computes all cell pairs in parallel at each time step
// ============================================================


// ── Yalla framework headers ───────────────────────────────────
#include "../include/dtypes.cuh"    // float3, point types
#include "../include/inits.cuh"     // helper functions for initialization
#include "../include/property.cuh"   // Property<T>: quantity per cell (e.g. number of neighbours)
#include "../include/solvers.cuh"    // Gabriel_solver: neighbour search on the GPU
#include "../include/vtk.cuh"       // Vtk_output: stores positions as .vtk files
#include <cmath>                    // sinf, cosf, sqrtf, M_PI
#include <iostream>             // std::cout for printing parameters    


// ============================================================
//  SECTION 1: PHYSICAL PARAMETERS (cell-cell interaction)
// ============================================================


const auto r_max = 1.f;
// Maximum distance at which two cells still "feel" each other.
// Cells farther apart than r_max completely ignore one another.


const auto r_min = 0.5f;
// Equilibrium distance: the force is zero at exactly this distance.
// Closer → repulsion, farther (up to r_max) → attraction.



// ============================================================
//  SECTION 2: SIMULATION PARAMETERS (time)
// ============================================================


const auto n_time_steps = 400u;
// Number of time steps. 200 × dt = 10 time units in total.
// Increase to ≥ 400 later for a visible pulse wave.


const auto dt = 0.05f;
// Size of one time step (Euler integration: x_new = x_old + F·dt).
// Smaller = more accurate, but slower. Too large → unstable simulation.



// ============================================================
//  SECTION 3: GEOMETRY (cylinder setup)
// ============================================================


const int n_ring    = 12;
// Number of cells per ring (= circumference of the cylinder in cells).


const int n_schicht = 5;
// Number of rings along the z-axis (= length of the cylinder in rings).


const auto n_cells = n_ring * n_schicht;
// Total number of cells = 12 × 5 = 60.
// Planned later: 100 × 359 = 35900 cells for a realistic blood vessel.



// ============================================================
//  SECTION 4: CYLINDER SHAPE & RESTORING FORCE
// ============================================================


const auto R_zylinder = n_ring * r_min / (2.f * (float)M_PI);
// Computes the base radius of the cylinder so that n_ring cells
// with spacing r_min fit seamlessly around the circumference.
// Formula: circumference = n_ring * r_min → R = circumference / (2π)


const auto k_radial = 1.0f;
// Stiffness of the restoring force toward the target radius.
// Higher = cells follow the target radius faster / more rigidly.
// Too high → overshooting, possible numerical instability.



// ============================================================
//  SECTION 5: DEVICE VARIABLES (GPU memory)
// ============================================================


__device__ int* d_neig;
// Pointer to an array on the GPU: d_neig[i] = number of neighbours of cell i.
// Reset and recomputed in every time step.
// __device__ = variable lives on the GPU, not on the CPU.


__device__ float d_R_dev;
// Base radius R_zylinder copied to the GPU.
// Needed because GPU (device) functions cannot directly access CPU constants.


__device__ float d_time;
// Current simulation time t, transferred from CPU to GPU before each time step
// (cudaMemcpyToSymbol).
// This lets the force function know where the pulse wave is at the moment.



// ============================================================
//  SECTION 6: WAVE PARAMETERS (pulse wave)
// ============================================================


const auto wave_amplitude = 0.15f;
// Amplitude of the pulse wave: how strongly the target radius deviates from the base radius.
// 0.15 means ±15% of r_min. Too large → cells overlap.


const auto wave_speed = 2.0f;
// Angular frequency ω: how fast the wave travels through the cylinder over time.
// Unit: radians per time unit. Higher = faster pulsation.


const auto wave_number = 0.5f;
// Wavenumber k: spatial frequency of the wave along the z-axis.
// k = 2π / wavelength. Higher = more wave peaks visible at the same time.



// ============================================================
//  SECTION 7: CORE FUNCTION – FORCE BETWEEN TWO CELLS
//
//  This function is called by the GPU for EVERY cell pair (i, j).
//  You only define the physics – Yalla handles the loop over all pairs.
//
//  Parameters:
//    Xi   – Position of cell i as an (x, y, z) vector
//    r    – Difference vector Xi − Xj (points from j to i)
//    dist – Distance |r| between i and j
//    i, j – Cell indices
//
//  Return value: dF – force contribution on cell i in this time step
// ============================================================


__device__ float3 simulation_step(
    float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};   // Force contribution, starts at 0



    // ── Special case: i == j (cell interacting with itself) ─────
    // Yalla also calls simulation_step with i == j.
    // Here we do not compute a cell-cell force,
    // but the radial restoring force toward the target radius.
    if (i == j) {


        // Current distance of the cell from the z-axis (= current radius)
        float r_ist = sqrtf(Xi.x * Xi.x + Xi.y * Xi.y);


        if (r_ist > 0.f) {  // Safety: avoid division by zero


            // Normalized direction vector pointing radially outward
            // r_hat points in the direction in which the force should act
            float3 r_hat = {Xi.x / r_ist, Xi.y / r_ist, 0.f};


            // ── Pulse wave formula ───────────────────────────────
            // Target radius is a sine wave travelling along +z:
            //   R_target(z, t) = R_base + A · sin(k·z − ω·t)
            //
            //   k·z  → spatial phase: the wave has different radii at different z
            //   ω·t  → temporal phase: the wave moves over time in the +z direction
            float R_target = d_R_dev
                           + wave_amplitude * sinf(wave_number * Xi.z
                                                   - wave_speed * d_time);


            // Deviation from the target radius:
            //   positive → cell is too far inside  → force outward
            //   negative → cell is too far outside → force inward
            float abweichung = R_target - r_ist;


            // Restoring force = direction × stiffness × deviation
            dF = r_hat * (k_radial * abweichung);
        }
        return dF;  // No further contribution for i == j
    }



    // ── Normal case: i ≠ j (force between two different cells) ─────
    if (dist > r_max) return dF;
    // Too far apart → no interaction, dF stays 0


    d_neig[i] += 1;
    // Increase neighbour counter for cell i (atomic would be safer, but this is sufficient here)


    // Linear force function:
    //   F = a · (1 − 2·dist)
    //
    //   dist = 0.0        → F = +1.0  (maximum repulsion, cells overlap)
    //   dist = r_min=0.5  → F =  0.0  (equilibrium, no force)
    //   dist = r_max=1.0  → F = -1.0  (maximum attraction, cells too far apart)
    auto a = 1.f; // Force strength coefficient
    auto F = a * (1.f - 2.f * dist);


    // Force vector: normalized direction vector (r/dist) × strength F
    // Positive F → points away from j (repulsion)
    // Negative F → points toward j (attraction)
    dF = r * F / dist;


    return dF;
}



// ============================================================
//  SECTION 8: main() – INITIALIZATION AND TIME LOOP
// ============================================================


int main(int argc, const char* argv[])
{
    // Copy base radius to the GPU so simulation_step can read it
    cudaMemcpyToSymbol(d_R_dev, &R_zylinder, sizeof(float));


    // Solution: main data structure of Yalla
    //   float3         → position type (x, y, z per cell)
    //   Gabriel_solver → neighbour search: only pairs with dist ≤ r_max are considered
    //   n_cells        → number of cells
    //   200            → maximum neighbours per cell (Gabriel graph parameter)
    //   r_max          → interaction radius
    Solution<float3, Gabriel_solver> cells{n_cells, 200, r_max};



    // ── Compute cylindrical initial positions ───────────────────
    // Cells are placed on a cylinder:
    //   - n_ring cells distributed evenly around the circumference (xy plane)
    //   - n_schicht rings stacked along the z-axis


    float dz = r_min * sqrtf(3.f) / 2.f;
    // Distance between two rings along z.
    // sqrtf(3)/2 ≈ 0.866 comes from the hexagonal lattice (densest packing).


    float d_winkel = 2.f * (float)M_PI / (float)n_ring;
    // Angular step between two neighbouring cells in the same ring.
    // 360° / n_ring = 30° per step for n_ring = 12.


    for (int s = 0; s < n_schicht; s++) {          // s = layer index (z direction)
        for (int p = 0; p < n_ring; p++) {         // p = position in the ring (angle)
            int   idx    = s * n_ring + p;         // linear index in the h_X array
            float winkel = p * d_winkel;           // angle of this cell in radians


            // Hexagonal offset: odd layers are shifted by half an angular step
            // → denser, more stable packing (like a brick pattern)
            if (s % 2 == 1) winkel += d_winkel * 0.5f;


            // Cylinder coordinates → Cartesian:
            //   x = R · cos(angle)
            //   y = R · sin(angle)
            //   z = s · dz
            cells.h_X[idx] = {
                R_zylinder * cosf(winkel),   // x-coordinate
                R_zylinder * sinf(winkel),   // y-coordinate
                s * dz                        // z-coordinate
            };
        }
    }


    cells.copy_to_device();
    // Transfer positions from CPU memory (h_X) to GPU memory (d_X).
    // From here on, take_step works on the GPU with d_X.



    // ── Initialize neighbour counters ──────────────────────────
    Property<int> h_neig{n_cells, "neighbours"};
    // Property: Yalla type for a scalar quantity per cell (here: int).
    // "neighbours" = name, stored that way in the .vtk file.


    cudaMemcpyToSymbol(d_neig, &h_neig.d_prop, sizeof(d_neig));
    // Write the GPU pointer to h_neig.d_prop into the __device__ variable d_neig,
    // so that simulation_step can write directly into the array.


    // Lambda function: called before each take_step (Generic_forces)
    // Here: reset neighbour counters to 0 so they are counted anew each time step
    auto fun = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) {
        thrust::fill(thrust::device, h_neig.d_prop, h_neig.d_prop + n, 0);
        // thrust::fill = GPU function, sets all n entries to 0
    };



    // ── Prepare output ──────────────────────────────────────────
    Vtk_output output{"pulsing"};
    // Creates files pulsing_0000.vtk, pulsing_0001.vtk, …
    // These can be visualized with ParaView or PyVista.



    // ── Main loop: time step by time step ───────────────────────
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {


        // Compute current simulation time and transfer it to the GPU
        // → simulation_step can read d_time to calculate the wave phase
        float current_time = time_step * dt;
        cudaMemcpyToSymbol(d_time, &current_time, sizeof(float));


        cells.copy_to_host();
        // GPU → CPU: fetch current positions for output


        h_neig.copy_to_host();
        // GPU → CPU: fetch neighbour counts for output


        cells.take_step<simulation_step>(dt, fun);
        // THE CORE CALL:
        //   1. fun() is called → reset neighbour counters
        //   2. GPU launches thousands of parallel threads
        //   3. Each thread calls simulation_step(Xi, r, dist, i, j)
        //   4. All dF contributions are summed for each cell i
        //   5. New position: d_X[i] += total_dF * dt  (Euler integration)


        output.write_positions(cells);
        // Write positions of all cells to pulsing_XXXX.vtk


        output.write_property(h_neig);
        // Write neighbour counts as a scalar field into the same .vtk file
        // → In ParaView, cells can be colored by neighbour count
    }


    return 0;
    // GPU memory is automatically released when the program ends
}
int main(int argc, const char* argv[])
+ {
    std::cout << "n_cells: " << n_cells << std::endl;
    std::cout << "n_ring: " << n_ring << std::endl;
    std::cout << "n_schicht: " << n_schicht << std::endl;
    std::cout << "n_time_steps: " << n_time_steps << ", dt: " << dt << std::endl;
    std::cout << "k_radial: " << k_radial << std::endl;
    std::cout << "R_zylinder: " << R_zylinder << std::endl;
    std::cout << "wave_amplitude: " << wave_amplitude << std::endl;
    std::cout << "wave_speed: " << wave_speed << std::endl;
    std::cout << "wave_number: " << wave_number << std::endl;
}