#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"
#include <cmath> // For sin, M_PI
#include <vector> // For std::vector

// Define M_PI if not available in some environments
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Constants for the simulation and cylinder
const auto r_max = 1.f; // Max Radius where cells interact with each other
const auto r_min = 0.5f; // The force is 0 for cells with this distance -> equilibrium distance
const auto n_time_steps = 200u; // More steps to see pulsation
const auto dt = 0.05f; // Size of the timesteps (smaller = more detailed and slower)

// Cylinder specific constants
const auto base_radius_initial = 2.0f; // Initial radius for cell placement
const auto cylinder_length = 10.0f;
const auto num_segments_z = 20u;
const auto num_points_per_circle = 30u; // More points for a smoother cylinder outline
const auto n_cells = num_segments_z * num_points_per_circle;

const auto oscillation_amplitude = 0.5f; // How much the radius changes (1.5 to 2.0)
const auto oscillation_frequency = 0.5f; // Frequency of pulsation 

// Force function - currently uses the basic force from foundation.cu.
// For a clear pulsation, comment out the `cells.take_step` line in main
// or modify this function to only maintain the cylindrical structure without inter-particle forces
// that could distort the intended pumping motion.
__device__ float3 simulation_step(
    float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;
    if (dist > r_max) return dF;

    auto a = 1.f;
    auto F = a * (1.f - 2.f * dist);

    dF = r * F / dist;
    return dF;
}

int main(int argc, const char* argv[])
{
    Solution<float3, Tile_solver> cells{n_cells};

    std::vector<float2> initial_radial_components(n_cells);
    std::vector<float> initial_z_coords(n_cells);

    float dz = cylinder_length / (num_segments_z - 1);
    float d_angle = 2.0f * M_PI / num_points_per_circle;

    for (unsigned int k = 0; k < num_segments_z; ++k) {
        float z_coord = k * dz - cylinder_length / 2.0f;

        for (unsigned int l = 0; l < num_points_per_circle; ++l) {
            float angle = l * d_angle;
            float x_component = cos(angle);
            float y_component = sin(angle);

            unsigned int index = k * num_points_per_circle + l;

            initial_radial_components[index] = {x_component, y_component};
            initial_z_coords[index] = z_coord;

            cells.h_X[index] = {
                base_radius_initial * x_component,
                base_radius_initial * y_component,
                z_coord
            };
        }
    }

    cells.copy_to_device();

    Vtk_output output{"pulsecylinder"};

    const auto wave_number = 2.0f * M_PI / cylinder_length;
    const auto omega = 2.0f * M_PI * oscillation_frequency;

    for (auto time_step = 0u; time_step <= n_time_steps; ++time_step) {
        float t = time_step * dt;

        for (unsigned int i = 0; i < n_cells; ++i) {
            float z = initial_z_coords[i];

            float local_radius =
                base_radius_initial *
                (1.0f + oscillation_amplitude * sin(omega * t - wave_number * z));

            cells.h_X[i].x = initial_radial_components[i].x * local_radius;
            cells.h_X[i].y = initial_radial_components[i].y * local_radius;
            cells.h_X[i].z = z;
        }

        cells.copy_to_device();

        // For clear pulsation, comment out cells.take_step... below:
        // cells.take_step<simulation_step>(dt);

        output.write_positions(cells);
    }

    return 0;
}