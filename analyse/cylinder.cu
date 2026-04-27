#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"
#include <cmath>


// cell-cell interaction parameter
const auto r_max = 1.f;  // Max Radius where cells interact with each other
const auto r_min = 0.5f;  // the force is 0 for cells with this distance -> equilibrium distance

// simulation parameter
const auto n_time_steps = 10u;
const auto dt = 0.05f;  // size of the timesteps -> smaller = more detailed and slower

// for creating the cell sheet we define the rows and columes size and not every single cell
const int n_ring = 12;
const int n_schicht = 5;
const auto n_cells = n_ring * n_schicht;        // LATER 100 * 359 cells


// Parameter for cylinder
const auto R_zylinder = n_ring * r_min / (2.f * (float)M_PI); // same radius as calculated in python for 359 cells
const auto k_radial = 1.0f;  // higher number = cells stay better at clyinder shape


// counter for the size of the neighborhood of a cell
// LATER : could be used to proove if neighbor cell was activated last time step to create the puls movement
__device__ int* d_neig;

// R_zylinder als device-Konstante verfügbar machen
__device__ float  d_R_dev;

// Xi = position of cell i
// r = vector from j to i
// dist = distance between i and j
// i = index of i
// j = index of j
// calculating a force away from j = r * F
__device__ float3 simulation_step(
    float3 Xi, float3 r, float dist, int i, int j)
{
    // The forces are calaculated for cell tupels that are not the same and where distance is not too big
    float3 dF{0};


    if (i == j) {
        // Aktuellen Abstand von der z-Achse berechnen
        float r_ist = sqrtf(Xi.x * Xi.x + Xi.y * Xi.y);

        if (r_ist > 0.f) {
            // Radiale Richtung (normiert, zeigt von z-Achse weg)
            float3 r_hat = {Xi.x / r_ist, Xi.y / r_ist, 0.f};

            // Kraft proportional zur Abweichung vom Soll-Radius
            // positiv = nach außen, negativ = nach innen
            float abweichung = R_zylinder - r_ist;
            dF = r_hat * (k_radial * abweichung);
        }
        return dF;
    }

    if (dist > r_max) return dF;

    d_neig[i] += 1;

    auto a = 1.f;
    auto F = a * (1.f - 2.f * dist);
    dF = r * F / dist;
    return dF;
}

int main(int argc, const char* argv[])
{

    // R auf GPU kopieren
    cudaMemcpyToSymbol(d_R_dev, &R_zylinder, sizeof(float));

    Solution<float3, Gabriel_solver> cells{n_cells, 50, r_max};

    // ── Zylindrische Initialisierung ───────────────────────────────
    // direkt aus deinem Python-Code übersetzt
    float dz       = r_min * sqrtf(3.f) / 2.f;   // Abstand zwischen Ringen
    float d_winkel = 2.f * (float)M_PI / (float)n_ring;       // Winkel zwischen Zellen im Ring

    for (int s = 0; s < n_schicht; s++) {
        for (int p = 0; p < n_ring; p++) {
            int   idx    = s * n_ring + p;
            float winkel = p * d_winkel;

            // Optional: Hexgitter-Versatz für bessere Packung
            if (s % 2 == 1) winkel += d_winkel * 0.5f;

            cells.h_X[idx] = {
                R_zylinder * cosf(winkel),   // x
                R_zylinder * sinf(winkel),   // y
                s * dz                        // z
            };
        }
    }

    cells.copy_to_device();

    Property<int> h_neig{n_cells, "neighbours"};
    cudaMemcpyToSymbol(d_neig, &h_neig.d_prop, sizeof(d_neig));

    auto fun = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) {
        thrust::fill(thrust::device, h_neig.d_prop, h_neig.d_prop + n, 0);
    };


    // for single cell here the cells get their position
    // cells.h_X[0] = {0.f, 0.f, 0.f};  //the position of this cell is the origin x=0, y=0, z=0
    // cells.copy_to_device();

    Vtk_output output{"cylinder"};
    // in every time step the simulation_step function is called one time
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        h_neig.copy_to_host(); // the number of neighbors gets saved
        cells.take_step<simulation_step>(dt, fun);
        output.write_positions(cells);
        output.write_property(h_neig);
    }
    return 0;
}