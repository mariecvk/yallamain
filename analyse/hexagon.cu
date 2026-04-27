
#include "./include/dtypes.cuh"
#include "./include/inits.cuh"
#include "./include/property.cuh"
#include "./include/solvers.cuh"
#include "./include/vtk.cuh"
#include <cmath>

const auto r_max = 1.f;  // Max Radius where cells interact with each other
const auto r_min = 0.5f;  // the force is 0 for cells with this distance -> equilibrium distance
const auto n_time_steps = 10u;
const auto dt = 0.05f;  // size of the timesteps -> smaller = more detailed and slower

// for creating the cell sheet we define the rows and columes size and not every single cell
const int rows = 359;
const int columes = 10;
const auto n_cells = rows * columes;        // 200 * 359 cells

// counter for the size of the neighborhood of a cell
// LATER : could be used to proove if neighbor cell was activated last time step to create the puls movement
__device__ int* d_neig;

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
    if (i == j) return dF;   //if i and j are the same = no forces
    if (dist > r_max) return dF;  // if the distance between i and j ist too big = no forces

    // an array that counts how big the neighborhood of a cell i is
    d_neig[i] += 1;

    // the forces are calculated here

    // F is the physical quantity of Force -> it describes how strong and in which direction the force acts on
    // positiv F = repulsion/abstoßung
    // negative F = attraction/anziehung
    auto a = 1.f;

    // the Force function here is 0 for distance of 0,5
    // if the distance is 0 -> Force F is +1 -> max repulsion
    // if the distance is 1 = r_max -> F is -1 -> max attraction
    // => 0,5 is the normal cell distance here where everything is in equilibrium
    auto F = a * (1.f - 2.f * dist);

    // dF is the Force that acts from cell j to cell i (j->i)
    // => it describes the change of the position from cell i in time resulting from the force
    // later the function gets called many times for i and this means (because we return dF) that the positionchanges resulting from all neighborhood cells j are added
    dF = r * F / dist;
    return dF;
}

int main(int argc, const char* argv[])
{
    Solution<float3, Gabriel_solver> cells{n_cells, 50, r_max};

    // Creatign the hexagonal sheet
    float spacing = r_min;                    // distance between the cells
    float dy = spacing * sqrtf(3.f) / 2.f;   // distance between the rows of the cells


    for (int row = 0; row < rows; row++) {
        for (int colum = 0; colum < columes; colum++) {
            int idx = row * columes + colum;

            float x = colum * spacing;
            // the odd numbers get a position slightly more to the rigth so you get the Hexagon shape
            if (row % 2 == 1) x += spacing * 0.5f;

            float y = row * dy;
            float z = 0.f;   // z coordinate is 0 because we first have a 2D sheet

            cells.h_X[idx] = {x, y, z}; // this is the position of a cell
        }
    }

    cells.copy_to_device();

    // creating a place for storage for the neighborhood information
    Property<int> h_neig{n_cells, "neighbours"};
    cudaMemcpyToSymbol(d_neig, &h_neig.d_prop, sizeof(d_neig));

    auto fun = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) {
        thrust::fill(thrust::device, h_neig.d_prop, h_neig.d_prop + n, 0);
    };


    // for single cell here the cells get their position
    // cells.h_X[0] = {0.f, 0.f, 0.f};  //the position of this cell is the origin x=0, y=0, z=0
    // cells.copy_to_device();

    Vtk_output output{"hexasheet"};
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