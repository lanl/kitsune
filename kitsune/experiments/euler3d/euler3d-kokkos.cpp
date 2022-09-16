/// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace std;

#include "Kokkos_DualView.hpp"
template <typename T>
using View = Kokkos::DualView<T*>;

struct Float3 {
  float x, y, z;
};

#define block_length 1

/*
 * Options
 *
 */
#define GAMMA 1.4
#define ITERATIONS 2000
#define NDIM 3
#define NNB 4
#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


#ifdef restrict
#define __restrict restrict
#else
#define __restrict
#endif

template <typename T>
void cpy(View<T> &dst, View<T> &src, int N) {
  src.sync_device();
  dst.sync_device();
  Kokkos::parallel_for("copy", N, KOKKOS_LAMBDA(const int &i) {
    dst.d_view(i) = src.d_view(i);
  });
  Kokkos::fence();
  dst.modify_device();
}

void dump(View<float> &variables, int nel, int nelr)
{
  variables.sync_host();

  {
    std::ofstream file("density-kokkos.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++)
      file << variables.h_view(i + VAR_DENSITY*nelr) << endl;
  }

  {
    std::ofstream file("momentum-kokkos.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++) {
      for(int j = 0; j != NDIM; j++)
        file << variables.h_view(i + (VAR_MOMENTUM+j)*nelr) << " ";
      file << endl;
    }
  }

  {
    ofstream file("density_energy-kokkos.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++)
      file << variables.h_view(i + VAR_DENSITY_ENERGY*nelr) << endl;
  }

}

void initialize_variables(int nelr,
                          View<float> & variables,
                          View<float> &ff_variable)
{
  variables.sync_device();
  ff_variable.sync_device();
  variables.modify_device();
  Kokkos::parallel_for("initialize_variables", nelr,
        KOKKOS_LAMBDA(const int &i) {
    for(int j = 0; j < NVAR; j++)
      variables.d_view(i + j*nelr) = ff_variable.d_view(j);
  });
  Kokkos::fence();
}

KOKKOS_FORCEINLINE_FUNCTION
void compute_flux_contribution(const float density,
                               const Float3& momentum,
                               const float density_energy,
                               const float pressure,
                               Float3& velocity,
                               Float3& fc_momentum_x,
                               Float3& fc_momentum_y,
                               Float3& fc_momentum_z,
                               Float3& fc_density_energy)
{
  fc_momentum_x.x = velocity.x*momentum.x + pressure;
  fc_momentum_x.y = velocity.x*momentum.y;
  fc_momentum_x.z = velocity.x*momentum.z;

  fc_momentum_y.x = fc_momentum_x.y;
  fc_momentum_y.y = velocity.y*momentum.y + pressure;
  fc_momentum_y.z = velocity.y*momentum.z;

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  fc_momentum_z.z = velocity.z*momentum.z + pressure;

  float de_p = density_energy+pressure;
  fc_density_energy.x = velocity.x*de_p;
  fc_density_energy.y = velocity.y*de_p;
  fc_density_energy.z = velocity.z*de_p;
}

KOKKOS_FORCEINLINE_FUNCTION
void compute_velocity(float density,
                      const Float3& momentum,
                      Float3& velocity)
{
  velocity.x = momentum.x / density;
  velocity.y = momentum.y / density;
  velocity.z = momentum.z / density;
}

KOKKOS_FORCEINLINE_FUNCTION
float compute_speed_sqd(const Float3 &velocity)
{
  return velocity.x*velocity.x +
         velocity.y*velocity.y +
         velocity.z*velocity.z;
}

KOKKOS_FORCEINLINE_FUNCTION
float compute_pressure(float density,
                       float density_energy,
                       float speed_sqd)
{
  return (float(GAMMA)-float(1.0f))*(density_energy -
          float(0.5f)*density*speed_sqd);
}

KOKKOS_FORCEINLINE_FUNCTION
float compute_speed_of_sound(float density, float pressure)
{
  return sqrtf(float(GAMMA)*pressure/density);
}

void compute_step_factor(int nelr,
                        View<float>& __restrict variables,
                        View<float>& areas,
                        View<float>& __restrict step_factors)
{
  variables.sync_device();
  areas.sync_device();
  step_factors.sync_device();
  step_factors.modify_device();

  Kokkos::parallel_for("compute_step_factor", nelr/block_length,
        KOKKOS_LAMBDA(const int &blk) {
    int b_start = blk*block_length;
    int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;

    for(int i = b_start; i < b_end; i++) {
      float density = variables.d_view(i + VAR_DENSITY*nelr);

      Float3 momentum;
      momentum.x = variables.d_view(i + (VAR_MOMENTUM+0)*nelr);
      momentum.y = variables.d_view(i + (VAR_MOMENTUM+1)*nelr);
      momentum.z = variables.d_view(i + (VAR_MOMENTUM+2)*nelr);

      float density_energy = variables.d_view(i + VAR_DENSITY_ENERGY*nelr);
      Float3 velocity;
      compute_velocity(density, momentum, velocity);
      float speed_sqd = compute_speed_sqd(velocity);
      float pressure = compute_pressure(density, density_energy, speed_sqd);
      float speed_of_sound = compute_speed_of_sound(density, pressure);

      // dt = float(0.5f) * sqrt(areas[i]) / (||v|| + c).... but
      // when we do time stepping, this later would need to be divided
      // by the area, so we just do it all at once
      step_factors.d_view(i) = float(0.5f) / (sqrtf(areas.d_view(i)) *
        (sqrtf(speed_sqd) + speed_of_sound));
    }
  });
  Kokkos::fence();
}

void compute_flux(int nelr,
                  View<int>& elements_surrounding_elements,
                  View<float>& normals,
                  View<float>& variables,
                  View<float>& fluxes,
                  View<float>& ff_variable,
                  const Float3 ff_flux_contribution_momentum_x,
                  const Float3 ff_flux_contribution_momentum_y,
                  const Float3 ff_flux_contribution_momentum_z,
                  const Float3 ff_flux_contribution_density_energy) {

  const float smoothing_coefficient = float(0.2f);
  elements_surrounding_elements.sync_device();
  normals.sync_device();
  variables.sync_device();
  fluxes.sync_device();
  ff_variable.sync_device();
  fluxes.modify_device();

  Kokkos::parallel_for("compute_flux", nelr/block_length,
        KOKKOS_LAMBDA(const int &blk) {
    int b_start = blk*block_length;
    int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;

    for(int i = b_start; i < b_end; ++i) {
      float density_i = variables.d_view(i + VAR_DENSITY*nelr);
      Float3 momentum_i;
      momentum_i.x = variables.d_view(i + (VAR_MOMENTUM+0)*nelr);
      momentum_i.y = variables.d_view(i + (VAR_MOMENTUM+1)*nelr);
      momentum_i.z = variables.d_view(i + (VAR_MOMENTUM+2)*nelr);

      float density_energy_i = variables.d_view(i + VAR_DENSITY_ENERGY*nelr);

      Float3 velocity_i;
      compute_velocity(density_i, momentum_i, velocity_i);
      float speed_sqd_i = compute_speed_sqd(velocity_i);
      float speed_i = sqrtf(speed_sqd_i);
      float pressure_i = compute_pressure(density_i,
                                          density_energy_i,
                                          speed_sqd_i);
      float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
      Float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
             flux_contribution_i_momentum_z;

      Float3 flux_contribution_i_density_energy;
      compute_flux_contribution(density_i, momentum_i,
                                density_energy_i, pressure_i,
                                velocity_i, flux_contribution_i_momentum_x,
                                flux_contribution_i_momentum_y,
                                flux_contribution_i_momentum_z,
                                flux_contribution_i_density_energy);

      float flux_i_density = float(0.0f);
      Float3 flux_i_momentum;
      flux_i_momentum.x = float(0.0f);
      flux_i_momentum.y = float(0.0f);
      flux_i_momentum.z = float(0.0f);
      float flux_i_density_energy = float(0.0f);

      Float3 velocity_nb;
      float density_nb, density_energy_nb;
      Float3 momentum_nb;
      Float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y,
             flux_contribution_nb_momentum_z;
      Float3 flux_contribution_nb_density_energy;
      float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

      for(int j = 0; j < NNB; j++) {
        Float3 normal;
        float normal_len;
        float factor;

        int nb = elements_surrounding_elements.d_view(i + j*nelr);
        normal.x = normals.d_view(i + (j + 0*NNB)*nelr);
        normal.y = normals.d_view(i + (j + 1*NNB)*nelr);
        normal.z = normals.d_view(i + (j + 2*NNB)*nelr);
        normal_len = sqrtf(normal.x*normal.x +
                          normal.y*normal.y +
                          normal.z*normal.z);

        if (nb >= 0) { // a legitimate neighbor
          density_nb = variables.d_view(nb + VAR_DENSITY*nelr);
          momentum_nb.x = variables.d_view(nb + (VAR_MOMENTUM)*nelr);
          momentum_nb.y = variables.d_view(nb + (VAR_MOMENTUM+1)*nelr);
          momentum_nb.z = variables.d_view(nb + (VAR_MOMENTUM+2)*nelr);
          density_energy_nb = variables.d_view(nb + VAR_DENSITY_ENERGY*nelr);
          compute_velocity(density_nb, momentum_nb, velocity_nb);
          speed_sqd_nb = compute_speed_sqd(velocity_nb);
          pressure_nb = compute_pressure(density_nb, density_energy_nb,
                                         speed_sqd_nb);
          speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
          compute_flux_contribution(density_nb, momentum_nb, density_energy_nb,
                                    pressure_nb, velocity_nb,
                                    flux_contribution_nb_momentum_x,
                                    flux_contribution_nb_momentum_y,
                                    flux_contribution_nb_momentum_z,
                                    flux_contribution_nb_density_energy);

          // artificial viscosity
          factor = -normal_len*smoothing_coefficient*float(0.5f) *
                        (speed_i + sqrtf(speed_sqd_nb) +
                         speed_of_sound_i + speed_of_sound_nb);
          flux_i_density += factor*(density_i-density_nb);
          flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
          flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
          flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
          flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

          // accumulate cell-centered fluxes
          factor = float(0.5f)*normal.x;
          flux_i_density += factor*(momentum_nb.x+momentum_i.x);
          flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x
                                    + flux_contribution_i_density_energy.x);
          flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x
                                    + flux_contribution_i_momentum_x.x);
          flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x
                                    + flux_contribution_i_momentum_y.x);
          flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x
                                    + flux_contribution_i_momentum_z.x);

          factor = float(0.5f)*normal.y;
          flux_i_density += factor*(momentum_nb.y+momentum_i.y);
          flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y
                                    + flux_contribution_i_density_energy.y);
          flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y
                                    + flux_contribution_i_momentum_x.y);
          flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y
                                    + flux_contribution_i_momentum_y.y);
          flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y
                                    + flux_contribution_i_momentum_z.y);

          factor = float(0.5f)*normal.z;
          flux_i_density += factor*(momentum_nb.z+momentum_i.z);
          flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z
                                    + flux_contribution_i_density_energy.z);
          flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z
                                    + flux_contribution_i_momentum_x.z);
          flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z
                                    + flux_contribution_i_momentum_y.z);
          flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z
                                    + flux_contribution_i_momentum_z.z);
        } else if(nb == -1) { // a wing boundary
          flux_i_momentum.x += normal.x*pressure_i;
          flux_i_momentum.y += normal.y*pressure_i;
          flux_i_momentum.z += normal.z*pressure_i;
        } else if(nb == -2) { // a far field boundary
          factor = float(0.5f)*normal.x;
          flux_i_density += factor*(ff_variable.d_view(VAR_MOMENTUM+0) +
                              momentum_i.x);
          flux_i_density_energy += factor*(ff_flux_contribution_density_energy.x
                                    + flux_contribution_i_density_energy.x);
          flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.x
                                    + flux_contribution_i_momentum_x.x);
          flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.x
                                    + flux_contribution_i_momentum_y.x);
          flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.x
                                    + flux_contribution_i_momentum_z.x);

          factor = float(0.5f)*normal.y;
          flux_i_density += factor*(ff_variable.d_view(VAR_MOMENTUM+1) +
                              momentum_i.y);
          flux_i_density_energy += factor*(ff_flux_contribution_density_energy.y
                                    + flux_contribution_i_density_energy.y);
          flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.y
                                    + flux_contribution_i_momentum_x.y);
          flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.y
                                    + flux_contribution_i_momentum_y.y);
          flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.y
                                    + flux_contribution_i_momentum_z.y);

          factor = float(0.5f)*normal.z;
          flux_i_density += factor*(ff_variable.d_view(VAR_MOMENTUM+2) +
                                  momentum_i.z);
          flux_i_density_energy += factor*(ff_flux_contribution_density_energy.z
                                    + flux_contribution_i_density_energy.z);
          flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.z
                                    + flux_contribution_i_momentum_x.z);
          flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.z
                                    + flux_contribution_i_momentum_y.z);
          flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.z
                                    + flux_contribution_i_momentum_z.z);
        }
      }
    fluxes.d_view(i + VAR_DENSITY*nelr) = flux_i_density;
    fluxes.d_view(i + (VAR_MOMENTUM+0)*nelr) = flux_i_momentum.x;
    fluxes.d_view(i + (VAR_MOMENTUM+1)*nelr) = flux_i_momentum.y;
    fluxes.d_view(i + (VAR_MOMENTUM+2)*nelr) = flux_i_momentum.z;
    fluxes.d_view(i + VAR_DENSITY_ENERGY*nelr) = flux_i_density_energy;
    }
  });
  Kokkos::fence();
}

void time_step(int j, int nelr,
               View<float>& old_variables,
               View<float>& variables,
               View<float>& step_factors,
               View<float>& fluxes)
{
  old_variables.sync_device();
  variables.sync_device();
  step_factors.sync_device();
  fluxes.sync_device();
  variables.modify_device();
  Kokkos::parallel_for("time_step", nelr/block_length,
      KOKKOS_LAMBDA(const int &blk) {
    int b_start = blk*block_length;
    int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;
    for(int i = b_start; i < b_end; ++i) {
      float factor = step_factors.d_view(i)/float(RK+1-j);
      variables.d_view(i + VAR_DENSITY*nelr) =
                old_variables.d_view(i + VAR_DENSITY*nelr) +
                factor*fluxes.d_view(i + VAR_DENSITY*nelr);
      variables.d_view(i + (VAR_MOMENTUM+0)*nelr) =
                old_variables.d_view(i + (VAR_MOMENTUM+0)*nelr) +
                factor*fluxes.d_view(i + (VAR_MOMENTUM+0)*nelr);
      variables.d_view(i + (VAR_MOMENTUM+1)*nelr) =
                old_variables.d_view(i + (VAR_MOMENTUM+1)*nelr) +
                factor*fluxes.d_view(i + (VAR_MOMENTUM+1)*nelr);
      variables.d_view(i + (VAR_MOMENTUM+2)*nelr) =
                old_variables.d_view(i + (VAR_MOMENTUM+2)*nelr) +
                factor*fluxes.d_view(i + (VAR_MOMENTUM+2)*nelr);
      variables.d_view(i + VAR_DENSITY_ENERGY*nelr) =
                old_variables.d_view(i + VAR_DENSITY_ENERGY*nelr) +
                factor*fluxes.d_view(i + VAR_DENSITY_ENERGY*nelr);
    }
  });
  Kokkos::fence();
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
  if (argc < 2) {
    cout << "specify data file name" << endl;
    return 1;
  }

  int iterations = ITERATIONS;
  if (argc > 2)
    iterations = atoi(argv[2]);

  const char* data_file_name = argv[1];

  // these need to be computed the first time in order to compute time step


  Kokkos::initialize(argc, argv); {

    View<float> ff_variable("ff_variable", NVAR);
    Float3 ff_flux_contribution_momentum_x,
           ff_flux_contribution_momentum_y,
           ff_flux_contribution_momentum_z;
    Float3 ff_flux_contribution_density_energy;

    // set far field conditions
    const float angle_of_attack =
          float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

    ff_variable.h_view(VAR_DENSITY) = float(1.4);

    float ff_pressure = float(1.0f);
    float ff_speed_of_sound =
          sqrtf(GAMMA * ff_pressure / ff_variable.h_view(VAR_DENSITY));
    float ff_speed = float(ff_mach) * ff_speed_of_sound;

    Float3 ff_velocity;
    ff_velocity.x = ff_speed * float(cos((float)angle_of_attack));
    ff_velocity.y = ff_speed * float(sin((float)angle_of_attack));
    ff_velocity.z = 0.0f;

    ff_variable.h_view(VAR_MOMENTUM + 0) =
        ff_variable.h_view(VAR_DENSITY) * ff_velocity.x;
    ff_variable.h_view(VAR_MOMENTUM + 1) =
        ff_variable.h_view(VAR_DENSITY) * ff_velocity.y;
    ff_variable.h_view(VAR_MOMENTUM + 2) =
        ff_variable.h_view(VAR_DENSITY) * ff_velocity.z;

    ff_variable.h_view(VAR_DENSITY_ENERGY) =
        ff_variable.h_view(VAR_DENSITY) *
            (float(0.5f) * (ff_speed * ff_speed)) +
        (ff_pressure / float(GAMMA - 1.0f));

    ff_variable.modify_host();

    Float3 ff_momentum;
    ff_momentum.x = ff_variable.h_view(VAR_MOMENTUM);
    ff_momentum.y = ff_variable.h_view(VAR_MOMENTUM + 1);
    ff_momentum.z = ff_variable.h_view(VAR_MOMENTUM + 2);
    compute_flux_contribution(ff_variable.h_view(VAR_DENSITY), ff_momentum,
          ff_variable.h_view(VAR_DENSITY_ENERGY), ff_pressure, ff_velocity,
          ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,
          ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);

    int nel;
    int nelr;

    // read in domain geometry
    ifstream file(data_file_name);
    file >> nel;
    nelr = block_length * ((nel / block_length) + min(1, nel % block_length));
    View<float> areas("areas", nelr);
    View<int> elements_surrounding_elements("elements_surrounding_elements",
                                            nelr * NNB);
    View<float> normals("normals", NDIM * NNB * nelr);

    // read in data
    for (int i = 0; i < nel; i++) {
      file >> areas.h_view(i);
      for (int j = 0; j < NNB; j++) {
        file >> elements_surrounding_elements.h_view(i + j * nelr);
        if (elements_surrounding_elements.h_view(i + j * nelr) < 0)
          elements_surrounding_elements.h_view(i + j * nelr) = -1;
        // it's coming in with Fortran numbering
        elements_surrounding_elements.h_view(i + j * nelr)--;

        for (int k = 0; k < NDIM; k++) {
          file >> normals.h_view(i + (j + k * NNB) * nelr);
          normals.h_view(i + (j + k*NNB)*nelr) =
                      -normals.h_view(i + (j + k*NNB)*nelr);
        }
      }
    }
    normals.modify_host();
    elements_surrounding_elements.modify_host();

    int last = nel-1;
    for(int i = nel; i < nelr; i++) {

      areas.h_view(i) = areas.h_view(last);

      for(int j = 0; j < NNB; j++) {
        // duplicate the last element
        elements_surrounding_elements.h_view(i + j*nelr) =
                elements_surrounding_elements.h_view(last + j*nelr);

        for(int k = 0; k < NDIM; k++)
          normals.h_view(i + (j + k*NNB)*nelr) =
                normals.h_view(last + (j + k*NNB)*nelr);
      }
    }
    normals.modify_host();
    areas.modify_host();
    elements_surrounding_elements.modify_host();

    // Create arrays and set initial conditions
    View<float> variables = View<float>("variables", nelr * NVAR);

    initialize_variables(nelr, variables, ff_variable);

    View<float> old_variables = View<float>("old_variables", nelr*NVAR);
    View<float> fluxes = View<float>("fluxes", nelr*NVAR);
    View<float> step_factors = View<float>("step_factors", nelr);

    // Begin iterations
    cout << iterations << " "; 
    auto start = chrono::steady_clock::now();
    double copy_total = 0.0;
    double sf_total = 0.0;
    double rk_total = 0.0;
    for(int i = 0; i < iterations; i++) {

      auto copy_start = chrono::steady_clock::now();
      cpy(old_variables, variables, nelr*NVAR);
      auto copy_end = chrono::steady_clock::now();
      copy_total += chrono::duration<double>(copy_end-copy_start).count();

      // for the first iteration we compute the time step
      auto sf_start = chrono::steady_clock::now();
      compute_step_factor(nelr, variables, areas, step_factors);
      auto sf_end = chrono::steady_clock::now();
      sf_total += chrono::duration<double>(sf_end-sf_start).count();

      auto rk_start = chrono::steady_clock::now();
      for(int j = 0; j < RK; j++) {
        compute_flux(nelr, elements_surrounding_elements, normals, variables,
                    fluxes, ff_variable,
                    ff_flux_contribution_momentum_x,
                    ff_flux_contribution_momentum_y,
                    ff_flux_contribution_momentum_z,
                    ff_flux_contribution_density_energy);
        time_step(j, nelr, old_variables, variables, step_factors, fluxes);
      }
      auto rk_end = chrono::steady_clock::now();
      rk_total += chrono::duration<double>(rk_end-rk_start).count();
    }

    auto end = chrono::steady_clock::now();
    cout << copy_total << " "
	 << sf_total << " "
	 << rk_total << " "
	 << chrono::duration<double>(end-start).count() << endl;
    dump(variables, nel, nelr);
  } Kokkos::finalize();

  return 0;
}
