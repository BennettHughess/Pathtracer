#include "../include/cuda_metric.cuh"
#include <cmath>

// Computes second derivative of position and returns as a 4-vector
CudaVec4 CudaMetric::get_acceleration(const CudaVec4& pos, const CudaVec4& vel) {

    CudaVec4 acceleration;
    double rho, rho_s;
    double mass = params.black_hole_mass;

    switch (type) {

        case Metric::SphericalMinkowskiMetric:

            acceleration = {

                0,                                                  // time component

                pos[1]*vel[2]*vel[2]                                // r component
                + pos[1]*sin(pos[2])*sin(pos[2])*vel[3]*vel[3],

                sin(pos[2])*cos(pos[2])*vel[3]*vel[3]               // theta component
                - (2/pos[1])*vel[1]*vel[2],

                -(2/pos[1])*vel[1]*vel[3]                           // phi component
                - (2/tan(pos[2]))*vel[2]*vel[3]
            };
            break;

        case Metric::CartesianMinkowskiMetric:

            // All christoffel symbols are zero for the Minkowski metric in (t,x,y,z) coords
            acceleration = {
                0,
                0,
                0,
                0
            };
            break;
            
        case Metric::SchwarzschildMetric:

            // Acceleration is computed from the geodesic equation
            acceleration = {

                (-2*mass/(pos[1]*pos[1]))*(1/(1 - 2*mass/pos[1]))*vel[1]*vel[0],            // time component

                (-mass/(pos[1]*pos[1]))*(1 - 2*mass/pos[1])*vel[0]*vel[0]                   // this whole thing is
                    + (mass/(pos[1]*pos[1]))*(1/(1 - 2*mass/pos[1]))*vel[1]*vel[1]          // the radial component
                    + pos[1]*(1 - 2*mass/pos[1])*vel[2]*vel[2]
                    + pos[1]*sin(pos[2])*sin(pos[2])*(1 - 2*mass/pos[1])*vel[3]*vel[3],

                (-2/pos[1])*vel[2]*vel[1] + sin(pos[2])*cos(pos[2])*vel[3]*vel[3],          // theta component

                (-2/pos[1])*vel[3]*vel[1] - (2/tan(pos[2]))*vel[3]*vel[2]                   // phi component

            };
            break;

        case Metric::CartesianIsotropicSchwarzschildMetric:

            // These variables are part of the christoffel symbols
            rho_s = 2*mass/4;
            rho = sqrt(pos[1]*pos[1] + pos[2]*pos[2] + pos[3]*pos[3]);

            // list of christoffel symbols (from https://arxiv.org/pdf/0904.4184)
            double gamma_x_tt, gamma_y_tt, gamma_z_tt;
            gamma_x_tt = 2*pow(rho,3)*rho_s*(rho - rho_s)*pos[1]/pow(rho + rho_s, 7);
            gamma_y_tt = 2*pow(rho,3)*rho_s*(rho - rho_s)*pos[2]/pow(rho + rho_s, 7);
            gamma_z_tt = 2*pow(rho,3)*rho_s*(rho - rho_s)*pos[3]/pow(rho + rho_s, 7);

            double gamma_t_tx, gamma_t_ty, gamma_t_tz;
            gamma_t_tx = 2*rho_s*pos[1]/( pow(rho,3)*(1 - pow(rho_s/rho,2)) );
            gamma_t_ty = 2*rho_s*pos[2]/( pow(rho,3)*(1 - pow(rho_s/rho,2)) );
            gamma_t_tz = 2*rho_s*pos[3]/( pow(rho,3)*(1 - pow(rho_s/rho,2)) );

            double gamma_x_xx, gamma_y_xy, gamma_z_xz, gamma_x_yy, gamma_x_zz;
            gamma_x_xx = -2*(rho_s/pow(rho,3)) * (pos[1]/(1 + rho_s/rho));
            gamma_y_xy = -2*(rho_s/pow(rho,3)) * (pos[1]/(1 + rho_s/rho));
            gamma_z_xz = -2*(rho_s/pow(rho,3)) * (pos[1]/(1 + rho_s/rho));
            gamma_x_yy = 2*(rho_s/pow(rho,3)) * (pos[1]/(1 + rho_s/rho));
            gamma_x_zz = 2*(rho_s/pow(rho,3)) * (pos[1]/(1 + rho_s/rho));

            double gamma_y_xx, gamma_x_xy, gamma_y_yy, gamma_z_yz, gamma_y_zz;
            gamma_y_xx = 2*(rho_s/pow(rho,3)) * (pos[2]/(1 + rho_s/rho));
            gamma_x_xy = -2*(rho_s/pow(rho,3)) * (pos[2]/(1 + rho_s/rho));
            gamma_y_yy = -2*(rho_s/pow(rho,3)) * (pos[2]/(1 + rho_s/rho));
            gamma_z_yz = -2*(rho_s/pow(rho,3)) * (pos[2]/(1 + rho_s/rho));
            gamma_y_zz = 2*(rho_s/pow(rho,3)) * (pos[2]/(1 + rho_s/rho));

            double gamma_z_xx, gamma_x_xz, gamma_z_yy, gamma_y_yz, gamma_z_zz;
            gamma_z_xx = 2*(rho_s/pow(rho,3)) * (pos[3]/(1 + rho_s/rho));
            gamma_x_xz = -2*(rho_s/pow(rho,3)) * (pos[3]/(1 + rho_s/rho));
            gamma_z_yy = 2*(rho_s/pow(rho,3)) * (pos[3]/(1 + rho_s/rho));
            gamma_y_yz = -2*(rho_s/pow(rho,3)) * (pos[3]/(1 + rho_s/rho));
            gamma_z_zz = -2*(rho_s/pow(rho,3)) * (pos[3]/(1 + rho_s/rho));

            // Acceleration is computed from the geodesic equation
            acceleration = {

                // time component
                - 2*gamma_t_tx*vel[0]*vel[1]
                    - 2*gamma_t_ty*vel[0]*vel[2]
                    - 2*gamma_t_tz*vel[0]*vel[3],

                // x component
                -gamma_x_tt*pow(vel[0],2)
                    - gamma_x_xx*pow(vel[1],2)
                    - gamma_x_yy*pow(vel[2],2)
                    - gamma_x_zz*pow(vel[3],2)
                    - 2*gamma_x_xy*vel[1]*vel[2]
                    - 2*gamma_x_xz*vel[1]*vel[3],

                // y component
                -gamma_y_tt*pow(vel[0],2)
                    - gamma_y_xx*pow(vel[1],2)
                    - gamma_y_yy*pow(vel[2],2)
                    - gamma_y_zz*pow(vel[3],2)
                    - 2*gamma_y_xy*vel[1]*vel[2]
                    - 2*gamma_y_yz*vel[2]*vel[3],
                
                // z component
                -gamma_z_tt*pow(vel[0],2)
                    - gamma_z_xx*pow(vel[1],2)
                    - gamma_z_yy*pow(vel[2],2)
                    - gamma_z_zz*pow(vel[3],2)
                    - 2*gamma_z_xz*vel[1]*vel[3]
                    - 2*gamma_z_yz*vel[2]*vel[3],

            };
            break;

    }
    return acceleration;

}

CudaMetric Metric_to_CudaMetric(Metric metric) {
    return CudaMetric(metric.get_type(), metric.get_params());
}