#ifndef MATERIALH
#define MATERIALH

#include "ray.h"
#include "hitable.h"
#include "Utility.h"

class material {
public:
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const = 0;

    __device__ bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted) const {
        vec3 uv = unit_vector(v);
        float dt = dot(uv, n);
        float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
        if (discriminant > 0) {
            refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
            return true;
        }
        else {
            return false;
        }
    }
  
    __device__ float schlick(float cosine, float ref_idx) const {
        float r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0)*pow((1 - cosine), 5);
    }

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
   
    // rejection method
    __device__ vec3 random_in_unit_sphere(curandState *local_rand_state) const {
        vec3 p;
        do {
            // generate a vector p whose x, y, z are all in (-1, 1)
            p = 2.0*RANDVEC3 - vec3(1, 1, 1);
        } while (p.squared_length() >= 1.0);
        return p;
    }

    __device__ vec3 reflect(const vec3& v, const vec3 &n) const {
        return v - 2 * dot(v, n) * n;
    }

};


#endif // !MATERIALH
