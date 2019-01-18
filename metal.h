#ifndef METALH
#define METALH

#include "material.h"

class metal : public material {
public:
    __device__ metal(const vec3 &a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0); // discard rays that have been scattered below the surface/material
    }

    vec3 albedo;
    float fuzz;
};

#endif // !METALH

