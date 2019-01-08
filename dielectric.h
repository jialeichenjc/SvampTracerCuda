#ifndef DIELECTRICH
#define DIELECTRICH

#include "material.h"
#include "Utility.h"

class dielectric : public material {
public:
    dielectric(float ri) : ref_idx(ri) {}
    virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx; // ray shoots from higher refraction index material into air
            // REVISIT: why multiply by ref_idx here??
            cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / ref_idx; // ray shoots from air to higher ref index material
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }

        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, ref_idx);
        }
        // total internal reflection
        else {
            scattered = ray(rec.p, reflected); // REVISIT: unnecessary line here
            reflect_prob = 1.0;
        }

        // randomly choosing between reflection and refraction, if refraction is possible
        if (util::gen_rand() < reflect_prob) {
            scattered = ray(rec.p, reflected);
        }
        else {
            scattered = ray(rec.p, refracted);
        }

        return true;    
    }

    float ref_idx;
};

#endif // !DIELECTRICH

