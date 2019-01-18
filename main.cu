
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

#include "Utility.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "metal.h"
#include "lambertian.h"
#include "dielectric.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray &r, hitable ** world, curandState * local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);

    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0, 0, 0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            vec3 unit_direction = unit_vector(r.direction());
            float t = 0.5 * (unit_direction.y() + 1.0); // 0 <= t <= 1
            // lerp between white and blue
            // lerp := (1-t) * start_value + t * end_value;
            vec3 c = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0);
}

__global__ void rand_init(curandState * rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState * rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// __global__ called from host, executed on device
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera ** cam, hitable ** world, curandState * rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), new lambertian(vec3(0.5, 0.5, 0.5)));

    }
}

//hitable * random_scene() {
//    int n = 500;
//    hitable ** list = new hitable*[n + 1];
//    list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
//
//    int i = 1;
//    for (int a = -11; a < 11; a++) {
//        for (int b = -11; b < 11; b++) {
//            float choose_mat = util::gen_rand();
//            vec3 center(a + 0.9 * util::gen_rand(), 0.2, b + 0.9 * util::gen_rand());
//            if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
//                if (choose_mat < 0.8) { // diffuse
//                    list[i++] = new sphere(center, 0.2,
//                        new lambertian(
//                            vec3(util::gen_rand() * util::gen_rand(), util::gen_rand() * util::gen_rand(), util::gen_rand() * util::gen_rand())));
//                }
//                else if (choose_mat < 0.95) { // metal
//                    list[i++] = new sphere(center, 0.2,
//                        new metal(
//                            vec3(0.5*(1 + util::gen_rand()), 0.5*(1 + util::gen_rand()), 0.5*(1 + util::gen_rand())), 0.5*util::gen_rand()));
//                }
//                else {
//                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
//                }
//            }
//        }
//    }
//
//    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
//    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
//    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
//
//    return new hitable_list(list, i);
//}

int main()
{
    std::ofstream outputFile;
    outputFile.open("output.ppm");

    int nx = 1200;
    int ny = 800;
    int ns = 10;

    srand(time(NULL));

    outputFile << "P3\n" << nx << " " << ny << "\n255\n";

    float R = cos(M_PI / 4);
    //hitable *list[5];
    //list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
    //list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
    //list[2] = new sphere(vec3( 1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 1.0));
    //list[3] = new sphere(vec3(-1, 0, -1), 0.5, new dielectric(1.5));
    //list[4] = new sphere(vec3(-1, 0, -1), -0.45, new dielectric(1.5)); // hollow glass sphere, with reverse normals

    hitable *world = random_scene();
    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;
    //  camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect)
    camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus);
    /*
    for (int i = 0; i < 50; i++) {
    std::cout << util::gen_rand() << std::endl;
    }*/

    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + (double)rand() / RAND_MAX) / float(nx);
                float v = float(j + (double)rand() / RAND_MAX) / float(ny);
                ray r = cam.get_ray(u, v);
                col += color(r, world, 0);
            }
            // averaging over the samples for anti-aliasing
            col /= float(ns);
            col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
            int ir = int(255.99 * col[0]);
            int ig = int(255.99 * col[1]);
            int ib = int(255.99 * col[2]);
            outputFile << ir << " " << ig << " " << ib << "\n";
        }
    }

    outputFile.close();
    //delete[] list;
    delete world;
    return 0;
}

