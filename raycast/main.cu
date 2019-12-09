////////////////////////////////
//-----------CUDA-------------//
////////////////////////////////
#include <cuda_runtime.h>             
#include <device_launch_parameters.h>
#include <curand.h>

////////////////////////////////
//-----------SFML-------------//
////////////////////////////////
#include <SFML/Graphics.hpp>

////////////////////////////////
//-----------STD--------------//
////////////////////////////////
#include <iostream>
#include <numeric>

////////////////////////////////
//-----------LOCAL------------//
////////////////////////////////
#include "sfml_context.h"
#include "sphere.h"
#include "camera.h"

//==============================================================

__device__ vector3f ray_cast (ray const & r, sphere const & sph) noexcept;
__global__ void render (sfml_context::color * const colorbuffer, int const width, int const height, sphere * sphs, int const sphc, camera const cam);

int const threads = 32;

int main ()
{
	sfml_context window (1920, 1080);

	window.clear ();

	sfml_context::color * device_mem;

	cudaMalloc (&device_mem, window.width () * window.height () * sizeof (sfml_context::color));

	float phi = 0.f;
	float theta = 0.4f;
	
	sphere sphs[2] = { sphere (vector3f (0.f, 0.f, 0.3f), 0.1f, vector3f (1.f, 1.f, 1.f)),
					   sphere (vector3f (0.f, 0.f, -1000.f), 1000.f, vector3f (0.3f, 0.7f, 0.3f)) };

	sphere * sphs_vptr = nullptr;
	cudaMalloc (&sphs_vptr, sizeof (sphere) * 2);
	cudaMemcpy (sphs_vptr, sphs, sizeof (sphere) * 2, cudaMemcpyHostToDevice);

	while (window.is_open())
	{
		float const s = 2e-3;

		phi = static_cast <float> (sf::Mouse::getPosition ().x) * s;
		theta = static_cast <float> (sf::Mouse::getPosition ().y) * s;

		float const dist = 2.f;

		camera const cam
		(
			1.f,
			static_cast <float> (window.width ()) / window.height (),
			vector3f (std::cos (phi) * std::cos (theta), std::sin (phi) * std::cos (theta), std::sin (theta)) * dist,
			vector3f (0.f, 0.f, 1.f),
			vector3f (0.f, 0.f, 0.3f)
		);

		render <<< window.width() * window.height() / threads, threads >>> (device_mem, window.width (), window.height (), sphs_vptr, 2, cam);
		cudaMemcpy (window.memory (), device_mem, window.width () * window.height () * sizeof (sfml_context::color), cudaMemcpyDeviceToHost);
		window.update ();
	}
	
	cudaFree (device_mem);
	return 0;
}

__device__ vector3f ray_cast (ray const & r, sphere * sphs, int sphere_count) noexcept
{
	float dist = 200000;

	sphere::intersection in{ false, SKY_COLOR, vector3f (), vector3f () };

	for (int i = 0; i < sphere_count; i++)
	{
		sphere::intersection new_intersection = sphs[i].ray_intersect (r, dist);

		if (new_intersection.happened && (in.happened && dot (in.pos - r.pos, in.pos - r.pos) > dot (new_intersection.pos - r.pos, new_intersection.pos - r.pos) || !in.happened))
		{
			in = new_intersection;
		}
	}

	return in.color;
}

__global__ void render (sfml_context::color * const colorbuffer, int const width, int const height, sphere * sphs, int const sphc, camera const cam)
{
	int const i = blockIdx.x * blockDim.x + threadIdx.x;
	float2 const p = { i % width, i / width };
	int const j = p.x + width * (height - p.y - 1);
	float2 const f = { (2.f * p.x) / width - 1.f, (2.f * p.y) / height - 1.f };

	vector3f color = ray_cast (cam.cast (f.x, f.y), sphs, sphc);

	colorbuffer[j] = RGBA8
	{
		static_cast <sf::Uint8> (255 * color.x),
		static_cast <sf::Uint8> (255 * color.y),
		static_cast <sf::Uint8> (255 * color.z),
		255
	};
}

__device__ vector3f hemi_vector3f (vector3f const & norm, float const rand1, float const rand2)
{
	vector3f const x0 = cross (norm, vector3f (0.f, 0.f, 1.f));
	vector3f x;
	if (dot (x0, x0) < 1e-4)
		x = normalize (cross (norm, vector3f (0.f, 1.f, 0.f)));
	else
		x = normalize (x0);
	vector3f const y = cross (norm, x);

	float const cos_theta = sqrtf (rand1);
	float const sin_theta = sqrtf (1.f - rand1);
	float const phi = rand2 * 3.14159265f * 2.f;
	return x * cos_theta * cos (phi) + y * cos_theta * sin (phi) + norm * sin_theta;
}
