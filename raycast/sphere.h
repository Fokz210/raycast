#pragma once
#include <cuda_runtime.h>
#include <cmath>

#include "ray.h"

#define SKY_COLOR vector3f (189.f / 255.f, 214.f / 255.f, 1.f)

class sphere
{
public:
	sphere (vector3f const & c, float const & r, vector3f col) :
		center (c),
		radius (r),
		color (col)
	{
	}

	struct intersection
	{
		bool happened;
		vector3f color;

		vector3f pos;
		vector3f norm;
	};

	__device__ intersection sphere::ray_intersect (ray const & r, float & dist) const noexcept
	{
		vector3f center_distance = center - r.pos;
		float proj_distance = dot (center_distance, r.dir);

		float sq_dist = dot (center_distance, center_distance) - proj_distance * proj_distance;
		if (sq_dist > radius * radius)
			return intersection{ false, SKY_COLOR, vector3f (), vector3f () };

		float intersection_dist = sqrtf (radius * radius - sq_dist);

		dist = proj_distance - intersection_dist;
		if (dist < 0) dist = proj_distance + intersection_dist;
		if (dist < 0)
			return intersection{ false, SKY_COLOR, vector3f (), vector3f () };
		else
			return intersection{ true, color, r.pos + r.dir * dist,  (r.pos + r.dir * dist - center) / radius };
	}

	vector3f center;
	float radius;
	vector3f color;
};

