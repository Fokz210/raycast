#pragma once
#include <device_launch_parameters.h>
#include "ray.h"

class camera
{
public:
	camera (float _fov, float _ratio, vector3f _pos, vector3f _up, vector3f _at) :
		fov (_fov),
		ratio (_ratio),
		pos (_pos),
		up (_up),
		at (_at)
	{
	}

	__device__ ray cast (float const u, float const v) const noexcept
	{
		vector3f dir = normalize (at - pos);
		vector3f right = normalize (cross (dir, up));
		vector3f real_up = cross (right, dir);

		return ray (pos, normalize (right * (u * ratio * fov) + real_up * (v * fov) + dir));
	}

	float fov;
	float ratio;
	vector3f pos;
	vector3f up;
	vector3f at;
};

