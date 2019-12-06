#pragma once
#include "vector3.hpp"

class ray
{
public:
	__device__ ray (vector3f position, vector3f direction) :
		pos (position),
		dir (direction)
	{
	}


	vector3f pos, dir;
};

