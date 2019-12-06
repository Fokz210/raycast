#pragma once

template <class colordesc>
class context_interface
{
public:
	using color = colordesc;

	context_interface ()
	{

	}

	virtual ~context_interface ()
	{

	}

	virtual unsigned width () const noexcept = 0;
	virtual unsigned height () const noexcept = 0;

	virtual color * operator [] (int y) = 0;
	virtual color const * operator [] (int y) const = 0;

	virtual void clear () noexcept = 0;
	virtual void update () noexcept = 0;
};


