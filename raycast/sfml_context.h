#pragma once

#include <SFML/Graphics.hpp>
#include "context_interface.h"

struct RGBA8
{
	sf::Uint8 r, g, b, a;
};

class sfml_context :
	public context_interface<RGBA8>
{
public:
	sfml_context (unsigned width, unsigned height);
	virtual ~sfml_context () override;

	virtual unsigned width () const noexcept override;
	virtual unsigned height () const noexcept override;

	virtual color * operator [] (int y) override;
	virtual color const *  operator [] (int y) const override;

	virtual void update () noexcept override;
	virtual void clear () noexcept override;

	color * memory () noexcept;
	bool is_open () const noexcept;

protected:
	virtual void handle_events () noexcept;

	color * pixels;
	sf::RenderWindow window;
	sf::Texture render_texture;
	unsigned canvas_width, canvas_height;
};

