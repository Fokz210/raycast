#include "sfml_context.h"

sfml_context::sfml_context (unsigned width, unsigned height) :
	pixels (new color[width * height]),
	window (sf::VideoMode (width, height), "raycast"),
	render_texture(),
	canvas_width (width),
	canvas_height (height)
{
	render_texture.create (width, height);
}

sfml_context::~sfml_context ()
{
	delete[] pixels ;
}

inline unsigned sfml_context::width () const noexcept
{
	return canvas_width;
}

inline unsigned sfml_context::height () const noexcept
{
	return canvas_height;
}

inline sfml_context::color * sfml_context::operator[](int y)
{
	return pixels + canvas_width * (canvas_height - y - 1);
}

inline sfml_context::color const * sfml_context::operator[](int y) const
{
	return pixels + canvas_width * (canvas_height - y - 1);
}

inline void sfml_context::update () noexcept
{
	handle_events ();

	render_texture.update (reinterpret_cast<sf::Uint8 *>(pixels));
	window.draw (sf::Sprite (render_texture));
	window.display ();
}

void sfml_context::clear () noexcept
{
	window.clear ();
}

bool sfml_context::is_open () const noexcept
{
	return window.isOpen();
}

sfml_context::color * sfml_context::memory () noexcept
{
	return pixels;
}

inline void sfml_context::handle_events () noexcept
{
	sf::Event event;

	while (window.pollEvent (event))
	{
		switch (event.type)
		{
		default:
			break;

		case sf::Event::Closed:
			window.close ();
			break;
		}
	}
}


