from intervaltree import IntervalTree

_huesByDegrees = IntervalTree()

# NOTE: Ranges are inclusive of lower but exclusive of upper

_huesByDegrees.addi(0, 15, 'red')
_huesByDegrees.addi(15, 40, 'orange')
_huesByDegrees.addi(40, 60, 'yellow')
_huesByDegrees.addi(60, 150, 'green')
_huesByDegrees.addi(150, 240, 'blue')
_huesByDegrees.addi(240, 300, 'purple')
_huesByDegrees.addi(300, 335, 'pink')
_huesByDegrees.addi(335, 361, 'red')


def get_color_name_from_hsl(h, s, l):
    # If saturation is very low, then this color is gray
    if s < 0.1:
        return 'gray'

    # If lightness is very low, this color is black
    if l < 0.15:
        return 'black'

    # If lightness is very high, this color is white:
    if l > 0.95:
        return 'white'

    if _huesByDegrees.overlaps(h):
        color_interval_set = _huesByDegrees[h]
        color_interval = color_interval_set.pop()
        color = color_interval.data

        # Special case for brown based on saturation
        if color == 'orange' and s < 0.50:
            color = 'brown'

        # If lightness is low, prepend 'dark'
        if l < 0.35:
            color = "dark {}".format(color)

        if l > 0.65:
            color = "light {}".format(color)

        return color

    return None



