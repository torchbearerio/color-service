from intervaltree import IntervalTree

_huesByDegrees = IntervalTree()

# NOTE: Ranges are inclusive of lower but exclusive of upper

_huesByDegrees.addi(0, 31, 'red')
_huesByDegrees.addi(31, 46, 'orange')
_huesByDegrees.addi(46, 76, 'yellow')
_huesByDegrees.addi(76, 106, 'light green')
_huesByDegrees.addi(106, 136, 'green')
_huesByDegrees.addi(136, 166, 'dark green')
_huesByDegrees.addi(166, 196, 'light blue')
_huesByDegrees.addi(196, 226, 'blue')
_huesByDegrees.addi(226, 256, 'dark blue')
_huesByDegrees.addi(256, 286, 'dark purple')
_huesByDegrees.addi(286, 316, 'purple')
_huesByDegrees.addi(316, 346, 'pink')
_huesByDegrees.addi(346, 360, 'red')


def get_color_name_from_hue(hue):
    if _huesByDegrees.overlaps(hue):
        color_interval_set = _huesByDegrees[hue]
        color_interval = color_interval_set.pop()
        return color_interval.data

    return None



