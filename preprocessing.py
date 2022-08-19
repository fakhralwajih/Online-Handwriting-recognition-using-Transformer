 #@title Preprocessing 
import math
import numpy as np
from scipy.stats import linregress
from scipy.special import binom

"""
All normalization steps are based on the following two papers:
Liwicki, M. ; Bunke, H.:  HMM-based on-line recognition  of  handwritten  whiteboard  notes.
In: Tenth  International Workshop on Frontiers in Handwriting Recognition,
Suvisoft, 2006
Jaeger, S. ; Manke, S. ; Waibel, A.: Npen++: An On-Line Handwriting Recognition System.
In: 7th International Workshop on Frontiers in Handwriting Recognition,
2000, S.249–260
"""


def preprocess_handwriting(ink, args):
    """
    Applies given normalization steps in args to ink of points in ink.
    Valid normalizations are "flip", "slope", "origin", "resample", "slant", "height",
    "smooth" and "delayed". Note that with application of "delayed" there will be
    two objects returned, the ink and the list of delayed strokes.
    The object that "ink" points to WILL BE CHANGED!
    """
    if "slope" in args:
        ink = correct_slope(ink)
    if "origin" in args:
        ink = move_to_origin(ink)
    #Added 
    if "flip_h" in args:
        ink = flip_horizontally(ink)
    if "slant" in args:
        ink = correct_slant(ink)
    if "height" in args:
        ink = normalize_height(ink)
    if "resample" in args:
        ink = resampling(ink)
    if "smooth" in args:
        ink = smoothing(ink)
    return ink

def flip_horizontally(ink):
    #Flip
    ink[:,0]=(ink[:,0]-ink[:,0].max())*-1
    return ink
def move_to_origin(ink):
    """
    Move ink so that the lower left corner
    of its bounding box is the origin afterwards.
    """
    #print('origin')
    min_x = min(ink[:, 0])
    min_y = min(ink[:, 1])
    return ink - [min_x, min_y, 0]


def flip_vertically(ink):
    """
    Rotates ink by 180 degrees.
    """
    #print('flip')
    max_y = max(ink[:, 1])
    return np.array([[x, max_y - y, p] for [x, y, p] in ink])


def correct_slope(ink):
    """
    Rotates ink so that the regression line through
    all points is the horizontal line afterwards.
    """
    #print('slope')
    [slope, intercept, _, _, _] = linregress(ink[:, :2])
    alpha = math.atan(-slope)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    min_x = min(ink[:, 0])
    min_y = min(ink[:, 1])
    rot_x = lambda x, y: min_x + cos_alpha * (x - min_x) - sin_alpha * (y - min_y)
    rot_y = lambda x, y: min_y + sin_alpha * (x - min_x) + cos_alpha * (y - min_y)
    new_ink = np.array([[rot_x(x, y), rot_y(x, y), p] for [x, y, p] in ink])
    new_min_x = min(new_ink[:, 0])
    new_min_y = min(new_ink[:, 1])
    return new_ink - [new_min_x, new_min_y, 0]


def correct_slant(ink):
    """
    Removes the most dominant slant-angle from the ink.
    """
    #print('slant')
    last_point = ink[0]
    angles = []
    for cur_point in ink[1:]:
        # check for penup
        if last_point[2] == 1:
            # don't measure angles for "invisible" lines
            last_point = cur_point
            continue
        if (cur_point[0] - last_point[0]) == 0:
            angles.append(90)
        else:
            angle = math.atan((cur_point[1] - last_point[1]) / float(cur_point[0] - last_point[0])) * 180 / math.pi
            angles.append(int(angle))
        last_point = cur_point
    # print("found {} angles for {} points".format(len(angles), len(ink)))
    # we move angles from [-90,90] to [0, 180] for calculations
    angles = np.array(angles) + 90
    bins = np.bincount(angles, minlength=181)
    # weighting all angles with discrete standard gaussian distribution
    weights = [binom(181, k)/181.0 for k in range (1, 182)]
    weights /= sum(weights)
    bins = bins.astype(float) * weights
    # smoothing entries with neighbours, first and last points remain unchanged
    gauss = lambda p, c, n: 0.25 * p + 0.5 * c + 0.25 * n
    smoothed = [bins[0]] + [gauss(bins[i-1], bins[i], bins[i+1]) for i in range(len(bins)-1)] + [bins[len(bins)-1]]
    # reverse interval shift
    slant = np.argmax(smoothed) - 90
    # print("slant is {}".format(slant))
    # print(len(smoothed))
    min_x = min(ink[:, 0])
    min_y = min(ink[:, 1])
    rotate = lambda x, y: min_x + (x - min_x) - math.tan(slant * math.pi / 180) * (y - min_y)
    return np.array([[rotate(x, y), y, p] for [x, y, p] in ink])


def resampling(ink, step_size=10):
    """
    Replaces given ink by a recalculated sequence of equidistant points.
    """
    #print('resampling')
    t = []
    t.append(ink[0, :])
    i = 0
    length = 0
    current_length = 0
    old_length = 0
    curr, last = 0, None
    len_ink = ink.shape[0]
    while i < len_ink:
        current_length += step_size
        while length <= current_length and i < len_ink:
            i += 1
            if i < len_ink:
                last = curr
                curr = i
                old_length = length
                length += math.sqrt((ink[curr, 0] - ink[last, 0])**2) + math.sqrt((ink[curr, 1] - ink[last, 1])**2)
        if i < len_ink:
            c = (current_length - old_length) / float(length-old_length)
            x = ink[last, 0] + (ink[curr, 0] - ink[last, 0]) * c
            y = ink[last, 1] + (ink[curr, 1] - ink[last, 1]) * c
            p = ink[last, 2]
            t.append([x, y, p])
    t.append(ink[-1, :])
    #np.savetxt('resample.txt', np.array(t))
    return np.array(t)


def normalize_height(ink, new_height=120):
    """
    Returns scaled ink whose height will be new_height.
    TODO: try to scale core height instead
    """
    #print('normalize')
    min_y = min(ink[:, 1])
    max_y = max(ink[:, 1])
    old_height = max_y - min_y
    scale_factor = new_height / float(old_height)
    ink[:, :2] *= scale_factor
    return ink


def smoothing(ink):
    """
    Applies gaussian smoothing to the ink with a (0.25, 0.5, 0.25) sliding
    window. Smoothing point p(t) uses un-smoothed points p(t-1) and p(t+1).
    """
    #print('smooth')
    s = lambda p, c, n: 0.25 * p + 0.5 * c + 0.25 * n
    smoothed = np.array([s(ink[i-1], ink[i], ink[i+1]) for i in range(1, ink.shape[0]-1)])
    # the code above also changes penups, so we just copy them again
    smoothed[:, 2] = ink[1:-1, 2]
    # we deleted the unsmoothed first and last points,
    # so the last penup needs to be moved to the second to last point
    smoothed[-1, 2] = 1
    #np.savetxt('smooth.txt', smoothed)
    return smoothed


def remove_delayed_strokes(ink):
    """
    Removes points of delayed strokes (segments between two penups)
    from the ink. Removal if right edge of stroke's bounding box
    is to the left of the right edge of the last non-delayed stroke.
    """
    #print('delayed')
    stroke_endpoints = np.where(ink[:, 2] == 1)[0]
    # first stroke is by convention never delayed
    begin = stroke_endpoints[0] + 1
    new_ink = []
    new_ink.extend(ink[:begin, :])
    delayed = []
    # delayed strokes must begin and end left of the current orientation point
    orientation_point = ink[begin-1, :2]
    for end in stroke_endpoints[1:]:
        stroke = ink[begin:end+1, :]
        max_x = max(stroke[:, 0])
        begin = end + 1
        if max_x >= orientation_point[0]:
            new_ink.extend(stroke)
            orientation_point = ink[begin-1, :2]
        else:
            delayed.append(stroke)
    return np.array(new_ink), np.array(delayed)