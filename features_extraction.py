#@title Feature Extraction 
import math
import numpy as np

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def calculate_feature_vector_sequence(ink, args, delayed_strokes=None):
    """
    Calculates all features named in args for each point in inkectory ink. Valid features are "dir", "curv",
    "penup", "hat", "vic_aspect", "vic_curl", "vic_line", "vic_slope" and "bitmap". Note that calculating the hat
    feature requires precalculated delayed strokes.
    """
    ma=moving_average(ink[:,0],20)
    ma=(ink[:,0]-ma).reshape(ma.shape[0],1)

    #print(ma.shape)
    #print(ink.shape)
    ink=np.hstack((ink,ma))

    #print(ink)

    return np.array([__calculate_feature_vector(ink, p, args, delayed_strokes) for p in range(len(ink))],dtype = np.float32)
def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    data1=data_set[:periods-1]
    data2=np.convolve(data_set, weights, mode='valid')
    if periods > len(data_set):
        return data1

    return np.concatenate([data1,data2])

def __calculate_feature_vector(ink, point_index, args, delayed_strokes=None):
    # calculating number of features is not pretty because dir, curv and bitmap are actually more than one feature...
    num_features = len(args)
    if "dir" in args:
        num_features += 1
    if "curv" in args:
        num_features += 1
    if "bitmap" in args:
        num_features += 8
    if len(ink) < 5:
        return np.zeros(num_features)
    feat_vec = []
    if "x_cor" in args:
        feat_vec.extend(np.array([float(ink[point_index, 3])]))
    if "y_cor" in args:
        feat_vec.extend(np.array([float(ink[point_index, 1])]))
    if "penup" in args:
        feat_vec.extend(np.array([float(ink[point_index, 2])]))
    if "dir" in args:
        feat_vec.extend(__writing_direction(ink, point_index))
    if "curv" in args:
        feat_vec.extend(__curvature(ink, point_index))
    if "hat" in args:
        feat_vec.append(__hat(ink, point_index, delayed_strokes))
    if "vic_aspect" in args:
        feat_vec.append(__vicinity_aspect(ink, point_index))
    if "vic_curl" in args:
        feat_vec.append(__vicinity_curliness(ink, point_index))
    if "vic_line" in args:
        feat_vec.append(__vicinity_lineness(ink, point_index))
    if "vic_slope" in args:
        feat_vec.append(__vicinity_slope(ink, point_index))
    if "bitmap" in args:
        feat_vec.extend(__context_bitmap(ink, point_index))
    return np.array(normalize(feat_vec))

def __x_cor(ink, point_idx):
    
    #print(float(ink[point_idx, 0]))
    return float(ink[point_idx, 0])

def __y_cor(ink, point_idx):
    return float(ink[point_idx, 1])

def __writing_direction(ink, point_idx):
    if point_idx == 0:
        # first point in inkectory
        d = ink[point_idx, :2] - ink[point_idx + 1, :2]
    elif point_idx == len(ink) - 1:
        # last point in inkectory
        d = ink[point_idx-1, :2] - ink[point_idx, :2]
    else:
        d = ink[point_idx-1, :2] - ink[point_idx + 1, :2]
    ds = np.linalg.norm(d)
    return d / ds if ds != 0 else [0.0, 0.0]


def __curvature(ink, point_idx):
    if point_idx == 0:
        # first point in inkectory
        [cos_prev, sin_prev] = __writing_direction(ink, point_idx)
        [cos_next, sin_next] = __writing_direction(ink, point_idx + 1)
    elif point_idx == len(ink) - 1:
        # last point in inkectory
        [cos_prev, sin_prev] = __writing_direction(ink, point_idx - 1)
        [cos_next, sin_next] = __writing_direction(ink, point_idx)
    else:
        [cos_prev, sin_prev] = __writing_direction(ink, point_idx - 1)
        [cos_next, sin_next] = __writing_direction(ink, point_idx + 1)
    curv_cos = cos_prev * cos_next + sin_prev * sin_next
    curv_sin = cos_prev * sin_next - sin_prev * cos_next
    return [curv_cos, curv_sin]


def __is_penup(ink, point_idx):
    return float(ink[point_idx, 2])


def __hat(ink, point_idx, delayed_strokes):
    if delayed_strokes is None:
        return 0.0
    for stroke in delayed_strokes:
        minx = min(stroke[:, 0])
        maxx = max(stroke[:, 0])
        miny = min(stroke[:, 1])
        # we check for each stroke if the point is under (smaller y coord) this stroke
        if minx <= ink[point_idx, 0] <= maxx and ink[point_idx, 1] < miny:
            return 1.0
    return 0.0


def __vicinity_aspect(ink, point_idx):
    # filter out cases where there is not enough points to either side
    if point_idx < 2 or point_idx > len(ink) - 3:
        return 0.0
    vicinity = ink[point_idx-2:point_idx+3, :2]
    dx = max(vicinity[:, 0]) - min(vicinity[:, 0])
    dy = max(vicinity[:, 1]) - min(vicinity[:, 1])
    if dx + dy == 0:
        return 0.0
    return 2 * float(dy) / (dx + dy) - 1


def __vicinity_curliness(ink, point_idx):
    # filter out cases where there is not enough points to either side
    if point_idx < 2 or point_idx > len(ink) - 3:
        return 0.0
    vicinity = ink[point_idx-2:point_idx+3, :2]
    dx = max(vicinity[:, 0]) - min(vicinity[:, 0])
    dy = max(vicinity[:, 1]) - min(vicinity[:, 1])
    segment_length = sum([np.linalg.norm(vicinity[i]-vicinity[i+1]) for i in range(len(vicinity)-2)])
    if max(dx, dy) == 0:
        return 0.0
    return float(segment_length) / max(dx, dy) - 2


def __vicinity_lineness(ink, point_idx):
    # filter out cases where there is not enough points to either side
    if point_idx < 2 or point_idx > len(ink) - 3:
        return 0.0
    v = ink[point_idx-2:point_idx+3, :2]
    first = 0
    last = len(v) - 1
    x1 = v[first, 0]
    x2 = v[last, 0]
    y1 = v[first, 1]
    y2 = v[last, 1]
    diag_line_length = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if diag_line_length == 0:
        # first and last point have same coordinates, so we return average squared distance to that point
        return sum([math.sqrt((y2 - y)**2 + (x2 - x)**2)**2 for [x, y] in v]) / len(v)
    dist_to_line = lambda x, y: abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return sum([dist_to_line(x, y)**2 for [x, y] in v]) / len(v)


def __vicinity_slope(ink, point_idx):
    # filter out cases where there is not enough points to either side
    if point_idx < 2 or point_idx > len(ink) - 3:
        return 0.0
    vicinity = ink[point_idx-2:point_idx+2, :2]
    first = 0
    last = len(vicinity) - 1
    xdiff = vicinity[last, 0] - vicinity[first, 0]
    if xdiff != 0:
        slope = (vicinity[last, 1] - vicinity[first, 1]) / xdiff
    else:
        slope = 0
    return math.cos(math.atan(slope))


def __context_bitmap(ink, point_idx, bin_size=10):
    # the current point lies in the center of the bitmap and we use a 3x3 grid around that point
    window_origin_x = ink[point_idx][0] - 3 * bin_size / 2
    window_origin_y = ink[point_idx][1] - 3 * bin_size / 2
    bitmap = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
    num_points = 0
    for p in ink:
        bin_x = int((p[0] - window_origin_x) / bin_size)
        bin_y = int((p[1] - window_origin_y) / bin_size)
        if 0 <= bin_x <= 2 and 0 <= bin_y <= 2:
            bitmap[bin_y][bin_x] += 1
            num_points += 1
    return normalize(np.array([p / float(num_points) for bin in bitmap for p in bin]))
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm