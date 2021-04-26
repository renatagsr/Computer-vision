def rect_to_bb(rect):
    #Take a bounding predicted by dlib and convert it to ther format
    #(x, y, w, h) as we would normally do with openCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    #Return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype='int'):
    #Initialize the list of (x, y)-coordinates
    coords = np.zeros((68,2), dtype=dtype)
    