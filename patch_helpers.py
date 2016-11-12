

def read_detections(filename):
    lines = [x.split() for x in open(filename).readlines()]
    return [[l[0]] + [int(x) for x in l[1:5]] for l in lines]
