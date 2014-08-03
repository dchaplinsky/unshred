class AbstractShredFeature(object):
    def __init__(self, sheet):
        self.sheet = sheet

    def get_info(self, shred, contour, name):
        # Features, tags suggestions
        return {}, ()
