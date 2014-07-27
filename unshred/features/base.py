class AbstractShredFeature(object):
    def __init__(self, sheet):
        self.sheet = sheet

    def get_info(self, shred, contour):
        # Features, tags suggestions
        return {}, ()
