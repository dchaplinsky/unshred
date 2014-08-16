class AbstractShredFeature(object):
    def __init__(self, sheet):
        self.sheet = sheet

    def get_info(self, shred, contour, name):
        """Processes each individual shred and returns features and tags
        suggestions.

        Check colours.py and geometry.py for examples

        Args:
            shred: openCV image in RGBA mode, rotated and with no background.
            contour: openCV contour (list of coords).
            name: identifier of this shred. You might use it to save debug
            images

        Returns:
            dict with determined features
            tuple or list of tags suggestions (might be empty)

            For example:
            {
                "histogram": [...],
                "width": ...,
                "height": ...
            },
            (
                "has blue ink",
                "part of the picture"
            )
        """
        return {}, ()
