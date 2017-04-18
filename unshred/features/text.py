import sys
import json
import os.path
import glob

import cv2
import numpy as np

import Image
import ImageEnhance
import ImageFilter
import ImageDraw
import ImageOps

import base

DEBUG = True

# Minimum number of detected text lines. 
# If numer of lines recognized are below this value - the result is undefined
MIN_LINES_FOR_RESULT = 3  

# Magic values
MAGIC_COLOR_THRESHOLD = 10
MAGIC_LINE_THRESHOLD = 10
MAGIC_SECTIONS_THRESHOLD = 64

MAGIN_GROUP_VALUE_THRESHOLD = 0.3
MAGIC_GROUP_LEN_THRESHOLD = 5

class TextFeatures(base.AbstractShredFeature):
    """ 
        Tries to guess the following features of the shread:

            * text direction (in angles) relative to original shread
            * number of text lines
            * positions of text lines (after rotation) and their heights

        Algorithm is pretty straightforward and slow:
            
            1. increase shread contrast
            2. rotate image from -45 to +45 angles and compute 
               image line histogram (sum of inverted pixels for every line)
            3. analyze histogram for every angle to compute resuling coefficients
            3. sort computed parameters and choose the best match
            
        Currently, the best match is selected by angle, at which 
        maximum number of text lines is found with minimum heights for each line.

        TODO: 
              * better way to increase contrast of the image (based on histogram)
              * include and analyze additional parameters of rotated shread, like 
                contrast of horizontal lines histogram, connect with lines detector 
                for more accurate results, etc..
              * improve performance by using OpenCV/numpy for computation
        
    """

    def enhance(self, image, enhancer_class, value):
        enhancer = enhancer_class(image);
        return enhancer.enhance(value)

    def desaturate(self, image):
        """ Get green component from the PIL image
        """
        r, g, b, a = image.split()
        return Image.merge("RGBA", (g,g,g,a))

    def get_derivative(self, values):
        """ Calculate derivative of a list of values
        """
        result = []
        for i in xrange(1, len(values) - 1):
            result.append((values[i + 1] - values[i - 1]) / 2.0)

        return result

    def get_sections(self, values):
        """ Analyze lines histogram and return list of sections 
            (consecutive blocks of data values > threashold)

            Args:
                values: horizontal line histogram

            Returns:

                List of Sections.
                    Section is an uninterrupted part of histogram
                    dictionary with keys:

                    pos:   start position of section
                    len:   length of section
                    value: inverted sum of pixels for that section
        """
        sections = []

        current_section = []
        is_in_section = False
        spacing_len = 0
        position = 0

        for i, value in enumerate(values):

            if value > MAGIC_SECTIONS_THRESHOLD:

                if not is_in_section:
                    sections.append({'len' : -spacing_len, 'value' : 0, 'pos': i - spacing_len})
                    is_in_section = True
                    spacing_len = 0

                current_section.append(value)
            else:

                if is_in_section:
                    is_in_section = False;
                    sections.append({ 'len' : len(current_section), 'value' : sum(current_section), 'pos' : i - len(current_section)})
                    current_section = []

                spacing_len += 1

        return sections     

    def get_histogram_for_angle(self, image, angle):
        """ 
            Rotates an image for the specified angle and calculates
            sum of pixel values for every row

            Args:

                image: PIL image object
                angle: rotation angle

            Returns:
                list of values. Each value is a sum of inverted pixel's for the corresponding row
        """

        copy = image.rotate(angle, Image.BILINEAR, True)

        line_histogram = []
                
        for i in xrange(copy.size[1]):
            line = copy.crop( (0, i, copy.size[0], i + 1))

            value = 0

            for pixel in line.getdata():
                if pixel[0] < MAGIC_COLOR_THRESHOLD and pixel[1] == 255:
                    value += 255 - pixel[0]
                
            line_histogram.append(value)           
                       
        return line_histogram

    def group_section_below_threshold(self, section, group_threshold):
        if section['len'] > 0 and section['value'] < group_threshold:
            return True
        
        if section['len'] <= 0 and section['len'] > - MAGIC_GROUP_LEN_THRESHOLD:
            return True

        return False

    def group_sections(self, sections):
        """ Groups adjacent sections which are devided by only few pixels.
        """        
        finished = False

        section_avg_value = 0
        positive_sections = [s['value'] for s in sections if s['len'] > 0]

        if len(positive_sections) > 0:
            section_avg_value = sum( positive_sections ) / float(len(positive_sections))

        group_threshold = section_avg_value * MAGIN_GROUP_VALUE_THRESHOLD

        while not finished:

            finished = True
            for i in xrange(1, len(sections) - 1):
                if self.group_section_below_threshold(sections[i], group_threshold):
                    sections[i-1]['len'] += sections[i+1]['len'] + sections[i]['len']
                    sections[i-1]['value'] += sections[i+1]['value']

                    sections[i:i+2] = []
                    finished = False
                    break

        if len(sections) == 0:
            return

        if self.group_section_below_threshold(sections[0], group_threshold):
            sections[0:1] = []

        if len(sections) == 0:
            return

        if self.group_section_below_threshold(sections[-1], group_threshold):
            sections[-1:] = []

    def log_format_sections(self, sections):
        """ Formats sections in human-readable form for debugging
        """
        
        data = []
        for section in sections:
            data.append("%s (%s)" % (section['len'], section['value']))

        return ", ".join(data)

    def get_derivative_coef(self, histogram):
        """ Calculates the square sum of derivative from histogram
            This can be used to measure "sharpness" of the histogram
        """
        derivative = self.get_derivative(histogram)
        return sum(map(lambda x: x*x, derivative))

    def get_rotation_info(self, image, angle):
        """ 
            Rotates image and compute resulting coefficients for the specified angle
            
            Args:
                image: grayscale python image
                angle: angle for which to rotate an image

            Returns:

                dictionary with values for the specified angle
                
                Coefficients currently computed:
                    nsc (Normalized Sections Count) - number of text lines, 
                         without those lines, which have very little pixels in them 

                    heights - sum of heights of lines 

                Additional lists returned (currently used only for debug and experiments):
                    derivative_pos: list of positive derivatives values for histogram
                    derivative_neg: list of negative derivatives values for histogram
                    full_sections:  list of sections with enough data for analysis
                    sections:       list of all sections
        """

        diagram = self.get_histogram_for_angle(image, angle)
        sections = self.get_sections(diagram)        

        self.group_sections(sections)            


        # Remove all spacing sections
        sections = [s for s in sections if s['len'] > 0]
        #positive_sections = [s for s in sections if s['len'] > 0]

        full_sections = []
        normalized_sections_count = 0
        sections_heights = 0

        if len(sections) > 0:
            # get average section size
            section_avg_value = sum( [s['value'] for s in sections] ) / float(len(sections))

            full_sections = [s for s in sections if s['value'] > 0.3 * section_avg_value]
            normalized_sections_count = len(full_sections)

            sections_heights = sum( map(lambda x: x['len'], full_sections) )

        return {'angle' : angle, 
                'nsc': normalized_sections_count,
                'heights': sections_heights,
                'derivative': self.get_derivative_coef(diagram),
                'full_sections': full_sections}

    def sort_result(self, result):
        """ Sort result by important parameters

            Args:
                result: list of dictionaries for each tested angle

            Returns:
                sorted dict with the most accurate result first
        """

        def sort_fun2(a, b):
            if b['nsc'] == a['nsc']:
                return a['heights'] - b['heights']

            return b['nsc'] - a['nsc']
                
        def sort_fun(a, b):
            if b['nsc'] == a['nsc']:
                return b['derivative'] - a['derivative']
            
            return b['nsc'] - a['nsc']

        result.sort( sort_fun2 )

    def info_for_angles(self, image):
        """Args:
                image: grayscale python image

            Returns:
                list of dicts with info for every angle tested
        """

        result = []
        for angle in xrange(-45, 45):
            
            if DEBUG: sys.stdout.write(".")

            rotation_info = self.get_rotation_info(image, angle)
            result.append(rotation_info) # diagram, derivative

        if DEBUG: sys.stdout.write("\n")            

        self.sort_result(result)

        return result


    def get_info(self, shred, contour, name):

        if DEBUG:        
            print "Processing file: %s" % (name)

        image = Image.fromarray(cv2.cvtColor(shred, cv2.COLOR_BGRA2RGBA))
        image = image.convert("LA")

        image = self.enhance(image, ImageEnhance.Brightness, 1.5);
        image = self.enhance(image, ImageEnhance.Contrast, 3);

        results = self.info_for_angles(image)

        top_result = results[0]       
        resulting_angle = top_result['angle']

        if DEBUG:
            result = image.rotate(resulting_angle, Image.BILINEAR, True)            
            result.save("results/%s" % (name))
        
        if top_result['nsc'] >= MIN_LINES_FOR_RESULT:
            return {'text_angle' : resulting_angle, 'text_sections' : [{'pos' : s['pos'], 'length' : s['len']} for s in top_result['full_sections']]}
        else:
            return {'text_angle' : "undefined" }
    
if __name__ == '__main__':

    def process_shred(full_name):

        features = TextFeatures(None)
        cv_image = cv2.imread(full_name, -1)

        file_name = os.path.split(full_name)[1]

        result = features.get_info(cv_image, None, file_name)

        if result == None:
            return

        with open("results/%s.json" %(file_name), "wt") as f_info:
            f_info.write( json.dumps(result, sort_keys=True,
                            indent=4, separators=(',', ': ')) )

    if len(sys.argv) < 2:
        print "Error: Please specify path or file"
        sys.exit(255)

    path = sys.argv[1]

    if os.path.isfile(path):
        process_shred(path)
    else:

        for full_name in glob.glob("%s\\*.png" % (path)): 

            if full_name.count("_ctx") > 0 or full_name.count("_mask") > 0:
                continue

            process_shred(full_name)
    