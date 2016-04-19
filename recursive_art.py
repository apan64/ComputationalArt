""" This Python package provides an implementation of generating 'art' via recursion """

import random, math, sys, cv2, numpy, pygame
from PIL import Image
func = ['prod', 'avg', 'x', 'y', 'diff', 'cos', 'sin', 'sqrt(abs)']

def build_random_function(min_depth, max_depth):
    """ Builds a random function of depth at least min_depth and depth
        at most max_depth (see assignment writeup for definition of depth
        in this context)

        min_depth: the minimum depth of the random function
        max_depth: the maximum depth of the random function
        returns: the randomly generated function represented as a nested list
                 (see assignment writeup for details on the representation of
                 these functions)
    """
    if(min_depth < 1):#base case, once min depth is reached there is a chance it returns x or y to end recursion
        if(random.randint(0, 1) == 1):#decides whether to end
            if(random.randint(0, 1) == 1):#chooses what to return
                return ['x']
            return ['y']
    if(max_depth == 0):#base case, can't go deeper than max
        if(random.randint(0, 1) == 1):
            return ['x']
        return ['y']
    choice = random.randint(0, len(func) - 1)#random choice from list for function
    if(choice > 4):
        return [func[choice], build_random_function(min_depth - 1, max_depth - 1)]#functions that only take one argument
    return [func[choice], build_random_function(min_depth - 1, max_depth - 1), build_random_function(min_depth - 1, max_depth - 1)]#functions for two arguments



def evaluate_random_function(f, x, y):
    """ Evaluate the random function f with inputs x,y
        Representation of the function f is defined in the assignment writeup

        f: the function to evaluate
        x: the value of x to be used to evaluate the function
        y: the value of y to be used to evaluate the function
        returns: the function value

        >>> evaluate_random_function(["x"],-0.5, 0.75)
        -0.5
        >>> evaluate_random_function(["y"],0.1,0.02)
        0.02
    """
    if(len(f) == 1):#base case
        return {'x' : x, 'y' : y}[f[0]]
    if(f[0] == 'cos'):#checks current function then recurs with operation
        return math.cos(math.pi * evaluate_random_function(f[1], x, y))
    if(f[0] == 'sin'):
        return math.sin(math.pi * evaluate_random_function(f[1], x, y))
    if(f[0] == 'prod'):
        return evaluate_random_function(f[1], x, y) * evaluate_random_function(f[2], x, y)
    if(f[0] == 'avg'):
        return .5*(evaluate_random_function(f[1], x, y) + evaluate_random_function(f[2], x, y))
    if(f[0] == 'x'):
        return evaluate_random_function(f[1], x, y)
    if(f[0] == 'y'):
        return evaluate_random_function(f[2], x, y)
    if(f[0] == 'sqrt(abs)'):
        return math.sqrt(abs(evaluate_random_function(f[1], x, y)))
    if(f[0] == 'diff'):
        return (evaluate_random_function(f[1], x, y) - evaluate_random_function(f[2], x, y)) / 2



def remap_interval(val, #changed to operate on python2 (I was using python3 so integer truncation wasn't an issue)
                   input_interval_start,
                   input_interval_end,
                   output_interval_start,
                   output_interval_end):
    """ Given an input value in the interval [input_interval_start,
        input_interval_end], return an output value scaled to fall within
        the output interval [output_interval_start, output_interval_end].

        val: the value to remap
        input_interval_start: the start of the interval that contains all
                              possible values for val
        input_interval_end: the end of the interval that contains all possible
                            values for val
        output_interval_start: the start of the interval that contains all
                               possible output values
        output_interval_end: the end of the interval that contains all possible
                            output values
        returns: the value remapped from the input to the output interval

        >>> remap_interval(0.5, 0, 1, 0, 10)
        5.0
        >>> remap_interval(5, 4, 6, 0, 2)
        1.0
        >>> remap_interval(5, 4, 6, 1, 2)
        1.5
    """
    baseInEnd = float(input_interval_end - input_interval_start) #set input range to start from zero
    baseOutEnd = float(output_interval_end - output_interval_start) #set output range to start from zero
    return baseOutEnd/baseInEnd * (val - input_interval_start) + output_interval_start #val translated from input range to output range, then output start added



def color_map(val):
    """ Maps input value between -1 and 1 to an integer 0-255, suitable for
        use as an RGB color code.

        val: value to remap, must be a float in the interval [-1, 1]
        returns: integer in the interval [0,255]

        >>> color_map(-1.0)
        0
        >>> color_map(1.0)
        255
        >>> color_map(0.0)
        127
        >>> color_map(0.5)
        191
    """
    # NOTE: This relies on remap_interval, which you must provide
    color_code = remap_interval(val, -1, 1, 0, 255)
    return int(color_code)


def test_image(filename, x_size=350, y_size=350):
    """ Generate test image with random pixels and save as an image file.

        filename: string filename for image (should be .png)
        x_size, y_size: optional args to set image dimensions (default: 350)
    """
    # Create image and loop over all pixels
    im = Image.new("RGB", (x_size, y_size))
    pixels = im.load()
    for i in range(x_size):
        for j in range(y_size):
            x = remap_interval(i, 0, x_size, -1, 1)
            y = remap_interval(j, 0, y_size, -1, 1)
            pixels[i, j] = (random.randint(0, 255),  # Red channel
                            random.randint(0, 255),  # Green channel
                            random.randint(0, 255))  # Blue channel

    im.save(filename)


def generate_art(filename, x_size=350, y_size=350):
    """ Generate computational art and save as an image file.

        filename: string filename for image (should be .png)
        x_size, y_size: optional args to set image dimensions (default: 350)
    """
    # Functions for red, green, and blue channels - where the magic happens!
    red_function = build_random_function(7, 9)
    green_function = build_random_function(7, 9)
    blue_function = build_random_function(7, 9)

    # Create image and loop over all pixels
    im = Image.new("RGB", (x_size, y_size))
    pixels = im.load()
    for i in range(x_size):
        for j in range(y_size):
            x = remap_interval(i, 0, x_size, -1, 1)
            y = remap_interval(j, 0, y_size, -1, 1)
            pixels[i, j] = (
                    color_map(evaluate_random_function(red_function, x, y)),
                    color_map(evaluate_random_function(green_function, x, y)),
                    color_map(evaluate_random_function(blue_function, x, y))
                    )
    im.save(filename)
    editPicture(pixels, x_size, y_size)

def editPicture(p, x_size, y_size): #edits a picture after having the pixels passed in as an argument, position of distortion based on face recognition
    editedFile = 'edited_art.png'
    im = Image.new("RGB", (x_size, y_size))
    pixels = im.load()
    for i in range(x_size):
        for j in range(y_size):
            pixels[i, j] = p[i, j]
    kernel = numpy.ones((21,21),'uint8')
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
    screen = pygame.display.set_mode((x_size, y_size))
    while(True):
        ret, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
            for i in range(int(w/3), int(w*2/3)):
                for j in range(int(h/3), int(h*2/3)):
                    try:
                        xRand = random.randint(0, x_size - 1)
                        yRand = random.randint(0, y_size - 1)
                        pixels[int(x+i - w/3), int(y+j - h/3)] = (int(pixels[xRand, yRand][0]*random.randint(1, 5)/random.randint(1, 5)), int(pixels[xRand, yRand][1]*random.randint(1, 5)/random.randint(1, 5)), int(pixels[xRand, yRand][2]*random.randint(1, 5)/random.randint(1, 5))) #randomly distorts image
                    except IndexError: #for when the face is outside of picture range, continues on with rest of program
                        pass
        im.save(editedFile)
        pic = pygame.image.load(editedFile)
        screen.blit(pic,(0,0))
        pygame.display.update()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pygame.quit()
            sys.exit()

if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    generate_art("myart.png")
