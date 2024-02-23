import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    channels = 3
    if rgb_image_info["alpha"]:
        channels += 1

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % channels == 0:
                r = row[elem]
            elif elem % channels == 1:
                g = row[elem]
            elif elem % channels == 2:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            gv = round(0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j])
            greyscale_pixel_array[i][j] += gv

    return greyscale_pixel_array


def computeStandardDeviationImage3x3(pixel_array, image_width, image_height):
    SobelAbsoluteArray = createInitializedGreyscalePixelArray(image_width, image_height)
    mean = 0.0
    pixel_sum = 0.0
    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    mean += pixel_array[y + i][x + j]
            mean = mean / 9
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    pixel_sum += abs(pixel_array[y + j][x + i] - mean)**2
            SobelAbsoluteArray[y][x] += math.sqrt(pixel_sum / 9.0)
            mean = 0.0
            pixel_sum = 0.0
    return SobelAbsoluteArray

def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    SobelAbsoluteArray = createInitializedGreyscalePixelArray(image_width, image_height)
    mean = 0.0
    pixel_sum = 0.0
    for y in range(2, image_height - 2):
        for x in range(2, image_width - 2):
            for i in [-2,-1,0,1,2]:
                for j in [-2,-1,0,1,2]:
                    mean += pixel_array[y + i][x + j]
            mean = mean / 25
            for eta in [-2,-1,0,1,2]:
                for xi in [-2,-1,0,1,2]:
                    pixel_sum += abs(pixel_array[y + eta][x + xi] - mean)**2
            SobelAbsoluteArray[y][x] += math.sqrt(pixel_sum / 25)
            mean = 0.0
            pixel_sum = 0.0
    return SobelAbsoluteArray

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    Thresholdlist = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] >= threshold_value:
                Thresholdlist[y][x] = 255
            else:
                Thresholdlist[y][x] = 0
    return Thresholdlist


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    min_max = computeMinAndMaxValues(pixel_array, image_width, image_height)
    if min_max[0] == min_max[1]: return greyscale_pixel_array
    for i in range (image_height):
        for j in range (image_width):
            value = round(((pixel_array[i][j] - min_max[0]) / (min_max[1] - min_max[0])) * 255)
            if value < 0: value = 0
            if value > 255: value = 255
            greyscale_pixel_array[i][j] += value
    return greyscale_pixel_array



def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min = 9000
    max = -500
    for inner in pixel_array:
        for each in inner:
            if each > max: max = each
            if each < min: min = each
    return (min, max)



def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    indivial_pixel = 0
    compare_x = 9
    for y in range(image_height):
        for x in range(image_width):
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if y + i < 0 or x + j < 0 or y + i >= image_height or x + j >= image_width:
                        indivial_pixel += 0
                    else:
                        indivial_pixel += pixel_array[y + i][x + j]
                        if pixel_array[y + i][x + j] == 255:
                            compare_x = 2295
            if indivial_pixel == compare_x:
                result_pixel_array[y][x] = 1
            indivial_pixel = 0
    return result_pixel_array

def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    indivial_pixel = 0
    compare_x = 9
    for y in range(image_height):
        for x in range(image_width):
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if y + i < 0 or x + j < 0 or y + i >= image_height or x + j >= image_width:
                        indivial_pixel += 0
                    else:
                        indivial_pixel += pixel_array[y + i][x + j]
            if indivial_pixel > 0:
                result_pixel_array[y][x] = 1
            indivial_pixel = 0
    return result_pixel_array


def bfs_traversal(pixel_array, visited, i, j, width, height, ccimg, count):
    num = 0

    q.enqueue((i, j))
    visited[i][j] = True

    while (not q.isEmpty()):

        a, b = q.dequeue()

        ccimg[a][b] = count
        num += 1

        for z in range(4):
            newI = a + x[z]
            newJ = b + y[z]
            if newI >= 0 and newI < height and newJ >= 0 and newJ < width and not visited[newI][newJ] and \
                    pixel_array[newI][newJ] != 0:
                visited[newI][newJ] = True
                q.enqueue((newI, newJ))

    return num


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    visited = []
    ccimg = []

    for i in range(image_height):
        temp1 = []
        temp2 = []
        for j in range(image_width):
            temp1.append(False)
            temp2.append(0)
        visited.append(temp1)
        ccimg.append(temp2)

    ccsizedict = {}
    count = 1
    max_num = -99
    max_count = -99
    for i in range(image_height):
        for j in range(image_width):
            if not visited[i][j] and pixel_array[i][j] != 0:
                num = bfs_traversal(pixel_array, visited, i, j, image_width, image_height, ccimg, count)
                ccsizedict[count] = num
                if num >= max_num:
                    max_num = num
                    max_count = count
                count += 1

    return (ccimg, ccsizedict, max_num, max_count)

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate1.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)



    #Step 1 to greyscale
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    #step 2 Filtering to detect high contrast regions
    standard_deviation_5x5_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    #step 3 Contrast Stretching
    scaled_standard_array = scaleTo0And255AndQuantize(standard_deviation_5x5_array, image_width, image_height)
    #step 4 Thresholding for Segmentation
    threshold_value = 150
    threshold_array = computeThresholdGE(scaled_standard_array, threshold_value, image_width, image_height)
    #step 5 Morphological operations
    dilated_array = computeDilation8Nbh3x3FlatSE(threshold_array, image_width, image_height)
    for i in range(3):
        dilated_array = computeDilation8Nbh3x3FlatSE(dilated_array, image_width, image_height)

    eroded_array = computeErosion8Nbh3x3FlatSE(dilated_array, image_width, image_height)
    for i in range(3):
        eroded_array = computeErosion8Nbh3x3FlatSE(eroded_array, image_width, image_height)

    #step 6 Connected component analysis
    (ccimg, ccsizedict, max_num, max_count) = computeConnectedComponentLabeling(eroded_array, image_width, image_height)


    # for y in range(image_height):
    #     for x in range(image_width):
    #         if ccimg[y][x] == max_count:
    #             if y <= min_y:
    #                 min_y = y
    #             if y >= max_y:
    #                 max_y = y
    #             if x <= min_x:
    #                 min_x = x
    #             if x >= max_x:
    #                 max_x = x
    ccsizedict_sorted_keys = sorted(ccsizedict, key=ccsizedict.get, reverse=True)

    keys_list = list(ccsizedict_sorted_keys)
    key_count = 0
    while True:
        min_x = 99999
        max_x = -99
        min_y = 99999
        max_y = -99
        key = keys_list[key_count]
        for y in range(image_height):
            for x in range(image_width):
                if ccimg[y][x] == key:
                    if y <= min_y:
                        min_y = y
                    if y >= max_y:
                        max_y = y
                    if x <= min_x:
                        min_x = x
                    if x >= max_x:
                        max_x = x
        aspect_ratio = (max_x - min_x) / (max_y - min_y)
        key_count += 1
        if 1.5 <= aspect_ratio <= 5:
            break

    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    bbox_min_x = min_x
    bbox_max_x = max_x
    bbox_min_y = min_y
    bbox_max_y = max_y

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)

    axs1[0,0].set_title('RGB to Greyscale')
    axs1[0,0].imshow(px_array, cmap='gray')

    axs1[0,1].set_title('Standard Deviation 5x5 of image')
    axs1[0,1].imshow(standard_deviation_5x5_array, cmap='gray')

    axs1[1, 0].set_title('Threshold of image')
    axs1[1, 0].imshow(threshold_array, cmap='gray')

    axs1[1, 1].set_title('Plate Detected of image')
    axs1[1, 1].imshow(px_array, cmap='gray')
    # Draw a bounding box as a rectangle into the input image
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # fig1, axs1 = pyplot.subplots(2,1)
    # axs1[0].set_title('Closing of image')
    # axs1[0].imshow(eroded_array, cmap='gray')
    # pyplot.show()


    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    q = Queue()
    x = [-1, 0, 1, 0]
    y = [0, 1, 0, -1]
    main()
