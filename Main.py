"""Author: Teresa Tang
Last updated: 1/4/2020 8:16 PM"""
import numpy as np
import cv2
import pygame

# global constants
OPAQUE_BLACK = (0, 0, 0, 255)

"""Purpose: Determine if a given point is out of bounds of the image"""
def isOutOfBounds(x, y, width, height):
    if x < 0 or x >= width:
        return True
    elif y < 0 or y >= height:
        return True
    else:
        return False

"""Purpose: To fill all center points in the center list if the point is not black or has already been filled with 
the color of interest"""
def colorPoint(center_list, image, color, width, height):
    opaque_color = color + (255,)
    idx = 0
    while(idx < len(center_list)):
        # delete center if it is out of bounds
        if isOutOfBounds(center_list[idx][0], center_list[idx][1], width, height):
            center_list.pop(idx)
        # delete center if black or given color
        elif image.get_at(center_list[idx]) == OPAQUE_BLACK or image.get_at(center_list[idx]) == opaque_color:
            center_list.pop(idx)
        else:
            image.set_at(center_list[idx], opaque_color)
            idx = idx + 1
    return center_list

"""Purpose: Generate the surrounding points to every center point (top, bottom, left, right, and diagonals)"""
def createCenters(center_list):
    # create list of surrounding points to every center point
    # the center point itself is discarded
    surround_list = []
    for point in center_list:
        # above
        surround_list.append((point[0], point[1] + 1))
        # below
        surround_list.append((point[0], point[1] - 1))
        # right
        surround_list.append((point[0] + 1, point[1]))
        # left
        surround_list.append((point[0] - 1, point[1]))
        # diagonals
        # using diagonals makes the fill process significantly slower
        """surround_list.append((point[0] + 1, point[1] + 1))
        surround_list.append((point[0] + 1, point[1] - 1))
        surround_list.append((point[0] - 1, point[1] + 1))
        surround_list.append((point[0] - 1, point[1] - 1))"""

    return surround_list

"""Purpose: Fills a given point with a color if it is not a boundary point"""
def fill(image, mouseX, mouseY, color, width, height):
    # the first center point is the mouse click point
    # a center point is one that can be filled
    center_list = [(mouseX, mouseY)]
    # determine if the mouse click point can be filled
    center_list = colorPoint(center_list, image, color, width, height)

    # while viable center points exist
    while(len(center_list) != 0):
        # generate the points connected to a given center point
        center_list = createCenters(center_list)
        # determine if center points can be filled
        center_list = colorPoint(center_list, image, color, width, height)

# deprecated
"""Purpose: To color points on the image surface corresponding to contour points black"""
def colorOutline(contours, image):
    for point in contours:
        image.set_at((point[0], point[1]), (0, 0, 0))
    return image

"""Purpose: To transfer the black outline in the contour image to a pygame surface"""
def outline(contour_image, image):
    contour_image = np.rot90(contour_image, k=1, axes=(0, 1))
    contour_image = np.flip(contour_image, axis=0)
    # search for black points in the binary image and set corresponding points on pygame surface black
    for row_idx in range(0, contour_image.shape[0]):
        for col_idx in range(0, contour_image.shape[1]):
            if (contour_image[row_idx][col_idx] == 0):
                image.set_at((row_idx, col_idx), OPAQUE_BLACK)
    return image

# not used
"""Purpose: Turn list of contour arrays into a 2D array of all contour points"""
def create2DArray(contours):
    # contourArray initializes as first array in the list of contour arrays
    contourArray = np.reshape(contours[0], (contours[0].shape[0], 2))

    for contour in contours[1:]:
        # turn 3D contour array into 2D array
        numRows = contour.shape[0]
        contour = np.reshape(contour, (numRows, 2))

        # concatenate 2D contour array onto array of all contour points
        contourArray = np.vstack((contourArray, contour))

    return contourArray

# not used
"""Purpose: Sort contour points by increasing x (sort every (x, y) with equal x by ascending y)"""
def sortArray(contours):
    total_rows = contours.shape[0]
    # sort by ascending x values
    contours = contours[np.argsort(contours[:, 0])]

    start_row_idx = 0
    row_idx = 1
    row_val = contours[0, :][0]
    for row in contours[1:, :]:
        # if x values differ
        if(row_val != row[0]):
            # sort section with same x values by ascending y value
            section = contours[start_row_idx:row_idx, :]
            section = section[np.argsort(section[:, 1])]
            contours[start_row_idx:row_idx, :] = section
            row_val = row[0]
            start_row_idx = row_idx
        # if last row is reached
        elif(row_idx == total_rows - 1):
            # create section with equal x values ending at the last row and sort by ascending y value
            section = contours[start_row_idx:total_rows, :]
            section = section[np.argsort(section[:, 1])]
            contours[start_row_idx:total_rows, :] = section

        row_idx = row_idx + 1
    return contours

# not used
"""Purpose: Create dictionary with keys as x coordinates and values as y values with the same x coordinate"""
def createDictionary(contours):
    contour_dict = {}
    total_rows = contours.shape[0]

    row_idx = 1
    # list of y values corresponding to the same x value
    y_list = [contours[0][1]]

    # x value of interest
    x = contours[0][0]

    for row in contours[1:, :]:
        # if x values differ
        if (x != row[0]):
            # add list of y values corresponding to original x value to the dictionary
            contour_dict[x] = y_list

            # start new list of y values corresponding to the new, different x value
            y_list = [contours[row_idx][1]]
            x = row[0]
        # if last row is reached
        elif (row_idx == total_rows - 1):
            # add last list of y values to the dictionary
            contour_dict[x] = y_list
        else:
            # add a new y value corresponding to the x value of interest
            y_list.append(contours[row_idx][1])

        row_idx = row_idx + 1
    return contour_dict

def findContours(img):
    # find image contours
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 180, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours, image

"""Purpose: Increase or decrease R, G, or B values (-1 in R coordinate position means this is the value to change, etc.)"""
def changeColor(color_bar, red, green, blue, y, bar_width, start, stop, step):
    # increment or decrement color coordinate
    val = start
    # decrementing
    if step < 0:
        while(val > stop):
            incremented_color = val

            # determine which color coordinate to change (red, blue, or green)
            if red == -1:
                color = (incremented_color,) + (green,) + (blue,) + (255,)
            elif green == -1:
                color = (red,) + (incremented_color,) + (blue,) + (255,)
            elif blue == -1:
                color = (red,) + (green,) + (incremented_color,) + (255,)

            # draw color onto color bar as a rectangle
            pygame.draw.rect(color_bar, color, (0, y, bar_width, 1))
            # place next color rectangle below the most recent one
            print(y)
            y = y + 1
            val = val + step
    elif step > 0:
        while (val < stop):
            incremented_color = val

            # determine which color coordinate to change (red, blue, or green)
            if red == -1:
                color = (incremented_color,) + (green,) + (blue,) + (255,)
            elif green == -1:
                color = (red,) + (incremented_color,) + (blue,) + (255,)
            elif blue == -1:
                color = (red,) + (green,) + (incremented_color,) + (255,)

            # draw color onto color bar as a rectangle
            pygame.draw.rect(color_bar, color, (0, y, bar_width, 1))
            # place next color rectangle below the most recent one
            print(y)
            y = y + 1
            val = val + step

    return y, color_bar

"""Purpose: Populate color palette with rainbow colors, adjusting size based on image size"""
def colorBar(color_bar, height, bar_width):
    y_val = 0
    step = 255/(height/6)
    print(step)

    # insert -1 to indicate color coordinate that will be changing
    # increase green while red and blue are zero
    y_val, color_bar = changeColor(color_bar, 255, -1, 0, y_val, bar_width, 0, 256, step)
    # decrease red while green is constant
    y_val, color_bar = changeColor(color_bar, -1, 255, 0, y_val, bar_width, 255, -1, -1 * step)
    # increase blue to 255
    y_val, color_bar = changeColor(color_bar, 0, 255, -1, y_val, bar_width, 0, 256, step)
    # decrease green until zero
    y_val, color_bar = changeColor(color_bar, 0, -1, 255, y_val, bar_width, 255, -1, -1 * step)
    # increase red until 255
    y_val, color_bar = changeColor(color_bar, -1, 0, 255, y_val, bar_width, 0, 256, step)
    # decrease blue until zero
    y_val, color_bar = changeColor(color_bar, 255, 0, -1, y_val, bar_width, 255, -1, -1 * step)

    return color_bar

# not in use
"""Purpose: Resize image to screen size"""
def resizeImage(raw_image):
    width = raw_image.shape[0]
    height = raw_image.shape[1]
    if(raw_image.shape[0] > 800):
        width = 800
    elif(raw_image.shape[1] > 800):
        height = 800

    return cv2.resize(raw_image, (width, height))

def main():
    # prevent array truncation during printing
    # np.set_printoptions(threshold=np.inf)

    # choose image to color
    image_name = "pusheen.jpg"
    raw_image = cv2.imread(image_name)
    # raw_image = resizeImage(raw_image)

    raw_contours, contour_image = findContours(raw_image)

    # color picker palette width
    color_bar_W = 50

    # create game
    pygame.init()

    # set screen size to image size
    width = contour_image.shape[1]
    height = contour_image.shape[0]
    print(width)
    print(height)
    screen = pygame.display.set_mode((width + color_bar_W, height))

    image = pygame.Surface((width, height))
    # white background
    image.fill((255, 255, 255))
    # transfer black outline of original image onto new image
    image = outline(contour_image, image)


    # create color selection palette
    color_bar = pygame.Surface((color_bar_W, height))
    color_bar.fill((255, 255, 255))
    # set white color bar pixel height
    white_height = 10
    # set height of entire color bar
    color_bar_H = height - (2 * white_height)
    print(color_bar_H)
    color_bar = colorBar(color_bar, color_bar_H, color_bar_W)

    # draw off black color bar at the bottom of the color bar
    # note that the black and white bar are the same height
    pygame.draw.rect(color_bar, (1, 1, 1), (0, color_bar_H - 1, color_bar_W, white_height))

    # default brush color
    fillColor = (255, 255, 255)

    game_running = True
    while(game_running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos

                # color bar selected
                if mouseX > width:
                    mouseX = mouseX - width
                    # set brush color
                    fillColor = (color_bar.get_at((mouseX, mouseY))[0], color_bar.get_at((mouseX, mouseY))[1], color_bar.get_at((mouseX, mouseY))[2])
                # fill area with selected color
                else:
                    print("Mouse X: "+str(mouseX) + " Mouse Y: "+str(mouseY))
                    print("Coloring in...")
                    fill(image, mouseX, mouseY, fillColor, width, height)
                    print("What a beautiful color!")

        screen.blit(image, (0, 0))
        screen.blit(color_bar, (width, 0))
        pygame.display.update()

    pygame.quit()
    quit()

if __name__ == "__main__":
    main()