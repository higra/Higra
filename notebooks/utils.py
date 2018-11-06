import time


def imshow2(image):
    dpi = 80.0
    h, w = image.shape[:2]
    figsize = (h / dpi, w / dpi)
    print(figsize)
    fig = plt.figure(figsize=(h / dpi, w / dpi))
    fig.figimage(image)
    plt.show()


def imshow(image, click_event=None):
    import matplotlib.pyplot as plt
    dpi = 80
    margin = 0.5  # (5% of the width/height of the figure...)
    h, w = image.shape[:2]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * w / dpi, (1 + margin) * h / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image, interpolation='none')
    
    plt.axis('off')
    plt.show()
    
    return fig, ax
    
 


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is ", str(time.time() - startTime_for_tictoc), " seconds.")
    else:
        print("Toc: start time not set")
