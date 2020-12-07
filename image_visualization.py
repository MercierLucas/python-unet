import matplotlib.pyplot as plt


def compare_images(images,titles,img_to_show=2):
    rows = len(images)
    for i in range(img_to_show):
        for r in range(0,rows):
            plt.subplot(img_to_show,rows,i*rows+r+1)
            plt.title(titles[r])
            img = images[r][i].squeeze()
            plt.imshow(img)
