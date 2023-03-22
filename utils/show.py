import matplotlib.pyplot as plt
def show(*img):
    if len(img) < 2:
        show_one(img[0])
        return

    fig, axes = plt.subplots(1, len(img))

    for img_i, i in zip(img, range(len(img))):
        axes[i].imshow(img_i, cmap='gray')
        axes[i].set_title('')

    fig.set_figwidth(20)  # ширина и
    fig.set_figheight(10)  # высота \"Figure\

    #     plt.gray()
    plt.show()


def show_one(img, n=10):
    fig, ax = plt.subplots()
    fig.set_figwidth(n)  # ширина и
    fig.set_figheight(n)  # высота \"Figure\
    plt.imshow(img, cmap='gray')
    plt.show()



def visualizer(generator):
    for data in generator:
        visualize(data)

def visualize(data):
    fig = plt.figure(constrained_layout=True, figsize=(18, 12))
    gs = fig.add_gridspec(4, 4)
    ax_orig = fig.add_subplot(gs[0:2, :2])
    ax_orig.set_title('Исходное изображение', fontsize=12)
    ax_masks = fig.add_subplot(gs[2:, 0:2])
    ax_masks.set_title('Маска', fontsize=12)
    ax_contours = fig.add_subplot(gs[:2, 2:])
    ax_contours.set_title('Рассчитанные контуры', fontsize=12)
    ax_gt = fig.add_subplot(gs[2:, 2:])
    ax_gt.set_title('GT', fontsize=12)

    ax_contours.axis('off')
    ax_orig.axis('off')
    ax_masks.axis('off')
    ax_gt.axis('off')

    keys = ['orig', 'mask',  'contours', 'gt']
    orig, mask,contours, gt = [data[key] for key in keys]
    # original
    ax_orig.clear()
    ax_orig.set_title('Исходное изображение', fontsize=12)
    ax_orig.imshow(orig, aspect='auto')
    ax_orig.axis('off')

    # masks
    ax_masks.clear()
    ax_masks.set_title('Маска', fontsize=12)
    ax_masks.imshow(mask, cmap='gray', aspect='auto')
    ax_masks.axis('off')

    # # contours
    ax_contours.clear()
    ax_contours.set_title('Рассчитанные контуры', fontsize=12)
    ax_contours.imshow(contours, aspect='auto')
    ax_contours.axis('off')

    ax_gt.clear()
    ax_gt.set_title('GT', fontsize=12)
    ax_gt.imshow(gt, aspect='auto')
    ax_gt.axis('off')
