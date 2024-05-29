import matplotlib.pyplot as plt


def set_figure(nr,
               nc,
               fs=10,
               fs_title=7.5,
               fs_legend=10,
               fs_xtick=3,
               fs_ytick=3,
               fs_axes=4,
               ratio=1,
               fc='black',
               aspect='equal',
               axis='off'):

    """
    Custom figure setup function that generates a nicely looking figure outline.
    It includes "making-sense"-font sizes across all text locations (e.g. title, axes).
    You can always change things later yourself through the outputs or plt.rc(...).
    """

    fig, axs = plt.subplots(nr, nc, figsize=(fs*nc*ratio, fs*nr))
    fig.set_facecolor(fc)

    try:
        axs = axs.flatten()
        for ax in axs:
            ax.set_facecolor(fc)
            ax.set_aspect(aspect)
            ax.axis(axis)
    except:
        axs.set_facecolor(fc)
        axs.set_aspect(aspect)
        axs.axis(axis)

    plt.rc("figure", titlesize=fs*fs_title)
    plt.rc("legend", fontsize=fs*fs_legend)
    plt.rc("xtick", labelsize=fs*fs_xtick)
    plt.rc("ytick", labelsize=fs*fs_ytick)
    plt.rc("axes", labelsize=fs*fs_axes, titlesize=fs*fs_title)

    return fig, axs