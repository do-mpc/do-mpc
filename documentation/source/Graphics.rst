********
Graphics
********




Animations
##########



.. note:

    ImageMagick can be quite slow when exporting GIFs due to memory restrictions (RAM).
    This becomes apparent when stitching up the final GIF from the individual exported frames
    takes the biggest share of the processing time. Check your limits from the terminal with:

    .. code-block:: console

        identify -list resource

    If your hardware permits it, set the memory limit to >2GiB, according to this guide_.


_guide: https://blog.bigbinary.com/2018/09/12/configuring-memory-allocation-in-imagemagick.html
