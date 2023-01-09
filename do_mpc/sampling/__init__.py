"""
Sampling tools for data generation.

For a quick introduction of the **do-mpc** sampling tools we are providing this video tutorial:

.. raw :: html

    <style>
    .video-container {
    position: relative;
    padding-bottom: 56.25%; /* 16:9 */
    height: 0;
    }
    .video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    }
    </style>


    <!-- HTML -->
    <div class="video-container">
    <iframe src="https://www.youtube-nocookie.com/embed/3ELyErkYPhE"
    title="YouTube video player" frameborder="0" allow="accelerometer;
    autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
    </iframe>
    </div>.
"""

from ._sampler import Sampler
from ._datahandler import DataHandler
from ._samplingplanner import SamplingPlanner
