API Reference
=============
Find below a table of all **do-mpc** modules.
Classes and functions of each module are shown on their respective page.

.. currentmodule:: do_mpc

.. rubric:: Core modules
The core modules are used to create the **do-mpc** control loop (click on elements to open documentation page):

.. graphviz::
    :name: control_loop
    :caption: **do-mpc** control loop and interconnection of classes.
    :align: center

    digraph G {
        graph [fontname = "Monaco"];
        node [fontname = "Monaco", fontcolor="#404040", color="#bdbdbd"];
        edge [fontname = "Monaco", color="#707070"];

        Model [label="Model", href="../api/do_mpc.model.Model.html#model", target="_top", shape=box, style=filled]
        MPC [href="../api/do_mpc.controller.MPC.html#mpc", target="_top", shape=box, style=filled]
        Simulator [href="../api/do_mpc.simulator.Simulator.html#simulator", target="_top", shape=box, style=filled]
        MHE [href="../api/do_mpc.estimator.MHE.html#mhe", target="_top", shape=box, style=filled]
        Data_MPC [label="MPCData", href="../api/do_mpc.data.MPCData.html#mpcdata", target="_top", shape=box, style=filled]
        Data_Sim [label="Data", href="../api/do_mpc.data.Data.html#data", target="_top", shape=box, style=filled]
        Data_MHE [label="Data", href="../api/do_mpc.data.Data.html#data", target="_top", shape=box, style=filled]
        Graphics [label="Graphics", href="../api/do_mpc.graphics.Graphics.html#graphics", target="_top", shape=box, style=filled]

        Model -> MPC;
        Model -> Simulator;
        Model -> MHE;

        Model [shape=box, style=filled]

        subgraph cluster_loop {{
            rankdir=TB;
            rank=same;
            MPC -> Simulator [label="inputs"];
            Simulator -> MHE [label="meas."];
            MHE -> MPC [label="states"];
        }}

        MPC -> Data_MPC;
        Simulator -> Data_Sim;
        MHE -> Data_MHE;

        Data_MPC -> Graphics;
        Data_Sim -> Graphics;
        Data_MHE -> Graphics;


    }


.. autosummary::
    :toctree: api

    model

    simulator

    optimizer

    controller

    estimator

    data

    graphics


.. rubric:: Sampling tools

.. currentmodule:: do_mpc.sampling
.. autosummary::
    :toctree: api

    samplingplanner

    sampler

    datahandler

..
    .. rubric:: Sampling tools
    .. autosummary::
        :toctree: api
        :recursive:

        sampling


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
    </div>
