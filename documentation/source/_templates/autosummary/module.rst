{{ objname | escape | underline }}

.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes
.. autosummary::
    :toctree:
    :nosignatures:
    {% for class in classes %}
    {{ class }}
    {% endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions
.. autosummary::
    :toctree:
    :nosignatures:
    {% for function in functions %}
    {{ function }}
    {% endfor %}
{% endif %}

This page is auto-generated. Page source is not available on Github.
