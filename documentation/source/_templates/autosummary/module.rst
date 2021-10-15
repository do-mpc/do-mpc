{{ objname | escape | underline }}


{% if classes %}
.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

.. rubric:: Classes
.. autosummary::
    :toctree:
    :nosignatures:
    {% for class in classes %}
    {{ class }}
    {% endfor %}
{% endif %}

{% if functions %}
.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

.. rubric:: Functions
.. autosummary::
    :toctree:
    :nosignatures:
    {% for function in functions %}
    {{ function }}
    {% endfor %}
{% endif %}

{% if modules %}
.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :recursive:
   :template: module.rst
   {% for module in modules %}
   {{ module }}
   {% endfor %}
{% endif %}

This page is auto-generated. Page source is not available on Github.
