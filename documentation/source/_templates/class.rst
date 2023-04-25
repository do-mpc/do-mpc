{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :show-inheritance:
    :special-members: __call__, __getitem__

    {% block methods %}
    {% if methods %}
    .. currentmodule:: {{ fullname }}
    .. rubric:: {{ _('Methods') }}

    {% for item in methods %}
    {%- if not item.startswith('_') %}
    .. autofunction:: {{ item }}
    {%- endif -%}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. currentmodule:: {{ fullname }}
    .. rubric:: {{ _('Attributes') }}

    {% for item in attributes %}
    .. autoattribute:: {{fullname}}.{{item}}
    {%- endfor %}
    {% endif %}
    {% endblock %}
