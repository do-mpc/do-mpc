{{ objname | escape | underline('=')}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :show-inheritance:
    :special-members: __call__, __getitem__

{% block methods %}
{% if methods %}
.. currentmodule:: {{ fullname }}

Methods
-------

{% for item in methods %}
{%- if not item.startswith('_') %}
 
{{ item | escape | underline('~') }}

.. autofunction:: {{ item }}

{%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
 
.. currentmodule:: {{ fullname }}

Attributes
----------
 
{% for item in attributes %}
 
{{ item | escape | underline('~') }}

.. autoattribute:: {{fullname}}.{{item}}

{%- endfor %}
{% endif %}
{% endblock %}
