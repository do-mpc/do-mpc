{{ objname | escape | underline}}
.. .. currentmodule:: {{ fullname }}
.. automodule:: {{ fullname }}
   {% block functions %}
   {% if functions %}
   .. currentmodule:: {{ fullname }}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :template: method.rst
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. currentmodule:: {{ fullname }}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: class.rst
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. currentmodule:: {{ fullname }}
.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
