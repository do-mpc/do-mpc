{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :special-members: __call__, __add__, __mul__, __getitem__

   {% block methods %}
   {% if methods %}
   .. currentmodule:: {{ fullname }}
   .. rubric:: {{ _('Methods') }}

   .. autosummary:: 
      :toctree:
      :template: method.rst
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') %}
      {{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. currentmodule:: {{ fullname }}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
      :template: attribute.rst
      {% for item in attributes %}
         {{ item }}
      {%- endfor %}
      {% endif %}
   {% endblock %}
