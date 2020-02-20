{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
   .. autosummary::
      :toctree:
   {% for item in methods %}
      {{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}
