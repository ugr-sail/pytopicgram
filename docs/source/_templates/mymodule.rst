{{ fullname }} 
{{ underline }}

.. automodule:: {{ fullname }}

.. rubric:: Functions

.. autosummary::
   :toctree:
   :recursive:   

   {% for item in functions %}
      {{ item }}
   {% endfor %}
