{{ fullname }} 
{{ underline }}

.. automodule:: {{ fullname }}

.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: mymodule.rst
   :recursive:   

   {% for item in modules %}
      {{ item }}
   {% endfor %}
