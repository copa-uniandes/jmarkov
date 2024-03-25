Usage
=====

Installation
------------

To use jMarkov, first install it using pip:

.. code-block:: console

   (.venv) $ pip install jmarkov

Markov chains 
-------------
.. autoclass:: jmarkov.markov_chain.markov_chain

Continuous-time Markov chains 
-----------------------------
.. autoclass:: jmarkov.ctmc.ctmc

   .. automethod:: jmarkov.ctmc.ctmc.__init__
   
   .. automethod:: jmarkov.ctmc.ctmc.steady_state

   .. automethod:: jmarkov.ctmc.ctmc.transient_probabilities

Discrete-time Markov chains 
-----------------------------
.. autoclass:: jmarkov.dtmc.dtmc
    
   .. automethod:: jmarkov.dtmc.dtmc.__init__
   
   .. automethod:: jmarkov.dtmc.dtmc.steady_state

   .. automethod:: jmarkov.dtmc.dtmc.transient_probabilities
