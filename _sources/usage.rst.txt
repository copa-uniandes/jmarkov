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

   .. automethod:: jmarkov.dtmc.dtmc.period
   
   .. automethod:: jmarkov.dtmc.dtmc.is_irreducible

   .. automethod:: jmarkov.dtmc.dtmc.is_ergodic



Discrete-time Markov Decision Processes (MDPs) 
----------------------------------------------

.. autoclass:: jmarkov.mdp.dtmdp.dtmdp

   .. automethod:: jmarkov.mdp.dtmdp.dtmdp.__init__
   
   .. automethod:: jmarkov.mdp.dtmdp.dtmdp._check_transition_matrices

   .. automethod:: jmarkov.mdp.dtmdp.dtmdp._check_immediate_returns

   .. automethod:: jmarkov.mdp.dtmdp.dtmdp._check_discount_factor
