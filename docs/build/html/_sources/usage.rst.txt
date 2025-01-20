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

   .. automethod:: jmarkov.ctmc.ctmc.first_passage_time

   .. automethod:: jmarkov.ctmc.ctmc.occupation_time
   
   .. automethod:: jmarkov.ctmc.ctmc.is_irreducible

   .. automethod:: jmarkov.ctmc.ctmc.is_ergodic

Discrete-time Markov chains 
-----------------------------

.. autoclass:: jmarkov.dtmc.dtmc

   .. automethod:: jmarkov.dtmc.dtmc.__init__
   
   .. automethod:: jmarkov.dtmc.dtmc.steady_state

   .. automethod:: jmarkov.dtmc.dtmc.transient_probabilities

   .. automethod:: jmarkov.dtmc.dtmc.period
   
   .. automethod:: jmarkov.dtmc.dtmc.is_irreducible

   .. automethod:: jmarkov.dtmc.dtmc.is_ergodic

Infinite Continuous-time Birth-death Markov chains 
--------------------------------------------------
.. autoclass:: jmarkov.ctbd.ctbd

   .. automethod:: jmarkov.ctbd.ctbd.__init__
   
   .. automethod:: jmarkov.ctbd.ctbd._check_birth_death_rates

   .. automethod:: jmarkov.ctbd.ctbd.steady_state

   .. automethod:: jmarkov.ctbd.ctbd.is_irreducible

   .. automethod:: jmarkov.ctbd.ctbd.is_ergodic

Finite Continuous-time Birth-death Markov chains 
------------------------------------------------
.. autoclass:: jmarkov.finite_ctbd.finite_ctbd

   .. automethod:: jmarkov.finite_ctbd.finite_ctbd.__init__
   
   .. automethod:: jmarkov.finite_ctbd.finite_ctbd._check_birth_death_rates

   .. automethod:: jmarkov.finite_ctbd.finite_ctbd.steady_state

   .. automethod:: jmarkov.finite_ctbd.finite_ctbd.is_irreducible

   .. automethod:: jmarkov.finite_ctbd.finite_ctbd.is_ergodic

Queueing systems 
----------------

.. autoclass:: jmarkov.queue.mmk.mmk

   .. automethod:: jmarkov.queue.mmk.mmk.__init__
   
   .. automethod:: jmarkov.queue.mmk.mmk.mean_number_entities

   .. automethod:: jmarkov.queue.mmk.mmk.mean_number_entities_queue

   .. automethod:: jmarkov.queue.mmk.mmk.mean_number_entities_service

   .. automethod:: jmarkov.queue.mmk.mmk.mean_time_system

   .. automethod:: jmarkov.queue.mmk.mmk.mean_time_queue

   .. automethod:: jmarkov.queue.mmk.mmk.mean_time_service

   .. automethod:: jmarkov.queue.mmk.mmk.is_stable


.. autoclass:: jmarkov.queue.mmkn.mmkn

   .. automethod:: jmarkov.queue.mmkn.mmkn.__init__
   
   .. automethod:: jmarkov.queue.mmkn.mmkn.mean_number_entities

   .. automethod:: jmarkov.queue.mmkn.mmkn.mean_number_entities_queue

   .. automethod:: jmarkov.queue.mmkn.mmkn.mean_number_entities_service

   .. automethod:: jmarkov.queue.mmkn.mmkn.mean_time_system

   .. automethod:: jmarkov.queue.mmkn.mmkn.mean_time_queue

   .. automethod:: jmarkov.queue.mmkn.mmkn.mean_time_service

.. autoclass:: jmarkov.queue.mmkn.mmkn

   .. automethod:: jmarkov.queue.mg1.mg1.__init__
   
   .. automethod:: jmarkov.queue.mg1.mg1.mean_number_entities

   .. automethod:: jmarkov.queue.mg1.mg1.mean_number_entities_queue

   .. automethod:: jmarkov.queue.mg1.mg1.mean_number_entities_service

   .. automethod:: jmarkov.queue.mg1.mg1.mean_time_system

   .. automethod:: jmarkov.queue.mg1.mg1.mean_time_queue

   .. automethod:: jmarkov.queue.mg1.mg1.mean_time_service

.. autoclass:: jmarkov.queue.gginf.gginf

   .. automethod:: jmarkov.queue.gginf.gginf.__init__
   
   .. automethod:: jmarkov.queue.gginf.gginf.mean_number_entities

   .. automethod:: jmarkov.queue.gginf.gginf.mean_number_entities_queue

   .. automethod:: jmarkov.queue.gginf.gginf.mean_number_entities_service

   .. automethod:: jmarkov.queue.gginf.gginf.mean_time_system

   .. automethod:: jmarkov.queue.gginf.gginf.mean_time_queue

   .. automethod:: jmarkov.queue.gginf.gginf.mean_time_service

Discrete-time Markov Decision Processes (MDPs) 
----------------------------------------------

.. autoclass:: jmarkov.mdp.dtmdp.dtmdp

   .. automethod:: jmarkov.mdp.dtmdp.dtmdp.__init__
   
   .. automethod:: jmarkov.mdp.dtmdp.dtmdp._check_transition_matrices

   .. automethod:: jmarkov.mdp.dtmdp.dtmdp._check_immediate_returns

   .. automethod:: jmarkov.mdp.dtmdp.dtmdp._check_discount_factor

Discrete-time Stochastic Dynamic Programming (SDPs) 
---------------------------------------------------

.. autoclass:: jmarkov.sdp.dtsdp.dtsdp

   .. automethod:: jmarkov.sdp.dtsdp.dtsdp.__init__
   
   .. automethod:: jmarkov.sdp.dtsdp.dtsdp._check_transition_matrices

   .. automethod:: jmarkov.sdp.dtsdp.dtsdp._check_immediate_returns

   .. automethod:: jmarkov.sdp.dtsdp.dtsdp._check_discount_factor

   .. automethod:: jmarkov.sdp.dtsdp.dtsdp._check_time_period

   .. automethod:: jmarkov.sdp.dtsdp.dtsdp.solve           