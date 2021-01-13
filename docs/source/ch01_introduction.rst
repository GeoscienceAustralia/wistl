.. _intro-page:

************
Introduction
************

Vulnerability and Adaptation to Wind Simulation (VAWS) is a software tool that can be used to model the vulnerability of small buildings such as domestic houses and light industrial sheds to wind. The primary use-case of VAWS is the examination of the change in vulnerability afforded by mitigation measures to upgrade a building’s resilience to wind hazard.

Background
==========

Development of VAWS commenced in 2009-2010 in a collaborative project, partly funded by the then Department of Climate Change and energy Efficiency (DCCE), between Geoscience Australia, James Cook University and JDH Consulting :cite:`GA2011,Wehner2010`. The development of the current version was undertaken as part of the Bushfire and Natural Hazard Cooperative Research Centre (BNHCRC) project “Improving the Resilience of Existing Housing to Severe Wind” led by James Cook University.


Overall logic
=============

The VAWS tool takes a component-based approach to modelling building vulnerability. It is based on the premise that overall building damage is strongly related to the failure of key connections.

The tool generates a building model by randomly selecting parameter values from predetermined probability distributions using a Monte Carlo process. Values include component and connection strengths, external pressure coefficients, shielding coefficients, wind speed profile with height, building orientation, debris damage parameters, and component masses.

Then, for progressive gust wind speed increments, it calculates the forces in all critical connections using influence coefficients, assesses which connections have failed and translates these into a damage scenario and costs the repair. Using the repair cost and the full replacement cost, it calculates a damage index for each wind speed.

For more details, see  :ref:`Chapter 5 <logic>`.

Key features
============

* Component-based approach:

  A house is modelled consisting of a large number of components, and overall damage is estimated based on damage to each of the components.

* Uncertainty captured through a Monte-Carlo process:

  Various uncertainties affecting house performance are modelled through a monte-carlo process.

* Inclusion of debris and water ingress induced damages:

  In addition to the damage to the connections by wind loads, debris and water ingress induced damages are modelled.

* Internal pressurisation:

  Internal pressure coefficients are calculated at each wind speed following the procedures of AS/NZS 1170.2 (Standards Australia, 2011) using the modelled envelope failures to determine envelope permeability.


Key uncertainties
=================

The Monte Carlo process capture a range of variability in both wind loading and component parameters. The parameter values are sampled for each model and kept the same through the wind steps.

- Wind direction

  For each house, its orientation with respect to the wind is chosen from the eight cardinal directions either randomly, or by the user.

- Gust wind profile

  Variation in the profile of wind speed with height is captured by the random sampling of a profile from a suite of user-provided profiles.

- Pressure coefficients for zones and coverages

  Pressure coefficients for different zones of the house surfaces envelope are randomly chosen from a Type III (Weibull) extreme value distribution with specified means for different zones of the house envelope, and specified coefficients of variation for different load effects.

- Strength and dead load

  Connection strengths and dead loads for generated houses are sampled from lognormal probability distributions.

Caveats and limitations
=======================

VAWS has been designed primarily as a tool for assessing vulnerability of houses to wind hazard. The simulation outcomes should be interpreted as vulnerability of a group of similar houses on average, even though an individual house is modelled. In other words, the tool is not capable of predicting performance of an individual house for a specific wind event.
