[![Build Status](https://travis-ci.com/GeoscienceAustralia/wistl.svg?token=r6qYcaeJV4Tb19SjCSya&branch=master)](https://travis-ci.com/GeoscienceAustralia/wistl)

# WISTL

WIND IMPACT SIMULATION ON TRANSMISSION LINES (WISTL) is a software tool for assessing transmission line vulnerability based on a Monte Carlo simulation. It simulates two different damage mechanisms: damage due to direct wind assuming towers are isolated, and cascading failure mechanisms where damage is induced by the collapse of adjacent towers. It produces a damage probability for each damage state by tower and probability distribution of the number of damaged towers by line.

# Overall logic

The primary mechanism for wind damage to transmission towers is the direct action of wind on a tower and conductors in severe wind events such as tropical cyclones. The secondary mechanism of damage is the collapse of neighbouring towers with the transfer of load to adjacent towers due to unbroken conductors. Collapse of one tower leads to additional pulling load on neighbouring towers connected to it in addition to wind loads. A cluster of towers can fail from the initiation of a primary failure due to wind with the subsequent pull down of neighbouring towers. This phenomenon is referred to as a cascading failure.
Damage due to the direct action of wind was modelled by defining tower fragility functions. The fragility functions are defined with respect to the type of tower based on functionality.
The cascading failure in the event of a primary tower failure was modelled by defining conditional probabilities for potential collapse patterns. Different sets of conditional probabilities were defined with respect to the function of the primary collapse tower.

# Documentation

User manual in html format at GitHub pages: http://geoscienceaustralia.github.io/wistl 

