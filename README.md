# Data-driven System Simulation for NYGH Emergency Department

**Repository author:** Sijia (Nancy) Li

**Supervised by:** Prof. Arik Senderovich, Prof. Opher Baron, and Prof. Dmitry Krass

## Background
A queueing system can be described as a system where customers arrive for service and a queue develops when there are more customers in the system than the number of servers. All queueing systems will have an arrival process, a service and departure process, and a number of servers. Simulation, although not exact, can be used to study queueing system behaviours and serve as the “digital twin” of complex real-life systems. 

This is a project in the area of healthcare that focuses on data-driven system simulation in hospital emergency department (ED). Our interest is to study the effect of congestions in the ED system, as hospital EDs can encounter challenges such as limited resource capacity and long patient sojourn time. We developed an automated pipeline consisting of three main steps:
1. **Get user specification:** ED system context, congestion representation by system state, interventions
2. **Construct ED model:** a) automatically learn patient length of stay (LOS) model from data, and b) apply interventions during simulation
3. **Apply diagnostics and analytics:** a) evaluate goodness of fit of LOS model, and b) study the impact of introducing interventions to the system
