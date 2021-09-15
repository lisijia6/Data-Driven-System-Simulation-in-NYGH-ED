# Code Descriptions


## 1.0 Queueing Simulators

#### [q_intervention.py](https://github.com/lisijia6/Data-Driven_System_Simulation_in_NYGH_ED/blob/main/4_synthetic_data_simulators/0%20Simulators/q_intervention.py)
* Original developer: [Arik Senderovich](https://github.com/ArikSenderovich1983/simulation_tmle/tree/master)
* Modified from code by original developer:
    1. Adjustable parameters: λ, classes, P(arrival_class), μ, P(speedup), n (# servers), laplace_params (location & scale for G/G/1 with appointments) 
    2. FCFS (first-come-first-served) --> M/G/1 and G/G/1 with appointments (arrivals of next customer is independent of the punctuality of the previous customer) can be simulated
    3. Data can be generated and written into CSV files, recording: 
          * **data_NIQ** [run#, timestamp, class_id, NIQ]
          * **data_NIS** [run#, timestamp, class_id, NIS]
          * **data_WIQ_TIS** [id_run, arrival_time (of customer), timestamp (of event), event_type (either d=departure or s=service_start), C (class#), A (intervention or not), elapsed (time passed since arrival)]


#### [q_priority.py](https://github.com/lisijia6/Data-Driven_System_Simulation_in_NYGH_ED/blob/main/4_synthetic_data_simulators/0%20Simulators/q_priority.py)
* Similar to FCFS queue implemented in q_intervention other than modifying how the heap is managed for queue (now the arrivals in the queue is sorted based on 1. priority, 2. arrival time). The event_calendar heap also keeps track of one additional value in each tuple -- priority of the customer.
* The priority queue assumes that customer already in service cannot be interrupted when a customer with a higher priority arrives. 


#### [simulation_runs.py](https://github.com/lisijia6/Data-Driven_System_Simulation_in_NYGH_ED/blob/main/4_synthetic_data_simulators/0%20Simulators/simulation_runs.py)
* "Grid Runs"
* Calculate performance measures by reading in from CSV data files _(NOTE: CALCULATION NEED TO BE ADJUSTED TO REFLECT CHANGES IN THE TRACKERS' DATA STRUCTURE IN q_intervention)_



## 2.0 Plotting
#### [plot_workload.py](https://github.com/lisijia6/Data-Driven_System_Simulation_in_NYGH_ED/blob/main/4_synthetic_data_simulators/0%20Simulators/plot_workload.py) and [resource.py](https://github.com/lisijia6/Data-Driven_System_Simulation_in_NYGH_ED/blob/main/4_synthetic_data_simulators/0%20Simulators/resource.py)
* Entirely from original developer: [Arik Senderovich](https://github.com/ArikSenderovich1983/simulation_tmle/tree/master)

#### [plotting.py](https://github.com/lisijia6/Data-Driven_System_Simulation_in_NYGH_ED/blob/main/4_synthetic_data_simulators/0%20Simulators/plotting.py)
* **SIMULATION (FCFS & Non-preemptive Priority)**
    1. Plotting expected values and 90th percentiles with 95% CI (confidence level adjustable)
    2. For k=1 class, fixed λ=1, μ=1.1, μ'= [2, 2.5, 3], P(intervention)=[0, 0.1, ..., 1.0] --> To simulate for k classes, additional variables can be added if required.
    3. Four performance measures: **TIS (time in system), WIQ (wait time in queue), NIS (number in system), NIQ (number in queue)**
    4. Supports generating plots for all classes as well as for individual classes

* **EXACT ANALYSIS**
    1. Single Class FCFS (first-come-first-served): Expected Values and 90th Percentiles (TIS, WIQ, NIS, NIQ)
    2. Non-preemptive Priority (Two Classes): Expected Values (TIS, WIQ, NIS, NIQ)

#### [plotting_all_results.py](https://github.com/lisijia6/Data-Driven_System_Simulation_in_NYGH_ED/blob/main/4_synthetic_data_simulators/0%20Simulators/plotting_all_results.py)
* Compare results from 4 methods on single plot (simulation, exact analysis, parametric, non-parametric) --> TIS, WIQ --> single class FCFS



