# Data-Driven System Simulation for NYGH Emergency Department

**Repository author:** Sijia (Nancy) Li

**Supervised by:** Prof. Arik Senderovich, Prof. Opher Baron, and Prof. Dmitry Krass


**Table of Contents**

[1.0 Background](#10-background)

[2.0 Pipeline](#20-pipeline)

[3.0 Other Useful Code](#30-other-useful-code)

[4.0 Examples of Using the Pipeline](#40-examples-of-using-the-pipeline)

[5.0 Limitations of Current Work](#50-limitations-of-current-work)

[6.0 Possible Future Improvements to GitHub Code](#60-possible-future-improvements-to-github-code)

[7.0 Contact](#70-contact)


## 1.0 Background
A queueing system can be described as a system where customers arrive for service and a queue develops when there are more customers in the system than the number of servers. All queueing systems will have an arrival process, a service and departure process, and a number of servers. Simulation, although not exact, can be used to study queueing system behaviours and serve as the “digital twin” of complex real-life systems. 

This is a project in the area of healthcare that focuses on data-driven system simulation in hospital emergency department (ED). Our interest is to study the effect of congestions in the ED system, as hospital EDs can encounter challenges such as limited resource capacity and long patient sojourn time. We developed an automated pipeline consisting of three main steps:
1. **Get user specification:** ED system context, congestion representation by system state, interventions
2. **Construct ED model:** a) automatically learn patient length of stay (LOS) model from data, and b) apply interventions during simulation
3. **Apply diagnostics and analytics:** a) evaluate goodness of fit of LOS model, and b) study the impact of introducing interventions to the system

The data used for this project is from North York General Hospital (NYGH) ED and is not shared on GitHub due to confidentiality. 



## 2.0 Pipeline
The automated data-drive pipeline is implemented in Python. The code for the pipeline can be found in the folder **"1_pipeline"**. The folder contains 5 Python files:
* **nygh_main_final.py**
* **nygh_pre_process_final.py**
* **nygh_simulation_final.py**
* **nygh_transform_results_final.py**
* **nygh_histogram_qqplots_kstest_final.py**

_**nygh_main_final.py**_ is the first file to read, as it is the file to use for running the pipeline. It is the main interface for asking the user to enter user specifications. It also ties together the other 4 Python files in the folder to complete the overall pipeline. In particular, the high-level steps that this file goes through are the following:
0. Asks for user inputs
1. Reads in data (1a: raw data to pre-process and read, OR 1b: cleaned data to read)
2. Splits data into training and testing sets
3. Builds patient length of stay (LOS) models, simulates the system, computes performance measures, and saves simulation results to file
4. Transforms simulation results into a format easily readable by Python and storable as pandas DataFrames
5. Constructs relative frequency histograms and Q-Q plots, and performs Kolmogorov–Smirnov tests

In Step 0, the "nygh_main" file collects inputs for Steps 1 to 5. 

* **Step 1** is conducted by either 1) reading in raw data and calling the *main\_preprocess\_data* function in  *nygh\_pre\_process\_final.py* to pre-process the data, or 2) directly reading in from file


* **Steps 2, 3, and 4** are executed together. 
    
    * **For Step 2**, user specifies the start and end (year, month) for training and testing data. The *select\_df\_year\_month* function in _**nygh\_simulation\_final.py**_ is used to filter for the data of interest and divide the data of interest into training and testing sets. 
    
    * **For Step 3**, the inputs to execute this step are the overall data of interest, the training data, and testing data, along with user specifications of congestion representations (system states) and interventions (% to cut down consult patients' length of stay). The *main\_model\_simulation\_performance* function in _**nygh\_simulation\_final.py**_ automatically:
        1. **Selects training features** according to context (e.g., patient static information, season, trend, holidays) and system state
        2. **Trains LOS models for each patient type** -- 3 patient types were defined: _**T123 Admitted**_ (patients with triage codes 1, 2, and 3, and are admitted to the hospital after ED visit), _**T123 Not Admitted**_ (patients with triage codes 1, 2, and 3, and are NOT admitted to the hospital after ED visit), and _**T45**_ (all patients with triage codes 4 and 5)
        3. **Simulates the system** for a given number of replications (n runs), and keeps track of key statistics along the way and saves patient length of stay (LOS) for all replications in Excel files
        4. **Computes key performance measures** including mean, median, standard deviation, 90th percentile, 95% confidence interval for the mean, and RMSE. 
        5. **Repeats all previous sub-steps** for all specified system states and level of interventions (0%, 10%, 20%, 30%, 40%, or 50% consult patient LOS cut down)
        6. **Saves performance measures results in Excel files**
    
    * **For Step 4**, the *main\_transform\_simulation\_results* function in  _**nygh\_transform\_results\_final.py**_ automatically transforms the results saved in the results file into a cleaner format (containing information for mean, median, and 90th percentile) that is easily readable as pandas DataFrames. 


* **Step 5** validates LOS models by examining the LOS distributions obtained in simulation (histograms & Q-Q plots), and performing K-S tests (K-S statistics). The _**nygh\_histogram\_qqplots\_kstest\_final.py**_ automatically reads in simulation data, constructs histograms and Q-Q plots, and saves all the plots to files. In addition, K-S tests are also performed along the way to compare LOS distribution from actual data to LOS distribution from simulated data. The K-S test results are also saved in files.



## 3.0 Other Useful Code
In the folder **"2_other"** contains some code that are not part of the pipeline, but related to the work. 
* **nygh\_plotting\_clustered\_bars\_final.py** can be modified to read in data and create clustered bar charts


## 4.0 Examples of Using the Pipeline
Four examples of running the pipeline by selecting different training and testing ranges are included in the folder **"3_pipeline_examples"**. In particular, below will walk through the first example in the folder **"pipeline_example_1..."**.

Inside the **"pipeline_example_1..."** folder, the file **"00_pipeline_example_1_terminal_output.txt"** is the first file to read. In the first part, it shows the answers to the question for this particular use of the pipeline. Corresponding to **Step 0**, all inputs to the pipeline are collected at the beginning. After the user enters all the inputs to the pipeline, the pipeline automatically goes through **Steps 1 to 5** in the pipeline. The second part of the terminal output shows some cues that were given to the user in the console as the pipeline is running. 

Going back to other files that were generated during the execution of the pipeline. The files can be categorized as follows (in order in which they are generated in the pipeline):

NOTE: n replications/runs = 2, system states = [0, 1], intervention = [0.5, 1.0] (corresponding to 50% consult LOS cut down, 0% consult LOS cut down/no intervention)

1. **Actual LOS Data by Patient Type**
    * _**LOS\_Dist\_Actual\_Patient\_Type=T123A.csv**_ contains all the LOS of the T123 admitted patients for n replications of simulation. 
    * _**LOS\_Dist\_Actual\_Patient\_Type=T123NA.csv**_ contains all the LOS of the T123 not admitted patients for n replications of simulation. 
    * _**LOS\_Dist\_Actual\_Patient\_Type=T45.csv**_ contains all the LOS of the T45 patients for n replications of simulation. 

2. **Simulated LOS Data by System States**
    In this pipeline example, 2 system states were selected (system state 0 and system state 1), therefore, 2 of these Excel files are created. Each of these Excel files contain 3 tabs, one for each patient type. 
    * _**LOS\_Dist\_Simulated\_System\_State=0.xlsx**_ contains all the LOS of the 3 types of patients for n replications of simulation, LOS models were built using the system state 0 congestion representation feature. 
    * _**LOS\_Dist\_Simulated\_System\_State=1.xlsx**_ contains all the LOS of the 3 types of patients for n replications of simulation, LOS models were built using the system state 1 congestion representation features. 

3. _**00\_Naive+Simulation\_Results.xlsx**_, contains 3 tabs:
    * "Naive" results (directly cutting down consult patients' LOS from data)
    * "System\_State\_0" results
    * "System\_State\_1" results

    Each tab contains columns of: number of patients
    (by patient type), number of consult patients (by patient type), % of consult (by patient type), consult LOS cut down percentage, expected mean (by patient type), expected median (by patient type), expected standard deviation (by patient type), and expected 90th percentile (by patient type). 

4. _**00\_mean_median_90P\_results.xlsx**_
    This file contains a cleaner view of expected mean, median, and 90th percentile results from previous file.

5. **Histograms (Relative Frequency)**
    *  _**Actual vs. Simulated LOS Relative Frequency (T123 Admitted).png**_
    *  _**Actual vs. Simulated LOS Relative Frequency (T123 Not Admitted).png**_
    *  _**Actual vs. Simulated LOS Relative Frequency (T45).png**_
    
    Each plot on the image overlays the relative frequency of LOS from simulated model onto that of LOS from actual data.

6. _**00\_KS\_test\_results.xlsx**_, contains the two-sample K-S test (actual vs. simulated LOS distributions) results.

7. **Q-Q Plots**
    * _**Q-Q Plot (T123 Admitted).png**_
    * _**Q-Q Plot (T123 Not Admitted).png**_
    * _**Q-Q Plot (T45).png**_
    
    Produces Q-Q plots of actual LOS quantiles vs. simulated LOS quantiles.

___
In this example, we also included an example of plotting a clustered bar chart, where we used the code from the file **nygh\_plotting\_clustered\_bars\_final.py** in the folder **"2_other"**.
* The file _**00\_clustered\_bar\_example.xlsx**_ contains some partial results from pipeline\_example\_1 outputs. When constructing clustered bar plots, the code reads in this Excel file and automatically produces and saves the clustered bar plot in the file _**Mean LOS values (T123 Admitted Patients).png**_. 
___ 


## 5.0 Limitations of Current Work
* Service process of the system is a **"black-box"** process that starts from triage to leaving ED (lacked timestamps in between)
* Representation of congestion ("system state") is **highly simplified**, by counts of patients in the ED


## 6.0 Possible Future Improvements to GitHub Code
1. **More generalizations on data pre-processing (current data pre-processing is specific to the raw data)**, e.g., feature selection, patient type categorization
2. **Integration of intervention analysis results to the pipeline**, e.g., plots of system LOS reduction vs. level of intervention
3. **Simplification of file inputs and outputs**, e.g., reduce number of intermediate results files generated in the pipeline, improve the organization of results produced during the simulation



## 7.0 Contact
Repository author: sijianancy.li@mail.utoronto.ca

Thank You!
