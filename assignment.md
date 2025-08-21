
1

Automatic Zoom
Senior ML Engineer – Home Assignment 
Objective 
The goal of this assignment is to design and implement an end-to-end machine learning pipeline for a binary classification 
problem using large-scale synthetic clinical data. You will simulate realistic patient-level data in a patient–year format, prepare 
temporal features through rolling aggregations during data preparation, and address the challenge of uncertainty in diagnosis 
year at the modeling/evaluation stage. The task should demonstrate your ability to tackle data volume and memory constraints, 
including applying feature selection or reduction where necessary to keep the pipeline efficient. Your solution should be 
modular, reproducible, and include explicit decision threshold selection for converting probabilities into binary outcomes. 
Finally, you are expected to deploy the trained model as a prediction endpoint, conduct training-related error analysis, and 
ensure that both the training pipeline and the serving endpoint run inside Docker containers. 
 
Assignment Details 
• Submission deadline: Within 7 days of receiving the task. 
• Cloud: Your choice. 
• AI tools: You may use any AI coding assistants. 
• Note: The data volume and compute requirements are designed to fit comfortably within free-tier limits of major cloud 
providers. You may parameterize dataset size to ensure free-tier feasibility, as long as the design clearly scales to the 
target volume. 
• Submission: Share a link to a GitHub repository containing your full solution. The repository must include all source 
code, configuration files, and Dockerfiles needed to run both the training pipeline and the serving endpoint in 
containers. 
• Follow-up interview: You will be asked to present your solution deployed to your chosen cloud platform (live demo of 
the running endpoint and a short walkthrough of the training pipeline setup). 
 
Core Requirements 
1. Data Generation – Clinical Simulation & Efficient Processing 
• Simulate synthetic clinical-like tabular data with: 
o One row per patient per calendar year (patient–year format). 
o Couple of million rows (runtime target; code should scale beyond this). 
o At least 20 features (numerical + categorical). 
o 5–10% positive prevalence (imbalanced outcome). 
• Store efficiently (partitioning, chunking, columnar formats). 
• Design the data to push memory limits, requiring feature selection/reduction later in the pipeline. 
 
2. ML Training Pipeline 
• Modular, re-runnable pipeline that: 
o Efficiently ingests data (avoid unnecessary full in-memory loads). 
o Prepares features, including: 
▪ Temporal rolling aggregations from previous X years (justify your choice of X). 
▪ Standard preprocessing (casting, missingness, encoding). 
o Handles imbalance (class weights, resampling, or other). 
o Considers diagnosis year uncertainty as a modeling/evaluation problem. 
o Applies feature selection/reduction under memory pressure. 
o Trains a binary classifier. 
o Selects and justifies a decision threshold for probability-to-label conversion. 
o Evaluates model performance with appropriate metrics. 
o Saves a versioned model artifact. 
• Must be runnable in Docker. 
 
3. Experiment Tracking 
• Log hyperparameters, metrics, and artifacts (e.g., model files, plots). 
 
4. Serving Endpoint 
• Provide a prediction endpoint that: 
o Accepts JSON input. 
o Returns predicted probability and class. 
• Must be runnable in Docker. 
 
5. Training-Related Error Analysis 
• Provide a short notebook/Markdown report with: 
o Slice analysis: compare metrics across 2–3 categorical subgroups. 
o Likely error causes (e.g., label uncertainty, subgroup performance) and at least one proposed improvement. 
 
6. Code Quality, Dependency Management & Containerization 
• Clean, maintainable, reproducible code. 
• Provide dependency setup. 
• Include Dockerfile’s for training and serving; optionally docker-compose.yml. 
• README.md must describe: 
o Architecture overview. 
o Setup/run instructions (local, cloud, Docker). 
o Key code areas. 
o How to reproduce an end-to-end run in Docker. 
 
Bonus Points (Optional) 
• Parallel or distributed data processing (e.g., Dask, Spark, Ray). 
• Distributed model training (multi-core/distributed-capable learners). 
• Testability: unit tests, model versioning support, and A/B rollout. 
• Support units of measure for clinical fields, including handling unit incompatibilities and conversions (e.g., mg/dL vs. 
mmol/L, height in cm vs. inches). 