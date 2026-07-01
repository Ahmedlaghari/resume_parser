# ============================================================
# question_bank.py — Static interview question templates
#
# These are the raw questions stored in the vector DB.
# The LLM in generator.py personalizes them after retrieval
# by filling in [SKILL], [COMPANY], [PROJECT] placeholders
# with the actual candidate's details.
#
# Schema per question:
#   id               → unique string (used as ChromaDB document ID)
#   text             → the question as it goes into the vector DB
#                      (complete sentence — embeddings work on real text)
#   category         → technical | experience | gap | behavioral
#   skill_tags       → list of skills/topics this question targets
#                      (used for metadata filtering during retrieval)
#   experience_level → entry | mid | senior | any
#   what_to_listen_for → recruiter guidance shown in final output
# ============================================================

QUESTION_BANK = [

    # ================================================================
    # CATEGORY: technical
    # Tests depth on skills the candidate claims to have.
    # ================================================================

    {
        "id": "tech_001",
        "text": "You have [SKILL] on your resume — can you walk me through a real problem you solved with it and what alternatives you considered?",
        "category": "technical",
        "skill_tags": ["general", "technical_depth"],
        "experience_level": "any",
        "what_to_listen_for": "Concrete problem, not textbook definition. Do they know the tradeoffs vs alternatives?"
    },
    {
        "id": "tech_002",
        "text": "Explain the difference between overfitting and underfitting. How did you detect and fix each in a real model you built?",
        "category": "technical",
        "skill_tags": ["machine_learning", "model_training", "deep_learning"],
        "experience_level": "any",
        "what_to_listen_for": "Concrete fix strategies: regularization, more data, simpler model, early stopping. Not just definitions."
    },
    {
        "id": "tech_003",
        "text": "Walk me through how you would build a training pipeline for a classification model from raw data to a deployed endpoint.",
        "category": "technical",
        "skill_tags": ["machine_learning", "mlops", "pipeline", "model_deployment"],
        "experience_level": "mid",
        "what_to_listen_for": "Do they mention data validation, feature engineering, experiment tracking, versioning, monitoring after deploy?"
    },
    {
        "id": "tech_004",
        "text": "What is your experience with PyTorch? Describe a specific neural network you built with it — architecture choices and why.",
        "category": "technical",
        "skill_tags": ["pytorch", "deep_learning", "neural_networks"],
        "experience_level": "mid",
        "what_to_listen_for": "Layer choices, activation functions, optimizer selection, loss function reasoning."
    },
    {
        "id": "tech_005",
        "text": "How does TensorFlow's computation graph differ from PyTorch's eager execution? When would you prefer one over the other?",
        "category": "technical",
        "skill_tags": ["tensorflow", "pytorch", "deep_learning"],
        "experience_level": "senior",
        "what_to_listen_for": "Understands static vs dynamic graphs, deployment implications (TF serving vs TorchScript)."
    },
    {
        "id": "tech_006",
        "text": "Describe how you have used Python's multiprocessing or async features to speed up a data pipeline or model inference.",
        "category": "technical",
        "skill_tags": ["python", "performance", "concurrency", "pipeline"],
        "experience_level": "mid",
        "what_to_listen_for": "GIL awareness, when to use multiprocessing vs threading vs async, real speedup numbers if possible."
    },
    {
        "id": "tech_007",
        "text": "Walk me through how a transformer-based model like BERT processes a sentence from raw text to an output embedding.",
        "category": "technical",
        "skill_tags": ["nlp", "transformers", "bert", "deep_learning"],
        "experience_level": "mid",
        "what_to_listen_for": "Tokenization, positional encoding, self-attention heads, [CLS] token usage, fine-tuning vs frozen."
    },
    {
        "id": "tech_008",
        "text": "How have you handled class imbalance in a dataset? Walk me through the specific techniques you used and how you measured improvement.",
        "category": "technical",
        "skill_tags": ["machine_learning", "data_preprocessing", "classification"],
        "experience_level": "any",
        "what_to_listen_for": "SMOTE, class weights, oversampling/undersampling, precision-recall tradeoff, not just accuracy as metric."
    },
    {
        "id": "tech_009",
        "text": "Explain how you have used Docker in your workflow. What does your Dockerfile typically look like for a Python ML project?",
        "category": "technical",
        "skill_tags": ["docker", "mlops", "deployment", "devops"],
        "experience_level": "any",
        "what_to_listen_for": "Multi-stage builds, layer caching, non-root user, pinned versions, entrypoint vs cmd."
    },
    {
        "id": "tech_010",
        "text": "What SQL window functions have you used, and when would you reach for a window function over a subquery or GROUP BY?",
        "category": "technical",
        "skill_tags": ["sql", "databases", "data_engineering"],
        "experience_level": "any",
        "what_to_listen_for": "ROW_NUMBER, RANK, LAG/LEAD, running totals. Should understand when aggregation loses row-level detail."
    },
    {
        "id": "tech_011",
        "text": "How do you approach feature engineering for a tabular dataset? What signals tell you a feature is worth keeping?",
        "category": "technical",
        "skill_tags": ["machine_learning", "feature_engineering", "data_science"],
        "experience_level": "any",
        "what_to_listen_for": "Correlation, mutual information, feature importance from tree models, avoiding leakage."
    },
    {
        "id": "tech_012",
        "text": "Describe your experience with Kubernetes. How have you used it to deploy or scale a model serving endpoint?",
        "category": "technical",
        "skill_tags": ["kubernetes", "mlops", "deployment", "devops"],
        "experience_level": "senior",
        "what_to_listen_for": "Pods, deployments, services, HPA for scaling, readiness probes, resource limits."
    },
    {
        "id": "tech_013",
        "text": "How have you built or consumed REST APIs in your projects? Describe the most complex API you designed and why you structured it that way.",
        "category": "technical",
        "skill_tags": ["api", "fastapi", "django", "flask", "backend"],
        "experience_level": "any",
        "what_to_listen_for": "Versioning, auth, pagination, error responses, documentation (OpenAPI), rate limiting awareness."
    },
    {
        "id": "tech_014",
        "text": "How do you evaluate an NLP model beyond accuracy? What metrics do you use and what do they actually tell you?",
        "category": "technical",
        "skill_tags": ["nlp", "machine_learning", "evaluation"],
        "experience_level": "mid",
        "what_to_listen_for": "BLEU, ROUGE, F1, perplexity, human evaluation. Should distinguish task-type appropriateness."
    },
    {
        "id": "tech_015",
        "text": "Walk me through how you have used Redis or another caching layer to improve performance in a system you built.",
        "category": "technical",
        "skill_tags": ["redis", "caching", "backend", "performance"],
        "experience_level": "mid",
        "what_to_listen_for": "Cache invalidation strategy, TTL choices, what specifically they cached and measurable impact."
    },
    {
        "id": "tech_016",
        "text": "How does backpropagation work? Can you trace the gradient flow for a two-layer network by hand?",
        "category": "technical",
        "skill_tags": ["deep_learning", "neural_networks", "machine_learning"],
        "experience_level": "senior",
        "what_to_listen_for": "Chain rule, vanishing gradient problem, why certain activations (ReLU) help, gradient clipping."
    },
    {
        "id": "tech_017",
        "text": "What is your experience with experiment tracking tools like MLflow or Weights & Biases? What specifically do you log and why?",
        "category": "technical",
        "skill_tags": ["mlops", "mlflow", "experiment_tracking", "machine_learning"],
        "experience_level": "mid",
        "what_to_listen_for": "Hyperparameters, metrics per epoch, artifacts, reproducibility, model registry usage."
    },
    {
        "id": "tech_018",
        "text": "Explain how you have used vector embeddings in a real project — what model generated them and how did you search them?",
        "category": "technical",
        "skill_tags": ["embeddings", "nlp", "vector_database", "semantic_search"],
        "experience_level": "mid",
        "what_to_listen_for": "Embedding model choice, cosine vs dot product similarity, approximate nearest neighbor (FAISS, ChromaDB)."
    },
    {
        "id": "tech_019",
        "text": "How do you handle data leakage when building a machine learning model? Give a concrete example where leakage was a risk.",
        "category": "technical",
        "skill_tags": ["machine_learning", "data_science", "model_training"],
        "experience_level": "any",
        "what_to_listen_for": "Target encoding before split, future data in time-series, train/validation/test contamination."
    },
    {
        "id": "tech_020",
        "text": "What cloud services have you used on AWS or GCP for machine learning workloads — storage, compute, and serving?",
        "category": "technical",
        "skill_tags": ["aws", "gcp", "cloud", "mlops"],
        "experience_level": "any",
        "what_to_listen_for": "S3/GCS for data, EC2/GCE or SageMaker/Vertex for training, Lambda/Cloud Functions for inference. Awareness of cost."
    },
    {
        "id": "tech_021",
        "text": "Walk me through how you have used Python generators or lazy evaluation to handle datasets that don't fit in memory.",
        "category": "technical",
        "skill_tags": ["python", "data_engineering", "performance", "big_data"],
        "experience_level": "mid",
        "what_to_listen_for": "yield keyword, itertools, PyTorch DataLoader with batch loading, chunked reads with pandas."
    },
    {
        "id": "tech_022",
        "text": "How have you implemented model monitoring in production? What drift do you detect and how do you alert on it?",
        "category": "technical",
        "skill_tags": ["mlops", "model_deployment", "monitoring", "production"],
        "experience_level": "senior",
        "what_to_listen_for": "Data drift vs concept drift, statistical tests (KS test, PSI), threshold-based alerts, retraining triggers."
    },
    {
        "id": "tech_023",
        "text": "Describe a time you optimized a SQL query or database schema that was causing performance problems. What was the root cause?",
        "category": "technical",
        "skill_tags": ["sql", "databases", "performance", "data_engineering"],
        "experience_level": "mid",
        "what_to_listen_for": "EXPLAIN/ANALYZE output, missing indexes, N+1 queries, denormalization tradeoffs."
    },
    {
        "id": "tech_024",
        "text": "How do attention mechanisms work in transformers, and why were they an improvement over RNNs for sequence tasks?",
        "category": "technical",
        "skill_tags": ["nlp", "transformers", "deep_learning", "attention"],
        "experience_level": "senior",
        "what_to_listen_for": "Vanishing gradient in RNNs, O(n²) attention complexity, parallelization advantage, long-range dependencies."
    },
    {
        "id": "tech_025",
        "text": "You listed [SKILL] — how would you use it to solve a specific type of problem that comes up in this role?",
        "category": "technical",
        "skill_tags": ["general", "technical_application"],
        "experience_level": "any",
        "what_to_listen_for": "Can they apply the skill to business problems, not just recite features."
    },
    {
        "id": "tech_026",
        "text": "What is your approach to writing unit tests for machine learning code? What do you actually test vs what do you skip?",
        "category": "technical",
        "skill_tags": ["testing", "python", "machine_learning", "software_engineering"],
        "experience_level": "mid",
        "what_to_listen_for": "Data shape tests, deterministic seed tests, mock for expensive operations, not testing model accuracy."
    },
    {
        "id": "tech_027",
        "text": "How have you used Git in a team environment — branching strategy, PR process, and handling merge conflicts?",
        "category": "technical",
        "skill_tags": ["git", "version_control", "software_engineering", "collaboration"],
        "experience_level": "any",
        "what_to_listen_for": "Feature branching, PR review habits, rebasing vs merging, protecting main branch."
    },
    {
        "id": "tech_028",
        "text": "What is transfer learning and how have you applied it? What did you freeze and what did you fine-tune, and why?",
        "category": "technical",
        "skill_tags": ["deep_learning", "transfer_learning", "nlp", "computer_vision"],
        "experience_level": "mid",
        "what_to_listen_for": "Frozen early layers, learning rate scheduling for fine-tuning, catastrophic forgetting, dataset size considerations."
    },
    {
        "id": "tech_029",
        "text": "Describe the most complex data pipeline you have built. How did you handle failures, retries, and data quality checks?",
        "category": "technical",
        "skill_tags": ["data_engineering", "pipeline", "airflow", "etl"],
        "experience_level": "mid",
        "what_to_listen_for": "Idempotency, dead-letter queues, schema validation, alerting, at-least-once vs exactly-once delivery."
    },
    {
        "id": "tech_030",
        "text": "How do you approach hyperparameter tuning? What tools or strategies have you used beyond manual trial and error?",
        "category": "technical",
        "skill_tags": ["machine_learning", "hyperparameter_tuning", "optuna", "model_training"],
        "experience_level": "any",
        "what_to_listen_for": "Grid search vs random search vs Bayesian optimization, early stopping in search, Optuna/Ray Tune."
    },
    {
        "id": "tech_031",
        "text": "How have you implemented or consumed OAuth or JWT-based authentication in a backend service?",
        "category": "technical",
        "skill_tags": ["authentication", "security", "backend", "api"],
        "experience_level": "mid",
        "what_to_listen_for": "Token expiry, refresh tokens, storing tokens safely, role-based access, not storing plain passwords."
    },
    {
        "id": "tech_032",
        "text": "What is your experience with computer vision tasks? Describe a specific model you built or fine-tuned and the dataset you used.",
        "category": "technical",
        "skill_tags": ["computer_vision", "deep_learning", "image_classification", "object_detection"],
        "experience_level": "mid",
        "what_to_listen_for": "CNN architecture choice, data augmentation strategy, evaluation metric appropriate to task (mAP, IoU)."
    },
    {
        "id": "tech_033",
        "text": "How have you used Pandas or NumPy for large-scale data manipulation? What were the performance bottlenecks and how did you fix them?",
        "category": "technical",
        "skill_tags": ["python", "pandas", "numpy", "data_engineering", "performance"],
        "experience_level": "any",
        "what_to_listen_for": "Vectorization over loops, chunked reads, dtype optimization, Dask for out-of-memory, avoiding apply()."
    },
    {
        "id": "tech_034",
        "text": "Explain how you have designed or used a CI/CD pipeline for a machine learning project. What stages did it include?",
        "category": "technical",
        "skill_tags": ["mlops", "ci_cd", "devops", "automation"],
        "experience_level": "senior",
        "what_to_listen_for": "Linting, tests, model evaluation gate, Docker build, staging deploy, canary/blue-green rollout."
    },
    {
        "id": "tech_035",
        "text": "How do you handle missing data in a dataset? Walk me through the decision process for choosing an imputation strategy.",
        "category": "technical",
        "skill_tags": ["machine_learning", "data_preprocessing", "data_science"],
        "experience_level": "any",
        "what_to_listen_for": "MCAR/MAR/MNAR distinction, mean vs median vs mode vs model-based imputation, when to drop."
    },

    # ================================================================
    # CATEGORY: experience
    # Probes specific jobs and projects from the candidate's resume.
    # ================================================================

    {
        "id": "exp_001",
        "text": "Walk me through the most technically complex project you have worked on. What was your specific contribution and what would you do differently now?",
        "category": "experience",
        "skill_tags": ["general", "project_depth", "ownership"],
        "experience_level": "any",
        "what_to_listen_for": "Clear ownership, specific technical decisions, honest reflection — not just a feature description."
    },
    {
        "id": "exp_002",
        "text": "Tell me about the [PROJECT] project on your resume. What problem did it solve, and what were the biggest technical decisions you made?",
        "category": "experience",
        "skill_tags": ["project_depth", "technical_decisions"],
        "experience_level": "any",
        "what_to_listen_for": "Do they understand the business problem, not just the code? Can they defend their technical choices?"
    },
    {
        "id": "exp_003",
        "text": "At [COMPANY], what was the scale of the systems you worked on — users, data volume, request throughput?",
        "category": "experience",
        "skill_tags": ["scale", "production", "systems_design"],
        "experience_level": "mid",
        "what_to_listen_for": "Concrete numbers matter. Vague answers suggest they were not close to the production system."
    },
    {
        "id": "exp_004",
        "text": "What is the most impactful thing you built or shipped at [COMPANY]? How did you measure that impact?",
        "category": "experience",
        "skill_tags": ["impact", "ownership", "metrics"],
        "experience_level": "any",
        "what_to_listen_for": "Business metric tied to their work (latency reduced, revenue increased, errors dropped). Not just 'built feature X'."
    },
    {
        "id": "exp_005",
        "text": "Describe a technical decision you made at [COMPANY] that turned out to be wrong. What happened and how did you handle it?",
        "category": "experience",
        "skill_tags": ["accountability", "learning", "decision_making"],
        "experience_level": "any",
        "what_to_listen_for": "Ownership (no blaming others), specific decision, what they learned, how they communicated the mistake."
    },
    {
        "id": "exp_006",
        "text": "How did you structure your team's workflow at [COMPANY]? What was the collaboration model between you and other engineers or data scientists?",
        "category": "experience",
        "skill_tags": ["collaboration", "teamwork", "agile", "communication"],
        "experience_level": "mid",
        "what_to_listen_for": "Awareness of team dynamics, how they handled disagreements, cross-functional work."
    },
    {
        "id": "exp_007",
        "text": "Describe the NLP or ML pipeline you built at [COMPANY]. What did the data flow look like end to end?",
        "category": "experience",
        "skill_tags": ["nlp", "machine_learning", "pipeline", "data_engineering"],
        "experience_level": "mid",
        "what_to_listen_for": "Ingestion → preprocessing → training → evaluation → serving. Bottlenecks, monitoring, retraining cadence."
    },
    {
        "id": "exp_008",
        "text": "You have [X] years of experience in [SKILL]. What has changed in your approach to it over that time?",
        "category": "experience",
        "skill_tags": ["growth", "learning", "technical_depth"],
        "experience_level": "senior",
        "what_to_listen_for": "Genuine progression visible — not just using the skill longer but thinking about it differently."
    },
    {
        "id": "exp_009",
        "text": "Tell me about a time you had to optimize a slow model or data pipeline under production pressure. What was your debugging process?",
        "category": "experience",
        "skill_tags": ["performance", "debugging", "production", "mlops"],
        "experience_level": "mid",
        "what_to_listen_for": "Profiling first, hypothesis-driven approach, measuring before and after, not just guessing."
    },
    {
        "id": "exp_010",
        "text": "Describe a project where you had to design the architecture from scratch. What did you consider when making the key decisions?",
        "category": "experience",
        "skill_tags": ["systems_design", "architecture", "ownership"],
        "experience_level": "senior",
        "what_to_listen_for": "Trade-offs explicitly named: consistency vs availability, build vs buy, monolith vs services."
    },
    {
        "id": "exp_011",
        "text": "Have you ever onboarded a junior engineer or led a small team? How did you make sure they were productive and not blocked?",
        "category": "experience",
        "skill_tags": ["mentoring", "leadership", "communication"],
        "experience_level": "senior",
        "what_to_listen_for": "Structured onboarding, pairing, clear ownership for junior, availability without micromanaging."
    },
    {
        "id": "exp_012",
        "text": "What is the most interesting bug you have debugged in production? Walk me through how you found it.",
        "category": "experience",
        "skill_tags": ["debugging", "production", "problem_solving"],
        "experience_level": "any",
        "what_to_listen_for": "Systematic approach: logs, metrics, hypothesis, isolation, fix, post-mortem. Not just 'I found the line'."
    },
    {
        "id": "exp_013",
        "text": "How did you handle technical debt at [COMPANY]? Did you have time allocated for it or did you fit it in alongside feature work?",
        "category": "experience",
        "skill_tags": ["technical_debt", "software_engineering", "prioritization"],
        "experience_level": "mid",
        "what_to_listen_for": "Pragmatic view of debt — not just 'we always paid it off' (unrealistic) and not 'we ignored it' (red flag)."
    },
    {
        "id": "exp_014",
        "text": "Tell me about a project that required you to pick up a new tool or technology quickly. How did you get up to speed?",
        "category": "experience",
        "skill_tags": ["learning", "adaptability", "self_directed"],
        "experience_level": "any",
        "what_to_listen_for": "Deliberate learning strategy, not just 'I Googled it'. Docs, official examples, small proof of concept first."
    },
    {
        "id": "exp_015",
        "text": "Have you ever had to push back on a requirement or a timeline? How did you handle that conversation?",
        "category": "experience",
        "skill_tags": ["communication", "stakeholder_management", "professional_skills"],
        "experience_level": "mid",
        "what_to_listen_for": "Evidence-based pushback, alternative solution offered, not just 'I refused'. Respectful but clear."
    },
    {
        "id": "exp_016",
        "text": "Walk me through how you ensured data quality in a pipeline you built. What could go wrong and how did you detect it?",
        "category": "experience",
        "skill_tags": ["data_engineering", "data_quality", "pipeline", "testing"],
        "experience_level": "mid",
        "what_to_listen_for": "Schema validation, null checks, distribution monitoring, alerting on anomalies — not just 'we tested it'."
    },
    {
        "id": "exp_017",
        "text": "Describe a time you had to explain a complex ML model or system to a non-technical stakeholder. How did you approach it?",
        "category": "experience",
        "skill_tags": ["communication", "stakeholder_management", "presentation"],
        "experience_level": "any",
        "what_to_listen_for": "Analogies, focusing on outcomes not mechanics, checking for understanding, not dumbing down condescendingly."
    },
    {
        "id": "exp_018",
        "text": "What is the largest dataset you have worked with? How did you process it and what infrastructure did you use?",
        "category": "experience",
        "skill_tags": ["big_data", "data_engineering", "scale", "performance"],
        "experience_level": "mid",
        "what_to_listen_for": "Concrete size (GB/TB), distributed processing choice, storage format (Parquet, ORC), not just 'it was big'."
    },
    {
        "id": "exp_019",
        "text": "Tell me about a time a model you deployed underperformed in production. What did you do next?",
        "category": "experience",
        "skill_tags": ["mlops", "production", "model_monitoring", "problem_solving"],
        "experience_level": "mid",
        "what_to_listen_for": "Root cause analysis (distribution shift, bad features, labeling error), rollback plan, retraining approach."
    },
    {
        "id": "exp_020",
        "text": "How have your priorities and technical approach evolved between your role at [COMPANY] and your more recent work?",
        "category": "experience",
        "skill_tags": ["growth", "career_progression", "learning"],
        "experience_level": "mid",
        "what_to_listen_for": "Genuine reflection on how their thinking matured — not just 'I learned more tools'."
    },

    # ================================================================
    # CATEGORY: gap
    # Addresses skills missing from the candidate that the role requires.
    # ================================================================

    {
        "id": "gap_001",
        "text": "This role requires [SKILL], which I don't see on your resume. Have you had any exposure to it, even informally or in a side project?",
        "category": "gap",
        "skill_tags": ["general_gap", "missing_skill"],
        "experience_level": "any",
        "what_to_listen_for": "Honesty about the gap, any adjacent knowledge, genuine curiosity about learning it — not false confidence."
    },
    {
        "id": "gap_002",
        "text": "You have strong experience with [MATCHED_SKILL], but not [MISSING_SKILL]. How do you see yourself bridging that gap if you joined?",
        "category": "gap",
        "skill_tags": ["general_gap", "learning_plan", "adjacent_skills"],
        "experience_level": "any",
        "what_to_listen_for": "Concrete learning plan, timeline estimation, whether they've already started, humility."
    },
    {
        "id": "gap_003",
        "text": "Docker and containerization are central to how we deploy models here. You haven't listed it — have you worked in a containerized environment at all?",
        "category": "gap",
        "skill_tags": ["docker", "containerization", "deployment", "gap"],
        "experience_level": "any",
        "what_to_listen_for": "Exposure through teammates, exposure to virtual environments, understanding of why containers exist."
    },
    {
        "id": "gap_004",
        "text": "We use Kubernetes for orchestrating our model serving. Have you worked at the infrastructure level before, or has that always been handled by a separate team?",
        "category": "gap",
        "skill_tags": ["kubernetes", "mlops", "infrastructure", "gap"],
        "experience_level": "mid",
        "what_to_listen_for": "Honest about what they have and haven't owned. Shows curiosity about learning it rather than dismissing it."
    },
    {
        "id": "gap_005",
        "text": "PyTorch is our primary framework and you listed TensorFlow — how comfortable are you switching, and how long do you think it would take you to be productive?",
        "category": "gap",
        "skill_tags": ["pytorch", "tensorflow", "deep_learning", "framework_gap"],
        "experience_level": "any",
        "what_to_listen_for": "Understanding that concepts transfer, realistic timeline, has already looked at PyTorch or is willing to."
    },
    {
        "id": "gap_006",
        "text": "The role involves building and maintaining data pipelines, but I see mostly model-building experience on your resume. How have you interfaced with data engineering work in the past?",
        "category": "gap",
        "skill_tags": ["data_engineering", "pipeline", "etl", "gap"],
        "experience_level": "any",
        "what_to_listen_for": "Any ETL work, awareness of upstream data quality issues, experience debugging pipelines others wrote."
    },
    {
        "id": "gap_007",
        "text": "We work with large-scale distributed systems and I don't see Spark or distributed computing on your resume. What is the largest scale you have worked at?",
        "category": "gap",
        "skill_tags": ["spark", "distributed_computing", "big_data", "gap"],
        "experience_level": "mid",
        "what_to_listen_for": "Honest about scale. If they've worked with chunked Pandas on big data, that shows adjacent awareness."
    },
    {
        "id": "gap_008",
        "text": "This position requires owning the full ML lifecycle including monitoring and retraining. I see model building but not MLOps experience — how have your models been maintained after deployment?",
        "category": "gap",
        "skill_tags": ["mlops", "model_monitoring", "retraining", "gap"],
        "experience_level": "mid",
        "what_to_listen_for": "Awareness that models degrade, even if someone else owned the ops. Interest in taking on more ownership."
    },
    {
        "id": "gap_009",
        "text": "You haven't listed cloud experience. Our entire stack runs on [CLOUD_PROVIDER] — have you used any managed services, even outside of work?",
        "category": "gap",
        "skill_tags": ["aws", "gcp", "azure", "cloud", "gap"],
        "experience_level": "any",
        "what_to_listen_for": "Personal projects on cloud, exposure through team, willingness to learn. Cloud concepts transfer quickly."
    },
    {
        "id": "gap_010",
        "text": "The role involves fine-tuning large language models, which I don't see on your resume. How familiar are you with the concepts around LLM fine-tuning?",
        "category": "gap",
        "skill_tags": ["llm", "fine_tuning", "nlp", "transformers", "gap"],
        "experience_level": "mid",
        "what_to_listen_for": "LoRA, PEFT, instruction tuning, dataset formatting — even theoretical knowledge shows engagement with the field."
    },
    {
        "id": "gap_011",
        "text": "I see backend development on your resume but the role also needs frontend work. Have you built any user-facing interfaces, even simple ones?",
        "category": "gap",
        "skill_tags": ["frontend", "javascript", "react", "fullstack", "gap"],
        "experience_level": "any",
        "what_to_listen_for": "Even dashboard work, Streamlit, Gradio, or basic HTML shows willingness to cross the stack."
    },
    {
        "id": "gap_012",
        "text": "SQL is critical for this role and I see it listed but without much detail. Can you describe the most complex query or schema you have written?",
        "category": "gap",
        "skill_tags": ["sql", "databases", "gap", "depth_check"],
        "experience_level": "any",
        "what_to_listen_for": "Listing a skill vs actually using it deeply. Joins, subqueries, indexes, transactions — what's their actual depth?"
    },
    {
        "id": "gap_013",
        "text": "This team does a lot of A/B testing and statistical analysis. I don't see that on your resume — are you comfortable designing and interpreting experiments?",
        "category": "gap",
        "skill_tags": ["statistics", "ab_testing", "experimentation", "gap"],
        "experience_level": "mid",
        "what_to_listen_for": "p-values, power analysis, sample size, Bonferroni correction, novelty effect. Conceptual awareness vs lived experience."
    },
    {
        "id": "gap_014",
        "text": "We write production code collaboratively with code reviews and CI/CD. Your experience looks more research-oriented — have you worked in a structured engineering workflow before?",
        "category": "gap",
        "skill_tags": ["software_engineering", "ci_cd", "code_review", "research_to_prod", "gap"],
        "experience_level": "any",
        "what_to_listen_for": "Acknowledges the difference between research and production code. Shows interest in engineering rigor."
    },
    {
        "id": "gap_015",
        "text": "You have not listed any experience with model interpretability tools like SHAP or LIME. Is that something you have worked with or thought about?",
        "category": "gap",
        "skill_tags": ["xai", "model_interpretability", "shap", "machine_learning", "gap"],
        "experience_level": "mid",
        "what_to_listen_for": "Awareness that black-box models are a concern, even basic knowledge of feature attribution methods."
    },

    # ================================================================
    # CATEGORY: behavioral
    # Situational questions relevant to an ML/engineering role.
    # ================================================================

    {
        "id": "beh_001",
        "text": "Tell me about a time your machine learning model performed well in offline evaluation but failed in production. What happened and what did you do?",
        "category": "behavioral",
        "skill_tags": ["production", "debugging", "mlops", "accountability"],
        "experience_level": "mid",
        "what_to_listen_for": "Distribution shift, evaluation metric mismatch, production data pipeline bugs. Systematic fix, not just a retry."
    },
    {
        "id": "beh_002",
        "text": "Describe a time you disagreed with a technical decision made by your team or manager. How did you handle it and what was the outcome?",
        "category": "behavioral",
        "skill_tags": ["communication", "teamwork", "professional_skills"],
        "experience_level": "any",
        "what_to_listen_for": "Evidence-based argument, open to being wrong, escalated appropriately, did not just comply without voicing concern."
    },
    {
        "id": "beh_003",
        "text": "Tell me about a time you had to deliver something under a tight deadline with incomplete information. How did you prioritize and what did you leave out?",
        "category": "behavioral",
        "skill_tags": ["prioritization", "time_management", "decision_making"],
        "experience_level": "any",
        "what_to_listen_for": "Clear triage logic, communicated tradeoffs to stakeholders, did not just work longer hours as the only answer."
    },
    {
        "id": "beh_004",
        "text": "Tell me about a time a project you were working on was cancelled or de-prioritized mid-way. How did you react?",
        "category": "behavioral",
        "skill_tags": ["resilience", "adaptability", "professional_skills"],
        "experience_level": "any",
        "what_to_listen_for": "Professionalism, ability to pivot without bitterness, extracted learning from the work done."
    },
    {
        "id": "beh_005",
        "text": "Describe a situation where you had to learn a new technology or domain quickly to unblock your team. How did you approach it?",
        "category": "behavioral",
        "skill_tags": ["learning", "adaptability", "self_directed"],
        "experience_level": "any",
        "what_to_listen_for": "Structured approach to learning, knowing when to ask for help vs push forward alone, time-boxed spikes."
    },
    {
        "id": "beh_006",
        "text": "Tell me about a time you made a significant mistake at work. What was it, how did you handle it, and what changed afterwards?",
        "category": "behavioral",
        "skill_tags": ["accountability", "learning", "professional_skills"],
        "experience_level": "any",
        "what_to_listen_for": "Takes direct ownership without over-explaining, concrete change in behavior after, communicated proactively."
    },
    {
        "id": "beh_007",
        "text": "Describe a time you had to collaborate with someone whose working style was very different from yours. What adjustments did you make?",
        "category": "behavioral",
        "skill_tags": ["teamwork", "communication", "adaptability"],
        "experience_level": "any",
        "what_to_listen_for": "Shows flexibility and empathy, not just tolerance. Did not just avoid the person or work around them."
    },
    {
        "id": "beh_008",
        "text": "Tell me about a time you had to balance multiple projects simultaneously. How did you decide what to work on first?",
        "category": "behavioral",
        "skill_tags": ["prioritization", "time_management", "organization"],
        "experience_level": "mid",
        "what_to_listen_for": "Explicit prioritization method (impact, deadline, dependencies), communicated trade-offs to stakeholders."
    },
    {
        "id": "beh_009",
        "text": "Describe a time you received critical feedback on your work. How did you respond and what did you do with it?",
        "category": "behavioral",
        "skill_tags": ["feedback", "growth", "professional_skills"],
        "experience_level": "any",
        "what_to_listen_for": "Did not get defensive, asked clarifying questions, acted on it concretely. Distinguishes useful vs noise feedback."
    },
    {
        "id": "beh_010",
        "text": "Tell me about a time you identified a risk or problem that no one else had noticed. How did you raise it and what happened?",
        "category": "behavioral",
        "skill_tags": ["proactivity", "communication", "ownership"],
        "experience_level": "any",
        "what_to_listen_for": "Raised it with evidence not just intuition, proposed a solution alongside the problem, not just alarming."
    },
    {
        "id": "beh_011",
        "text": "Describe the most ambiguous project you have worked on — where requirements were unclear. How did you move it forward?",
        "category": "behavioral",
        "skill_tags": ["ambiguity", "decision_making", "ownership", "communication"],
        "experience_level": "mid",
        "what_to_listen_for": "Clarifying questions first, making reasonable assumptions explicit, checkpoints with stakeholders, not paralysis."
    },
    {
        "id": "beh_012",
        "text": "Tell me about a time you advocated for a technical investment that was hard to justify to business stakeholders. How did you make the case?",
        "category": "behavioral",
        "skill_tags": ["communication", "stakeholder_management", "technical_leadership"],
        "experience_level": "senior",
        "what_to_listen_for": "Translated technical need into business risk or cost. Used data where possible. Shows business awareness."
    },
    {
        "id": "beh_013",
        "text": "Describe a time you had to give difficult feedback to a peer or someone junior to you. How did you approach the conversation?",
        "category": "behavioral",
        "skill_tags": ["communication", "leadership", "mentoring", "professional_skills"],
        "experience_level": "senior",
        "what_to_listen_for": "Specific, private, timely feedback. Focused on behavior not personality. Followed up to see if it landed."
    },
    {
        "id": "beh_014",
        "text": "Tell me about a time you had to switch approaches mid-project because something was not working. What triggered the switch and how did you manage it?",
        "category": "behavioral",
        "skill_tags": ["adaptability", "decision_making", "problem_solving"],
        "experience_level": "any",
        "what_to_listen_for": "Clear signal that triggered the pivot (not just gut feel), communicated the change, did not sunk-cost the original plan."
    },
    {
        "id": "beh_015",
        "text": "Describe a time you worked with a cross-functional team — product, design, data, or business. What made it work or not work?",
        "category": "behavioral",
        "skill_tags": ["collaboration", "communication", "cross_functional"],
        "experience_level": "mid",
        "what_to_listen_for": "Awareness of different perspectives and priorities, shared vocabulary, early alignment on success metrics."
    },
]


# ================================================================
# METADATA HELPERS — used by the retriever to filter by category
# ================================================================

def get_questions_by_category(category: str) -> list[dict]:
    return [q for q in QUESTION_BANK if q["category"] == category]


def get_all_texts() -> list[str]:
    return [q["text"] for q in QUESTION_BANK]


def get_all_ids() -> list[str]:
    return [q["id"] for q in QUESTION_BANK]


def get_all_metadatas() -> list[dict]:
    return [
        {
            "category": q["category"],
            "skill_tags": ",".join(q["skill_tags"]),
            "experience_level": q["experience_level"],
            "what_to_listen_for": q["what_to_listen_for"],
        }
        for q in QUESTION_BANK
    ]


# ================================================================
# QUICK TEST — run this file directly to see counts and a sample
# ================================================================

if __name__ == "__main__":
    from collections import Counter

    counts = Counter(q["category"] for q in QUESTION_BANK)
    print(f"Total questions: {len(QUESTION_BANK)}")
    print("By category:")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")

    print("\nSample (first of each category):")
    seen = set()
    for q in QUESTION_BANK:
        if q["category"] not in seen:
            seen.add(q["category"])
            print(f"\n[{q['category'].upper()}] {q['id']}")
            print(f"  Q: {q['text'][:80]}...")
            print(f"  Listen for: {q['what_to_listen_for'][:80]}...")
