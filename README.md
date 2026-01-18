
#  IVF Trigger Day MLOps Pipeline

An end-to-end **MLOps data pipeline** for IVF Trigger Day prediction, focused on **data ingestion, data quality validation using Great Expectations, and preprocessing** with production-style best practices.

##  Project Overview 

In real-world ML systems, **data quality issues are the #1 cause of model failure**.
This project demonstrates how to:

* Ingest IVF clinical data
* Validate raw data using **Great Expectations**
* Enforce **data quality gates**
* Prepare clean, validated data for downstream ML models
* Follow **industry-standard MLOps structure**

##  Project Structure

ivf_trigger_day_mlops/
│
├── data/
│   ├── raw/                # Raw input dataset (not committed)
│   └── processed/          # Cleaned data after preprocessing
│
├── gx/                     # Great Expectations configuration
│   ├── expectations/
│   │   ├── raw_trigger_day_suite.json
│   │   └── trigger_day_expectations.json
│   ├── validations/
│   └── great_expectations.yml
│
├── src/
│   ├── data_ingestion.py       # Loads raw dataset
│   ├── data_validation.py      # Validates data using Great Expectations
│   ├── add_expectations.py     # Adds expectations programmatically
│   ├── create_ge_datasource.py # Creates GE datasource
│   ├── add_ge_asset.py         # Registers data asset
│   └── data_preprocessing.py   # Data cleaning & transformation
│
├── main.py                 # Pipeline orchestrator
├── .gitignore
├── README.md
└── requirements.txt

##  Pipeline Workflow

### 1) Data Ingestion

* Reads raw IVF Trigger Day dataset from `data/raw`
* Performs basic checks
* push to mysql database
* again loaded to pandas dataframe
* Confirms schema & columns
 File: `src/data_ingestion.py`


### 2️) Data Validation (Great Expectations)

* Uses **Great Expectations** to validate:

  * Column existence
  * Data types
  * Value ranges
  * Null constraints
  * Business logic rules
* Generates **Data Docs (web-based validation reports)**

#   Files:

* `src/data_validation.py`
* `gx/expectations/*.json`

# Output:

* Validation success/failure
* Interactive HTML Data Docs

### 3️) Data Preprocessing

* Runs **only if validation passes**
* Cleans data
* Prepares dataset for ML training

# File: `src/data_preprocessing.py`


### 4️) Pipeline Orchestration

All steps are orchestrated using a single entry point:

📄 **`main.py`**

```python
python main.py
```

Pipeline automatically:

* Stops if validation fails
* Proceeds only with high-quality data

---

## 🌐 Great Expectations Data Docs

Data Docs provide a **visual validation report**.

### How to open Data Docs:

```powershell
great_expectations docs build
```

Then open the generated HTML file from:

```
gx/uncommitted/data_docs/local_site/index.html
```

##  Tech Stack

* **Python**
* **Great Expectations**
* **Pandas**
* **MLOps best practices**
* **Git & GitHub**

##  Why This Project Matters

✔ Industry-style MLOps pipeline
✔ Data quality enforcement before ML
✔ Production-ready structure
✔ Interview-ready explanation
✔ Scalable for CI/CD & cloud deployment


##  Example Dataset Columns

* `Patient_ID`
* `Age`
* `AMH (ng/mL)`
* `Avg_Follicle_Size_mm`
* `Trigger_Recommended (0/1)`
* `BMI`
* `AFC`
* `Visit_Date`


##  Data Privacy

* Raw data is **excluded from Git**
* `.gitignore` prevents sensitive data leaks
* Designed following **data governance best practices**


##  Future Enhancements

*  MLflow experiment tracking
*  Model training & evaluation
*  GitHub Actions CI pipeline
*  Dockerization
*  Cloud deployment (AWS / Azure)

---

## 👨 Author

**Abhishek Magadum**
Computer Science Engineer | MLOps & Data Engineering Enthusiast


##  How to Run

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py
```

---

##  Status

# Data ingestion complete
# Data validation with Great Expectations
# Data Docs generated
# GitHub-ready MLOps project
