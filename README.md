# DTSC303--Cryptocurrency-Project
# Cryptocurrency Fraud Detection using Big Data & Machine Learning
DTSC303 Big Data Computing Project  
Team: Anshi Doshi, Dhwaj Jain, Maanya Raichura, Prakarti  
Date: April 22, 2026


Overview

This project implements a distributed big data pipeline for detecting fraudulent cryptocurrency transactions using the Elliptic++ Bitcoin dataset. The pipeline combines:

- Big Data Tools: Apache Spark, MapReduce, AWS EMR, S3, YARN
- Machine Learning: Random Forest, Logistic Regression, K-Means Clustering
- Graph Analytics: Transaction network analysis, wallet behavior profiling
- Distributed Computing: Multi-node cluster processing with optimized resource management

 Key Features

✅ Real-time fraud detection on 203,769 Bitcoin transactions  
✅ Network analysis of 2.8M+ wallet connections  
✅ Temporal fraud pattern identification across 49 time steps  
✅ Scalable architecture handling imbalanced datasets (9.2:1 licit-to-illicit ratio)  
✅ High accuracy: Random Forest achieves 98% accuracy, 0.786 AUC-PR

---

Dataset
Elliptic++ Bitcoin Dataset

Source: Georgia Tech / ACM SIGKDD 2023  
Location: AWS S3 (see configuration section)

The dataset contains 9 CSV files with Bitcoin transaction and wallet data:

| File | Rows | Description |
|------|------|-------------|
| `txs_classes.csv` | 203,769 | Transaction IDs with class labels (illicit/licit/unknown) |
| `txs_edgelist.csv` | 234,355 | Transaction-to-transaction connections (graph edges) |
| `txs_features.csv` | 203,769 | 184 features per transaction + time step |
| `wallets_classes.csv` | 822,942 | Wallet addresses with class labels |
| `wallets_features.csv` | 203,769 | Wallet behavior features |
| `AddrAddr_edgelist.csv` | 2,868,964 | Wallet-to-wallet connections |
| `AddrTx_edgelist.csv` | 477,117 | Input wallet → transaction links |
| `TxAddr_edgelist.csv` | 837,127 | Transaction → output wallet links |
| `wallets_features_classes_combined.csv` | - | Combined wallet data |

Data Characteristics:
- Class Distribution: 77.1% Unknown, 20.6% Licit, 2.3% Illicit
- Temporal Range: 49 time steps
- Graph Structure: 203,769 nodes, 234,355 edges (transaction graph)
- Imbalance Ratio: 9.2:1 (licit: illicit among labeled data)

---

 🏗️ Architecture

 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AWS S3 Bucket                          │
│         (Raw Data Storage & Output Repository)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│                    AWS EMR Cluster                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Master Node  │  │ Core Node 1  │  │ Core Node 2  │      │
│  │              │  │              │  │              │      │
│  │ YARN RM      │  │ Executor     │  │ Executor     │      │
│  │ Spark Driver │  │ (4GB RAM)    │  │ (4GB RAM)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                            │
│           HDFS (Intermediate Storage & Shuffle)            │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Pipeline                        │
│                                                               │
│  1. Data Loading (S3 → Spark DataFrames)                    │
│  2. Class Distribution (MapReduce)                           │
│  3. Graph Degree Analysis (MapReduce)                        │
│  4. Wallet Network Analysis (Spark SQL)                      │
│  5. Temporal Analysis (Window Functions)                     │
│  6. ML Fraud Detection (Random Forest + Logistic Reg)        │
│  7. Wallet Clustering (K-Means)                              │
│  8. Case Analysis (Easy/Hard/Average)                        │
│  9. YARN Performance Monitoring                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Results                           │
│  • CSV Tables → S3                                           │
│  • PNG Charts → S3                                           │
│  • Performance Metrics → YARN Logs                           │
└─────────────────────────────────────────────────────────────┘
```

Prerequisites

 Software Requirements

- Python: 3.8 or higher
- Java: 8 or 11 (required for Spark)
- Apache Spark: 3.x
- AWS Account with EMR and S3 access
- Git (for cloning the repository)

 AWS Services

- Amazon EMR (Elastic MapReduce) - Version 6.x or higher
- Amazon S3 - For data storage
- AWS CLI - Configured with appropriate credentials

 Hardware Recommendations

For Local Development/Testing:
- RAM: 16GB+ recommended
- CPU: 4+ cores
- Storage: 10GB free space

For Production (EMR Cluster):
- Master Node: m5.xlarge or equivalent
- Core Nodes: 2× m5.xlarge (minimum)
- Storage: EBS with 50GB+ per node

---

 Installation
# Manual Installation

```bash
pip install pyspark==3.5.0
pip install boto3==1.34.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
```

 ☁️ AWS Setup
 Dataset location in S3
```
s3://my-flame-emr-bucket/cryptocurrency team/trial2/
```

Expected Files in S3:
```
s3://my-flame-emr-bucket/cryptocurrency team/trial2/
├── txs_classes.csv
├── txs_edgelist.csv
├── txs_features.csv
├── wallets_classes.csv
├── wallets_features.csv
├── wallets_features_classes_combined.csv
├── AddrAddr_edgelist.csv
├── AddrTx_edgelist.csv
└── TxAddr_edgelist.csv
```
Running the Pipeline

SSH + spark-submit (Recommended)

```bash
# SSH into EMR Master Node
ssh -i your-key.pem hadoop@<EMR_MASTER_DNS>

# Run the pipeline
spark-submit main.py
```

 Methodology

 Data Processing Pipeline

# Stage 1: Data Loading
- Load 9 CSV files from S3 into Spark DataFrames
- Apply schema inference and validation
- Cache frequently accessed tables
- Runtime: ~50 seconds

# Stage 2: Class Distribution (MapReduce - PS-1)
- Map: Transform each transaction to `(class, 1)` key-value pair
- Reduce: Aggregate counts per class
- Generate pie charts and bar charts
- Output: Class distribution statistics
- Runtime: ~4 seconds

# Stage 3: Graph Degree Analysis (MapReduce - PS-2)
- Map: Extract node degrees from edgelist
- Reduce: Compute in-degree and out-degree distributions
- Identify hub nodes (high-degree transactions)
- Finding: 5/20 top hubs are illicit
- Runtime: ~6 seconds

# Stage 4: Wallet Network Analysis (Spark SQL - PS-3)
- Join `AddrAddr_edgelist` with `wallets_classes`
- Analyze illicit-to-licit wallet connections
- Identify money laundering patterns
- Finding: One illicit wallet made 840 transfers to 399 unique addresses
- Runtime: ~15 seconds

# Stage 5: Temporal Analysis (Window Functions - PS-5)
- Apply sliding window over 49 time steps
- Compute rolling fraud rates
- Detect temporal fraud bursts
- Finding: Fraud rate peaks at 2.6% around step 32-33
- Runtime: ~7 seconds

# Stage 6: ML Fraud Detection (Spark MLlib - PS-4)
- Models: Random Forest (50 trees), Logistic Regression
- Validation: 5-fold cross-validation
- Features: 184 transaction features
- Train-Test Split: Time-based (steps 1-34 train, 35-49 test)
- Handling Imbalance: Stratified sampling
- Runtime: ~406 seconds (67% of total pipeline)

Model Performance:

| Model | Accuracy | F1-Score | AUC-PR | Precision | Recall |
|-------|----------|----------|--------|-----------|--------|
| Random Forest | 0.980 | 0.979 | 0.786 | 0.974 | 0.722 |
| Logistic Regression | 0.864 | 0.889 | 0.363 | 0.288 | 0.719 |

# Stage 7: Wallet Clustering (K-Means)
- Cluster 822,942 wallets into 4 behavioral groups
- Features: in-degree, out-degree, txs sent/received
- Cluster 0: 815,270 low-activity wallets
- Clusters 1-3: High-activity hubs (exchanges, mixing services)
- Runtime: ~75 seconds

# Stage 8: Case Analysis
- Categorize fraud into Easy/Hard/Average cases
- Easy (24.8%): Surrounded by illicit neighbors
- Hard (67.7%): Surrounded by licit neighbors (structurally invisible)
- Average (7.5%): Mixed neighborhood
- Runtime: ~2 seconds

# Stage 9: YARN Performance Monitoring (PS-6)
- Collect resource utilization metrics
- Profile bottlenecks (shuffle operations, broadcasting)
- Generate performance visualizations

---

 Results

 Key Findings

1. Class Imbalance Challenge
   - 77.1% of transactions are unlabeled
   - Among labeled data: 9.2:1 licit-to-illicit ratio
   - Standard accuracy metrics are misleading → Use AUC-PR instead

2. Network Structure Insights
   - Illicit transactions hide among legitimate high-volume nodes
   - 5 out of top 20 most-connected transactions are fraudulent
   - Average node degree: 2.3 | Max degree: 473

3. Temporal Patterns
   - Fraud is not uniformly distributed over time
   - Clear spikes at specific time steps (coordinated attacks)
   - Cumulative fraud rate rises from 0.22% → 2.6%

4. The Hard Fraud Problem
   - 67.7% of illicit transactions are surrounded entirely by licit neighbors
   - Graph-based rules alone cannot detect these cases
   - Requires full 184-dimensional feature space analysis

5. Model Superiority
   - Random Forest outperforms Logistic Regression across all metrics
   - High precision (0.974) → Very few false alarms
   - AUC-PR of 0.786 demonstrates effective fraud ranking

Output Files

All outputs are saved to S3:

CSV Tables: `s3://my-flame-emr-bucket/cryptocurrency team/trial2/output/csv/`
- `01_class_distribution.csv`
- `02_graph_degree_stats.csv`
- `03_top_high_degree_nodes.csv`
- `04_wallet_network_summary.csv`
- `05_illicit_hubs.csv`
- `06_temporal_fraud_stats.csv`
- `07_ml_model_comparison.csv`
- `08_confusion_matrices.csv`
- `09_feature_importance.csv`
- `10_wallet_cluster_profiles.csv`
- `11_case_analysis_summary.csv`

PNG Charts: `s3://my-flame-emr-bucket/cryptocurrency team/trial2/output/charts/`
- `01_class_distribution_pie.png`
- `02_class_distribution_bar.png`
- `03_graph_degree_distribution.png`
- `04_wallet_network_illicit_hubs.png`
- `05_temporal_fraud_trends.png`
- `06_cumulative_fraud_rate.png`
- `07_model_performance_comparison.png`
- `08_confusion_matrix_rf.png`
- `09_confusion_matrix_lr.png`
- `10_feature_importance_top20.png`
- `11_roc_curves.png`
- `12_precision_recall_curves.png`
- `13_wallet_clustering_pca.png`
- `14_wallet_cluster_profiles.png`
- `15_cluster_illicit_vs_licit.png`
- `16_case_analysis.png`
- `17_pipeline_overview.png`

 Download Results

```bash
# Download all CSV results
aws s3 sync s3://my-flame-emr-bucket/cryptocurrency\ team/trial2/output/csv/ ./output/csv/

# Download all charts
aws s3 sync s3://my-flame-emr-bucket/cryptocurrency\ team/trial2/output/charts/ ./output/charts/
```

---

Performance

 Execution Metrics

Total Pipeline Runtime: 605.6 seconds (~10.1 minutes)

Stage-wise Breakdown:

| Stage | Description | Runtime (s) | % of Total |
|-------|-------------|-------------|------------|
| 1 | Data Loading (S3 → Spark) | 49.5 | 8.2% |
| 2 | MapReduce: Class Distribution | 4.4 | 0.7% |
| 3 | MapReduce: Graph Degree | 5.6 | 0.9% |
| 4 | Spark SQL: Wallet Network | 14.6 | 2.4% |
| 5 | Window Functions: Temporal | 6.9 | 1.1% |
| 6 | ML: Fraud Detection (5-Fold CV) | 406.4 | 67.1% |
| 7 | ML: Wallet Clustering (K-Means) | 75.1 | 12.4% |
| 8 | Case Analysis | 1.9 | 0.3% |
| 9 | YARN Performance Summary | <1 | <0.1% |

 Resource Utilization

Spark Configuration:
```
Executor Memory:     4 GB
Driver Memory:       4 GB
Shuffle Partitions:  16
Default Parallelism: 20
Serialization:       Kryo
```

Cluster Specs:
- Master Node: 1× m5.xlarge (4 vCPUs, 16 GB RAM)
- Core Nodes: 2× m5.xlarge (4 vCPUs, 16 GB RAM each)
- Total Resources: 12 vCPUs, 48 GB RAM

Bottlenecks Identified:
1. Model Broadcasting (Stage 6): Large Random Forest model binaries (1-4 MB) broadcast to executors
2. Shuffle Operations: Join-heavy operations in Stage 4
3. S3 Latency: Initial data load from S3

Optimizations Applied:
- ✅ Caching: Frequently accessed DataFrames cached in memory
- ✅ Partition Tuning: 16 shuffle partitions balanced for a 2-node cluster
- ✅ Adaptive Query Execution (AQE): Runtime join optimization enabled
- ✅ Kryo Serialization: Reduced memory footprint vs Java serialization


 Dataset & Code

- GitHub Repository: [https://github.com/dhwxj-jxin/DTSC303--Cryptocurrency-Project](https://github.com/dhwxj-jxin/DTSC303--Cryptocurrency-Project)
- Dataset Source: Elliptic++ (Georgia Tech), https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l 

Project Team:
- Anshi Doshi - anshi.doshi@flame.edu.in
- Dhwaj Jain - dhwaj.jain@flame.edu.in
- Maanya Raichura - maanya.raichura@flame.edu.in
- Prakarti - prakarti@flame.edu.in

Course: DTSC303 - Big Data Computing  
Institution: FLAME University
Submission Date: April 22, 2026


Version History

- v1.0.0 (2026-04-22) - Initial release
  - Complete pipeline implementation
  - All 9 stages operational
  - Full documentation
  - 17 visualization charts
  - 11 CSV output tables
*Last Updated: April 22, 2026*
